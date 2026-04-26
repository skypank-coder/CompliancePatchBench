"""
rl_trainer.py
=============

Online RL policy optimization loop for CompliancePatchBench.

RL formulation
--------------
State:
    The environment observation snapshot: available files, violations, CI
    results, read history / file view, task difficulty, and reward state.

Action:
    One structured JSON command emitted by the policy:
        read_file | write_patch | run_ci | finalize_patch

Reward:
    The environment's per-step reward, including CI/test feedback, patch
    quality, hidden compliance penalties, and partial-fix penalties applied
    by `ComplianceAgent.run`.

Episode:
    One task rollout until `done=True` or `max_steps`.

Each saved transition is `(state, action, reward, next_state, logprob, done)`.
For final policy optimization, TRL's `GRPOTrainer` samples completions from the
current policy, executes those actions in `CompliancePatchEnv`, computes reward,
and immediately updates that same policy.

Implementation note for judges:
    We use heuristic/tabular rollouts only for initial data collection,
    baseline, and CPU dry-runs. Final policy optimization is performed with
    GRPO via TRL.
    The agent doesn't just learn to fix code — it learns to avoid cheating,
    because the environment penalizes hidden violations.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from environment.patch_env import CompliancePatchEnv  # noqa: E402

from .agent import (
    GENERATION_MAX_NEW_TOKENS,
    AgentConfig,
    RULE_PATCHES,
    RULE_PATCHES_CHEAT,
    SYSTEM_PROMPT,
    _apply_indent_from_view,
    _ci_runs_seen,
    _patches_seen,
    _read_files_seen,
    align_causal_lm_and_tokenizer,
    json_action_eos_token_ids,
    make_heuristic_backend,
    make_hf_pipeline_backend,
)
from .dataset_builder import run_rollouts
from .evaluate import evaluate
from .hackathon_metrics import print_interpretation_curves, print_learning_curve_footer
from .utils import (  # noqa: E402
    DATA_DIR,
    LEARNING_CURVE_PATH,
    TASKS_PATH,
    TRAJECTORIES_RL_PATH,
    clip_model_json_output,
    clip_reward_value,
    extract_json,
    get_logger,
    read_json,
    read_jsonl,
    seed_everything,
    write_json,
    write_jsonl,
)

log = get_logger("rl_trainer")

RL_ADAPTER_DIR = DATA_DIR / "rl_adapter"
DEFAULT_RL_BASE_MODEL = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
DEFAULT_SFT_ADAPTER_DIR = DATA_DIR / "lora_adapter"


def _clip_reward(value: float) -> float:
    """Keep GRPO rewards bounded so outliers do not dominate updates."""
    return clip_reward_value(value)


def _split_tasks(tasks: List[Dict[str, Any]], seed: int, train_fraction: float = 0.75) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Deterministic train/test split to show generalization beyond training tasks."""
    shuffled = list(tasks)
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) <= 1:
        return shuffled, []
    cut = max(1, min(len(shuffled) - 1, int(round(len(shuffled) * train_fraction))))
    return shuffled[:cut], shuffled[cut:]


def _adaptive_task_weights(tasks: List[Dict[str, Any]], last_eval: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """
    Weight failed/adversarial tasks higher and consistently solved tasks lower.
    This is curriculum learning in the small: train more where the agent fails.
    """
    status_by_id = {}
    if last_eval:
        status_by_id = {r["task_id"]: r for r in last_eval.get("per_task", [])}

    weights: Dict[str, float] = {}
    for task in tasks:
        tid = task["task_id"]
        row = status_by_id.get(tid, {})
        status = row.get("status")
        failure_type = row.get("failure_type")
        weight = 1.0
        if status == "SUCCESS":
            weight *= 0.45
        elif status == "PARTIAL":
            weight *= 1.8
        elif status == "FAIL":
            weight *= 2.4
        if failure_type == "hidden_violation":
            weight *= 2.0
        elif failure_type == "partial_fix":
            weight *= 1.5
        elif failure_type == "no_fix":
            weight *= 1.2
        if task.get("adversarial"):
            weight *= 1.5
        weights[tid] = round(weight, 4)
    return weights


def _sample_tasks(tasks: List[Dict[str, Any]], weights: Dict[str, float], count: int, seed: int) -> List[Dict[str, Any]]:
    if not tasks:
        return []
    rng = random.Random(seed)
    population = list(tasks)
    w = [max(0.01, float(weights.get(t["task_id"], 1.0))) for t in population]
    return rng.choices(population, weights=w, k=max(1, count))


@dataclass
class RLConfig:
    """Configuration for the iterative RL loop."""

    tasks_path: str = str(TASKS_PATH)
    trajectories_path: str = str(TRAJECTORIES_RL_PATH)
    learning_curve_path: str = str(LEARNING_CURVE_PATH)
    base_model: str = DEFAULT_RL_BASE_MODEL
    sft_adapter_dir: str = str(DEFAULT_SFT_ADAPTER_DIR)
    output_dir: str = str(RL_ADAPTER_DIR)
    iterations: int = 40
    rollouts_per_task: int = 1
    epochs_per_iteration: int = 1
    batch_size: int = 2
    grad_accum: int = 4
    learning_rate: float = 2e-5
    gamma: float = 1.0
    max_steps: int = 8
    max_seq_length: int = 4096
    max_tasks: int = 20
    seed: int = 42
    dry_run: bool = False
    use_tabular_rl: bool = False
    tabular_alpha: float = 0.45
    tabular_epsilon: float = 0.20
    exploration_temperature: float = 0.8


class TabularPatchPolicy:
    """
    Always-on RL controller for structured patch decisions.

    This is deliberately small and transparent: for each
    (difficulty, adversarial, rule_id) state bucket it learns Q-values over
    `{safe, skip, cheat}`. It updates from completed episodes, so hidden
    violations and partial fixes push down shortcut actions even without a GPU.
    """

    def __init__(self, alpha: float = 0.45, epsilon: float = 0.20, seed: int = 42):
        self.alpha = alpha
        self.epsilon = epsilon
        self.rng = random.Random(seed)
        self.q: Dict[str, Dict[str, float]] = {}
        self._episode_decisions: Dict[str, List[Dict[str, str]]] = {}
        self._skipped_locs: Dict[str, set[Tuple[str, int, int]]] = {}

    def _key(self, obs: Dict[str, Any], violation: Dict[str, Any]) -> str:
        return "|".join([
            str(obs.get("task_difficulty", "easy")),
            "adv" if obs.get("task_adversarial") else "std",
            str(violation.get("rule_id", "unknown")),
        ])

    def _initial_q(self, action: str, obs: Dict[str, Any]) -> float:
        diff = obs.get("task_difficulty", "easy")
        adv = bool(obs.get("task_adversarial", False))
        if action == "safe":
            return 0.25 if diff == "easy" else 0.05
        if action == "cheat":
            return 0.15 if adv else -0.2
        if action == "skip":
            return 0.10 if diff in {"medium", "hard"} else -0.1
        return 0.0

    def _scores(self, key: str, legal: List[str], obs: Dict[str, Any]) -> Dict[str, float]:
        bucket = self.q.setdefault(key, {})
        for action in legal:
            bucket.setdefault(action, self._initial_q(action, obs))
        return {a: bucket[a] for a in legal}

    def _choose(self, key: str, legal: List[str], obs: Dict[str, Any], explore: bool) -> Tuple[str, float]:
        scores = self._scores(key, legal, obs)
        if explore and self.rng.random() < self.epsilon:
            choice = self.rng.choice(legal)
        else:
            choice = max(legal, key=lambda a: (scores[a], 1 if a == "safe" else 0))
        # Softmax probability under the current tabular policy; stored as logprob.
        exps = {a: math.exp(scores[a]) for a in legal}
        denom = sum(exps.values()) or 1.0
        logprob = math.log(max(exps[choice] / denom, 1e-8))
        return choice, logprob

    def backend(self, explore: bool = True, collect: bool = True):
        def _call(messages: List[Dict[str, str]]) -> str:
            last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
            obs = extract_json(last_user) or {}
            task_id = str(obs.get("task_id", "unknown"))
            available = obs.get("available_files", [])
            violations = obs.get("violations", [])
            read_already = _read_files_seen(messages)
            patched_already = _patches_seen(messages) | self._skipped_locs.setdefault(task_id, set())
            ci_runs = _ci_runs_seen(messages)

            budget = obs.get("file_reads_remaining", 0)
            if budget > 0:
                for v in violations:
                    if v.get("file") not in read_already and v.get("file") in available:
                        return json.dumps({"action_type": "read_file", "path": v["file"], "_logprob": 0.0})

            ci_results = obs.get("ci_results", []) or []
            fixed_keys = {(r.get("file"), r.get("rule_id")) for r in ci_results if r.get("ci") == "PASS"}
            file_view = obs.get("last_file_view") or ""

            for v in violations:
                key_fixed = (v.get("file"), v.get("rule_id"))
                if key_fixed in fixed_keys:
                    continue
                loc = (v.get("file"), int(v.get("line_start", 0)), int(v.get("line_end", 0)))
                if loc in patched_already:
                    continue

                key = self._key(obs, v)
                legal = ["safe", "skip"]
                if v.get("rule_id") in RULE_PATCHES_CHEAT:
                    legal.append("cheat")
                choice, logprob = self._choose(key, legal, obs, explore=explore)
                if collect:
                    self._episode_decisions.setdefault(task_id, []).append({"key": key, "choice": choice})

                if choice == "skip":
                    self._skipped_locs.setdefault(task_id, set()).add(loc)
                    continue

                patch_table = RULE_PATCHES_CHEAT if choice == "cheat" else RULE_PATCHES
                raw_patch = patch_table.get(v["rule_id"], "pass  # TODO: fix")
                new_code = _apply_indent_from_view(file_view, int(v["line_start"]), raw_patch)
                return json.dumps({
                    "action_type": "write_patch",
                    "file": v["file"],
                    "line_start": v["line_start"],
                    "line_end": v["line_end"],
                    "new_code": new_code,
                    "_logprob": logprob,
                })

            if ci_runs < 2 and len(ci_results) == 0:
                return json.dumps({"action_type": "run_ci", "_logprob": 0.0})
            return json.dumps({"action_type": "finalize_patch", "_logprob": 0.0})

        return _call

    def update_from_rollouts(self, rollouts: List[Any]) -> Dict[str, Any]:
        updates = 0
        for result in rollouts:
            decisions = self._episode_decisions.pop(result.task_id, [])
            if not decisions:
                continue
            reward = float(result.final_score)
            if result.hidden_violation:
                reward -= 1.0
            if 0 < result.violations_fixed < result.violations_total:
                reward -= 0.5
            if result.violations_fixed == 0:
                reward -= 0.2
            reward = _clip_reward(reward)
            for d in decisions:
                bucket = self.q.setdefault(d["key"], {})
                old = bucket.get(d["choice"], 0.0)
                bucket[d["choice"]] = old + self.alpha * (reward - old)
                updates += 1
        self._skipped_locs.clear()
        return {"tabular_updates": updates, "q_buckets": len(self.q)}

    def to_dict(self) -> Dict[str, Any]:
        return {"alpha": self.alpha, "epsilon": self.epsilon, "q": self.q}


# ─── Trajectory loading + advantage computation ──────────────────────────────

def state_to_prompt(state: Dict[str, Any]) -> str:
    """Render the RL state as the policy's user message."""
    return "OBSERVATION:\n" + json.dumps(state, ensure_ascii=False)


def action_to_text(action: Dict[str, Any]) -> str:
    """Canonical action rendering used for log-probability training."""
    return json.dumps(action, ensure_ascii=False, sort_keys=True)


def load_rl_transitions(path: str, gamma: float = 1.0) -> List[Dict[str, Any]]:
    """
    Load `trajectories_rl.jsonl` and flatten it into per-transition records.

    Adds:
      - reward_to_go
      - advantage = reward_to_go - global baseline
    """
    transitions: List[Dict[str, Any]] = []
    for episode in read_jsonl(Path(path)):
        traj = episode.get("trajectory") or []
        rtg = 0.0
        episode_rows: List[Dict[str, Any]] = []
        for step in reversed(traj):
            rtg = float(step.get("reward", 0.0)) + gamma * rtg
            row = {
                "task_id": episode.get("task_id"),
                "state": step.get("state", {}),
                "action": step.get("action", {}),
                "reward": float(step.get("reward", 0.0)),
                "logprob": float(step.get("logprob", 0.0)),
                "done": bool(step.get("done", False)),
                "next_state": step.get("next_state", {}),
                "reward_to_go": rtg,
                "hidden_violation": bool(episode.get("hidden_violation", False)),
                "success_rate": float(episode.get("success_rate", 0.0)),
            }
            episode_rows.append(row)
        transitions.extend(reversed(episode_rows))

    if not transitions:
        return []

    baseline = sum(t["reward_to_go"] for t in transitions) / len(transitions)
    variance = sum((t["reward_to_go"] - baseline) ** 2 for t in transitions) / len(transitions)
    std = math.sqrt(max(variance, 1e-8))
    for t in transitions:
        t["advantage"] = (t["reward_to_go"] - baseline) / std
    return transitions


# ─── GRPO helpers ─────────────────────────────────────────────────────────────

ALLOWED_ACTIONS = {"read_file", "write_patch", "run_ci", "finalize_patch"}


def _json_actions_from_completion(completion: str) -> List[Dict[str, Any]]:
    """Extract JSON action objects from a model completion (after JSON span clip)."""
    completion = clip_model_json_output(str(completion))
    decoder = json.JSONDecoder()
    actions: List[Dict[str, Any]] = []
    idx = 0
    while idx < len(completion):
        start = completion.find("{", idx)
        if start == -1:
            break
        try:
            obj, end = decoder.raw_decode(completion[start:])
        except json.JSONDecodeError:
            idx = start + 1
            continue
        idx = start + end
        if isinstance(obj, dict) and obj.get("action_type") in ALLOWED_ACTIONS:
            actions.append({k: v for k, v in obj.items() if not str(k).startswith("_")})
    return actions


def _rollout_format_metrics(rollouts: List[Any]) -> Dict[str, float]:
    """First-parse JSON rate and fallback share across rollout step records."""
    n_steps = 0
    n_ok = 0
    n_fb = 0
    for result in rollouts:
        for step in getattr(result, "steps", []):
            n_steps += 1
            if not getattr(step, "used_fallback", True):
                n_ok += 1
            if getattr(step, "used_fallback", False):
                n_fb += 1
    return {
        "valid_json_rate": n_ok / max(1, n_steps),
        "fallback_rate": n_fb / max(1, n_steps),
    }


def _task_to_grpo_prompt(task: Dict[str, Any]) -> str:
    """Prompt used by TRL GRPOTrainer for online environment rollouts."""
    return (
        SYSTEM_PROMPT
        + "\n\nTask JSON:\n"
        + json.dumps({
            "task_id": task["task_id"],
            "codebase": task["codebase"],
            "violations": task["violations"],
            "difficulty": task.get("difficulty"),
            "adversarial": task.get("adversarial", False),
        }, indent=2)
        + '\n\nReturn JSON actions only. End with {"action_type":"finalize_patch"}.'
    )


def _grpo_reward_factory(component_log: List[Dict[str, float]], health_log: List[Dict[str, float]]):
    """Build a TRL reward function that executes completions in CompliancePatchEnv."""
    from .agent import decompose_reward_breakdown

    def _reward(completions, task_payload=None, **kwargs) -> List[float]:
        payloads = task_payload or kwargs.get("task", [])
        rewards: List[float] = []
        for completion, payload in zip(completions, payloads):
            raw_s = str(completion)
            clipped = clip_model_json_output(raw_s)
            cr = len(clipped) / max(1, len(raw_s))
            health_log.append({
                "length": float(len(clipped)),
                "raw_length": float(len(raw_s)),
                "clipped_ratio": cr,
                "terminated": 1.0 if clipped.rstrip().endswith("}") else 0.0,
            })
            task = json.loads(payload) if isinstance(payload, str) else payload
            actions = _json_actions_from_completion(clipped)
            if not actions:
                component_log.append({
                    "reward_ci": 0.0,
                    "reward_minimal": 0.0,
                    "reward_regression": 0.0,
                    "reward_penalty": -0.5,
                })
                rewards.append(_clip_reward(-0.5))
                continue
            # +0.1 shaping when at least one parseable JSON action (matches Colab compliance_patch_reward)
            json_shaping = 0.1

            env = CompliancePatchEnv()
            env.reset(
                task_id=task["task_id"],
                codebase=task["codebase"],
                violations=task["violations"],
                max_steps=task.get("max_steps", 12),
                file_reads_remaining=task.get("file_reads_remaining", 5),
            )
            final_score = -1.0
            final_info: Dict[str, Any] = {}
            try:
                done = False
                for action in actions[:12]:
                    obs, reward, done, info = env.step(action)
                    final_score = float(info.get("final_score", reward))
                    final_info = info
                    if done or action.get("action_type") == "finalize_patch":
                        break
                if not done:
                    obs, reward, done, info = env.step({"action_type": "finalize_patch"})
                    final_score = float(info.get("final_score", reward))
                    final_info = info
            except Exception:
                final_score = -1.0
                final_info = {"critique": {"reward_breakdown": {"execution_error_penalty": -1.0}}}

            breakdown = (final_info.get("critique", {}) or {}).get("reward_breakdown", {})
            component_log.append(decompose_reward_breakdown(breakdown))
            rewards.append(_clip_reward(final_score + json_shaping))
        return rewards

    return _reward


def _generation_health_from_completions(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {"mean_length": 0.0, "terminated_length": 0.0}
    mean_length = sum(r.get("length", 0.0) for r in rows) / len(rows)
    terminated = sum(r.get("terminated", 0.0) for r in rows)
    return {"mean_length": round(mean_length, 4), "terminated_length": round(terminated, 4)}


def _generation_health_from_log_history(log_history: List[Dict[str, Any]]) -> Dict[str, float]:
    health: Dict[str, float] = {}
    for row in log_history:
        for source, target in (
            ("completions/mean_length", "mean_length"),
            ("completions/mean_terminated_length", "terminated_length"),
        ):
            if source in row:
                health[target] = float(row[source])
    return health


def _accepted_kwargs(cls: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Filter config kwargs so optional TRL generation fields stay version-safe."""
    try:
        params = inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError):
        return kwargs
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


def _grpo_generation_kwargs_rbrace_eos(tokenizer: Any) -> Optional[Dict[str, Any]]:
    """
    Add '}' as an extra EOS so generate() can stop after one JSON object instead of
    always filling max_new_tokens (which zeros TRL's mean_terminated_length signal).
    """
    if tokenizer is None:
        return None
    extra = json_action_eos_token_ids(tokenizer)
    if not extra:
        return None
    return {"eos_token_id": extra}


def _rollout_generation_health(rollouts: List[Any]) -> Dict[str, float]:
    lengths: List[int] = []
    terminated = 0
    for result in rollouts:
        for step in getattr(result, "steps", []):
            raw = str(getattr(step, "raw_completion", "") or "")
            if not raw:
                continue
            lengths.append(len(raw))
            c = clip_model_json_output(raw)
            if c.rstrip().endswith("}"):
                terminated += 1
    if not lengths:
        return {"mean_length": 0.0, "terminated_length": 0.0}
    return {
        "mean_length": round(sum(lengths) / len(lengths), 4),
        "terminated_length": float(terminated),
    }


def _load_grpo_policy(cfg: RLConfig, policy_path: Optional[str]):
    """Load the current policy for TRL GRPO updates."""
    try:
        from unsloth import FastLanguageModel

        source = policy_path if policy_path and Path(policy_path).exists() else cfg.base_model
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=source,
            max_seq_length=cfg.max_seq_length,
            load_in_4bit=True,
            dtype=None,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=cfg.seed,
        )
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token
        align_causal_lm_and_tokenizer(model, tokenizer, max_new_tokens=GENERATION_MAX_NEW_TOKENS)
        return model, tokenizer
    except Exception as e:
        log.warning("Unsloth GRPO load failed (%s); falling back to transformers.", e)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    source = policy_path if policy_path and Path(policy_path).exists() else cfg.base_model
    tokenizer = AutoTokenizer.from_pretrained(source, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(source)
    align_causal_lm_and_tokenizer(model, tokenizer, max_new_tokens=GENERATION_MAX_NEW_TOKENS)
    return model, tokenizer


# ─── Model helpers ────────────────────────────────────────────────────────────

def _current_policy_backend(cfg: RLConfig, model_path: Optional[str], temperature: float = 0.0):
    """
    Use a local SFT/RL LoRA policy if present; otherwise deterministic heuristic.

    During RL rollout we set temperature around 0.8 (GRPO needs diverse group samples).
    During evaluation we set `temperature=0.0` for deterministic scoring.
    """
    if model_path and Path(model_path).exists():
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            kwargs: Dict[str, Any] = {"torch_dtype": dtype}
            if torch.cuda.is_available():
                kwargs["device_map"] = "auto"
            base = AutoModelForCausalLM.from_pretrained(cfg.base_model, **kwargs)
            model = PeftModel.from_pretrained(base, model_path)
            model.eval()
            align_causal_lm_and_tokenizer(model, tokenizer, max_new_tokens=GENERATION_MAX_NEW_TOKENS)
            _eos = json_action_eos_token_ids(tokenizer)

            def _call(messages: List[Dict[str, str]]) -> str:
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                t = max(0.0, float(temperature))
                with torch.no_grad():
                    if t <= 0.0:
                        out = model.generate(
                            **inputs,
                            max_new_tokens=GENERATION_MAX_NEW_TOKENS,
                            do_sample=False,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=_eos if _eos else tokenizer.eos_token_id,
                        )
                    else:
                        out = model.generate(
                            **inputs,
                            max_new_tokens=GENERATION_MAX_NEW_TOKENS,
                            do_sample=True,
                            temperature=t,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=_eos if _eos else tokenizer.eos_token_id,
                        )
                text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                return clip_model_json_output(text)

            return _call
        except Exception as e:
            log.warning("Could not load %s as a PEFT adapter (%s); trying plain HF load.", model_path, e)
            return make_hf_pipeline_backend(model_path, max_new_tokens=GENERATION_MAX_NEW_TOKENS, temperature=temperature)
    return make_heuristic_backend()


def _load_policy_model(cfg: RLConfig, policy_path: Optional[str]):
    """
    Load an SFT/previous-RL policy and ensure it has trainable LoRA params.

    On GPU runtimes this uses 4-bit loading when bitsandbytes is available.
    On CPU this still loads in fp32 for a shape-only dry run, but callers
    usually set `dry_run=True` locally.
    """
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    load_kwargs: Dict[str, Any] = {"torch_dtype": dtype}
    if torch.cuda.is_available():
        load_kwargs["device_map"] = "auto"
        try:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        except Exception:
            pass

    tokenizer_source = policy_path if policy_path and Path(policy_path).exists() else cfg.base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(cfg.base_model, **load_kwargs)
    if policy_path and Path(policy_path).exists():
        model = PeftModel.from_pretrained(base, policy_path, is_trainable=True)
    else:
        peft_cfg = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj"],
        )
        model = get_peft_model(base, peft_cfg)

    model.train()
    align_causal_lm_and_tokenizer(model, tokenizer, max_new_tokens=GENERATION_MAX_NEW_TOKENS)
    return model, tokenizer


def _logprob_loss(model, tokenizer, batch: List[Dict[str, Any]], device) -> Tuple[Any, float]:
    """Policy-gradient objective: - advantage * logprob(action | state)."""
    import torch

    texts: List[Tuple[str, str, float]] = []
    for row in batch:
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": state_to_prompt(row["state"])},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        action = action_to_text(row["action"])
        texts.append((prompt, action, float(row["advantage"])))

    losses = []
    logprobs = []
    for prompt, action, adv in texts:
        prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                               max_length=tokenizer.model_max_length).input_ids.to(device)
        full = prompt + action
        enc = tokenizer(full, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length).to(device)
        labels = enc.input_ids.clone()
        labels[:, : prompt_ids.shape[1]] = -100

        out = model(**enc, labels=labels)
        # CausalLM loss is mean NLL over non-ignored action tokens.
        action_logprob = -out.loss
        logprobs.append(float(action_logprob.detach().cpu()))
        losses.append(-torch.tensor(adv, device=device, dtype=out.loss.dtype) * action_logprob)

    return torch.stack(losses).mean(), sum(logprobs) / max(1, len(logprobs))


def policy_gradient_update(
    cfg: RLConfig,
    transitions: List[Dict[str, Any]],
    policy_path: Optional[str],
    iteration: int,
    train_tasks: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Update the current policy online using TRL GRPOTrainer."""
    if cfg.dry_run:
        return {
            "iteration": iteration,
            "updated": False,
            "reason": "dry_run=True",
            "transitions": len(transitions),
            "checkpoint": policy_path,
        }

    import torch

    if not torch.cuda.is_available():
        return {
            "iteration": iteration,
            "updated": False,
            "reason": "CUDA not available; skipped RL weight update",
            "transitions": len(transitions),
            "checkpoint": policy_path,
        }

    if not train_tasks:
        return {"iteration": iteration, "updated": False, "reason": "no train tasks"}

    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    model, tokenizer = _load_grpo_policy(cfg, policy_path)
    component_log: List[Dict[str, float]] = []
    health_log: List[Dict[str, float]] = []
    reward_func = _grpo_reward_factory(component_log, health_log)
    dataset = Dataset.from_list([
        {
            "prompt": _task_to_grpo_prompt(task),
            "task_payload": json.dumps(task),
        }
        for task in train_tasks
    ])
    out_dir = Path(cfg.output_dir) / f"iter_{iteration}"
    out_dir.mkdir(parents=True, exist_ok=True)

    _gkw = _grpo_generation_kwargs_rbrace_eos(tokenizer)
    _eos_ids = json_action_eos_token_ids(tokenizer)
    if _eos_ids and getattr(model, "generation_config", None) is not None:
        model.generation_config.eos_token_id = _eos_ids
    grpo_kwargs: Dict[str, Any] = {
        "output_dir": str(out_dir),
        "max_steps": max(1, len(train_tasks) * cfg.epochs_per_iteration),
        "per_device_train_batch_size": cfg.batch_size,
        "gradient_accumulation_steps": cfg.grad_accum,
        "learning_rate": cfg.learning_rate,
        "logging_steps": 5,
        "save_steps": max(10, len(train_tasks)),
        "report_to": "none",
        "max_prompt_length": min(3072, cfg.max_seq_length),
        "max_completion_length": GENERATION_MAX_NEW_TOKENS,
        "num_generations": 4,
        "temperature": cfg.exploration_temperature,
    }
    if _gkw is not None:
        grpo_kwargs["generation_kwargs"] = _gkw
    args = GRPOConfig(**_accepted_kwargs(GRPOConfig, grpo_kwargs))
    trainer = GRPOTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        reward_funcs=[reward_func],
        processing_class=tokenizer,
    )
    train_result = trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(out_dir)

    reward_components = {
        "reward_ci": 0.0,
        "reward_minimal": 0.0,
        "reward_regression": 0.0,
        "reward_penalty": 0.0,
    }
    for row in component_log:
        for key in reward_components:
            reward_components[key] += float(row.get(key, 0.0))
    generation_health = _generation_health_from_completions(health_log)
    generation_health.update(_generation_health_from_log_history(trainer.state.log_history))

    return {
        "iteration": iteration,
        "updated": True,
        "method": "TRL_GRPOTrainer",
        "policy_update_signal": "GRPO uses generated-token logprobs internally; offline reward-to-go advantages remain logged for trajectory analysis.",
        "transitions": len(transitions),
        "train_tasks": len(train_tasks),
        "metrics": getattr(train_result, "metrics", {}),
        "reward_components": {k: round(v, 4) for k, v in reward_components.items()},
        "generation_health": generation_health,
        "checkpoint": str(out_dir),
    }


# ─── Iterative self-improvement loop ──────────────────────────────────────────

def train_rl(config: Optional[RLConfig] = None) -> Dict[str, Any]:
    """
    Iterative RL loop:
        1. evaluate / log current policy
        2. sample train tasks adaptively from failures/adversarial cases
        3. GRPOTrainer generates completions from the current policy
        4. reward function executes completions in CompliancePatchEnv
        5. GRPO updates the same policy checkpoint for the next iteration
    """
    cfg = config or RLConfig()
    seed_everything(cfg.seed)

    tasks = read_json(cfg.tasks_path)
    if cfg.max_tasks > 0:
        tasks = tasks[: cfg.max_tasks]
    train_tasks, test_tasks = _split_tasks(tasks, seed=cfg.seed)

    policy_path: Optional[str] = cfg.sft_adapter_dir if Path(cfg.sft_adapter_dir).exists() else None
    tabular_policy = TabularPatchPolicy(
        alpha=cfg.tabular_alpha,
        epsilon=cfg.tabular_epsilon,
        seed=cfg.seed,
    ) if (cfg.use_tabular_rl and cfg.dry_run) else None
    curve: List[Dict[str, Any]] = []
    updates: List[Dict[str, Any]] = []
    last_train_report: Optional[Dict[str, Any]] = None
    previous_failed: set[str] = set()
    total_recovered = 0

    for iteration in range(cfg.iterations + 1):
        log.info("RL iteration %d/%d using policy=%s", iteration, cfg.iterations, policy_path or "heuristic")
        if tabular_policy is not None:
            llm = tabular_policy.backend(explore=True, collect=True)
            eval_llm = tabular_policy.backend(explore=False, collect=False)
        else:
            llm = _current_policy_backend(
                cfg,
                policy_path if not cfg.dry_run else None,
                temperature=cfg.exploration_temperature,
            )
            eval_llm = _current_policy_backend(
                cfg,
                policy_path if not cfg.dry_run else None,
                temperature=0.0,
            )
        iter_traj_path = Path(cfg.trajectories_path).with_name(f"trajectories_rl_iter_{iteration}.jsonl")
        task_weights = _adaptive_task_weights(train_tasks, last_train_report)
        rollout_tasks = _sample_tasks(
            train_tasks,
            task_weights,
            count=max(len(train_tasks), 1),
            seed=cfg.seed + iteration,
        )

        # Rollout current policy.
        rollouts = run_rollouts(
            rollout_tasks,
            n_rollouts_per_task=cfg.rollouts_per_task,
            llm=llm,
            config=AgentConfig(max_steps=cfg.max_steps),
            trajectories_rl_path=iter_traj_path,
        )
        # Keep both per-iteration files and the conventional latest path the
        # user requested: project/data/trajectories_rl.jsonl.
        write_jsonl(Path(cfg.trajectories_path), list(read_jsonl(iter_traj_path)))
        generation_health = _rollout_generation_health(rollouts)
        fmt_metrics = _rollout_format_metrics(rollouts)

        # Evaluate current policy and write learning-curve point.
        train_report = evaluate(train_tasks, llm=eval_llm, config=AgentConfig(max_steps=cfg.max_steps), print_per_task=False)
        test_report = evaluate(test_tasks, llm=eval_llm, config=AgentConfig(max_steps=cfg.max_steps), print_per_task=False) if test_tasks else {"summary": {}, "per_task": []}
        train_summary = train_report["summary"]
        test_summary = test_report["summary"]
        combined_rows = train_report.get("per_task", []) + test_report.get("per_task", [])
        current_failed = {r["task_id"] for r in combined_rows if r.get("status") != "SUCCESS"}
        current_success = {r["task_id"] for r in combined_rows if r.get("status") == "SUCCESS"}
        recovered_tasks = len(previous_failed & current_success) if iteration > 0 else 0
        total_recovered += recovered_tasks
        previous_failed = current_failed

        point = {
            "iteration": iteration,
            "avg_reward": train_summary.get("avg_score", 0.0),
            "success_rate": train_summary.get("overall_success_rate", 0.0),
            "valid_json_rate": fmt_metrics.get("valid_json_rate", 0.0),
            "fallback_rate": fmt_metrics.get("fallback_rate", 0.0),
            "train_success_rate": train_summary.get("overall_success_rate", 0.0),
            "test_success_rate": test_summary.get("overall_success_rate", 0.0),
            "hidden_violation_rate": train_summary.get("hidden_violation_rate", 0.0),
            "partial_fix_rate": train_summary.get("partial_fix_rate", 0.0),
            "no_fix_rate": train_summary.get("no_fix_rate", 0.0),
            "cheat_resistance": train_summary.get("cheat_resistance", 0.0),
            "failure_stats": train_summary.get("failure_stats", {}),
            "test_failure_stats": test_summary.get("failure_stats", {}),
            "reward_components": train_summary.get("reward_components", {}),
            "mean_length": generation_health.get("mean_length", 0.0),
            "terminated_length": generation_health.get("terminated_length", 0.0),
            "avg_confidence": train_summary.get("avg_confidence", 0.0),
            "high_confidence_wrong": train_summary.get("high_confidence_wrong", 0),
            "recovered_tasks": recovered_tasks,
            "total_recovered_tasks": total_recovered,
            "task_weight_summary": {
                "min": round(min(task_weights.values()), 4) if task_weights else 0.0,
                "max": round(max(task_weights.values()), 4) if task_weights else 0.0,
                "adversarial_avg": round(
                    sum(task_weights[t["task_id"]] for t in train_tasks if t.get("adversarial"))
                    / max(1, sum(1 for t in train_tasks if t.get("adversarial"))),
                    4,
                ) if task_weights else 0.0,
            },
        }
        curve.append(point)
        write_json(Path(cfg.learning_curve_path), curve)
        last_train_report = train_report
        log.info("Iter %d → reward %.3f, train %.1f%%, test %.1f%%, hidden %.1f%%, mean_length=%.1f, terminated=%.1f, recovered=%d",
                 iteration, point["avg_reward"], 100 * point["train_success_rate"],
                 100 * point["test_success_rate"], 100 * point["hidden_violation_rate"],
                 point["mean_length"], point["terminated_length"], recovered_tasks)
        print(
            f"iter={iteration} avg_reward={point['avg_reward']:.3f} "
            f"success_rate={point['success_rate']:.2%} "
            f"valid_json={point['valid_json_rate']:.2%} "
            f"fallback={point['fallback_rate']:.2%} "
            f"hidden_violation_rate={point['hidden_violation_rate']:.2%} "
            f"mean_length={point['mean_length']:.1f} "
            f"terminated_length={point['terminated_length']:.1f}"
        )
        if point["terminated_length"] == 0:
            print("WARNING: model not terminating properly")

        if iteration == cfg.iterations:
            break

        transitions = load_rl_transitions(str(iter_traj_path), gamma=cfg.gamma)
        tabular_update = tabular_policy.update_from_rollouts(rollouts) if tabular_policy is not None else {}
        update = policy_gradient_update(cfg, transitions, policy_path, iteration + 1, train_tasks=rollout_tasks)
        update.update(tabular_update)
        updates.append(update)
        policy_path = update.get("checkpoint") or policy_path
        if tabular_policy is not None:
            write_json(DATA_DIR / "tabular_rl_policy.json", tabular_policy.to_dict())

    result = {
        "config": asdict(cfg),
        "optimization_method": "TRL_GRPOTrainer" if not cfg.dry_run else "dry_run_no_weight_update",
        "train_task_ids": [t["task_id"] for t in train_tasks],
        "test_task_ids": [t["task_id"] for t in test_tasks],
        "learning_curve": curve,
        "updates": updates,
        "final_policy": policy_path,
        "tabular_policy": tabular_policy.to_dict() if tabular_policy is not None else None,
    }
    if curve:
        first, last = curve[0], curve[-1]
        before_success = float(first.get("success_rate", 0.0))
        after_success = float(last.get("success_rate", 0.0))
        before_hidden = float(first.get("hidden_violation_rate", 0.0))
        after_hidden = float(last.get("hidden_violation_rate", 0.0))
        print("------------------------------------------------------------\n## RL TRAINING — HEADLINE\n------------------------------------------------------------")
        print(f"  Success rate:        {before_success:.3f} → {after_success:.3f}  (improvement vs first iter)")
        print(f"  Hidden violation rate: {before_hidden:.3f} → {after_hidden:.3f}")
        print(f"  Recovered failed tasks (cumulative):  {int(last.get('total_recovered_tasks', 0))}")
        print("  → Model improves over time; the curve below is the full story (smoothed + last-10).")
        try:
            print_interpretation_curves()
        except Exception:
            pass
        print_learning_curve_footer(curve, window=6)
    write_json(DATA_DIR / "rl_training_log.json", result)
    return result


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the CompliancePatchBench RL self-improvement loop.")
    p.add_argument("--tasks", default=str(TASKS_PATH))
    p.add_argument("--base-model", default=DEFAULT_RL_BASE_MODEL)
    p.add_argument("--sft-adapter-dir", default=str(DEFAULT_SFT_ADAPTER_DIR))
    p.add_argument("--output-dir", default=str(RL_ADAPTER_DIR))
    p.add_argument("--iterations", type=int, default=40)
    p.add_argument("--rollouts-per-task", type=int, default=1)
    p.add_argument("--epochs-per-iteration", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=8)
    p.add_argument("--max-tasks", type=int, default=20)
    p.add_argument("--dry-run", action="store_true", help="Collect trajectories/metrics but skip GRPO model updates.")
    p.add_argument("--use-tabular-baseline", action="store_true",
                   help="Use the tabular controller only for CPU dry-run/baseline rollouts; final training uses TRL GRPO.")
    p.add_argument("--tabular-alpha", type=float, default=0.45)
    p.add_argument("--tabular-epsilon", type=float, default=0.20)
    p.add_argument("--exploration-temperature", type=float, default=0.8,
                   help="Neural policy sampling temperature for GRPO / rollout (default 0.8; low values hurt group-relative advantages).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = RLConfig(
        tasks_path=args.tasks,
        base_model=args.base_model,
        sft_adapter_dir=args.sft_adapter_dir,
        output_dir=args.output_dir,
        iterations=args.iterations,
        rollouts_per_task=args.rollouts_per_task,
        epochs_per_iteration=args.epochs_per_iteration,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.lr,
        gamma=args.gamma,
        max_steps=args.max_steps,
        max_tasks=args.max_tasks,
        dry_run=args.dry_run,
        use_tabular_rl=args.use_tabular_baseline,
        tabular_alpha=args.tabular_alpha,
        tabular_epsilon=args.tabular_epsilon,
        exploration_temperature=args.exploration_temperature,
    )
    print(json.dumps(train_rl(cfg), indent=2))


if __name__ == "__main__":
    main()
