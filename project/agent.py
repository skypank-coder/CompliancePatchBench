"""
agent.py
========

Robust agent loop for CompliancePatchBench.

Highlights vs the original:
    - STRICT JSON output enforced via system prompt + retry
    - Self-healing retry (`max_retries`) when the model emits invalid JSON
    - Heuristic fallback action so a single bad reply never crashes a rollout
    - Bounded context window (last `context_window` messages only)
    - Deterministic generation (temperature=0 by default)
    - Pluggable LLM backend: OpenAI-compatible HTTP, HuggingFace pipeline, or
      a tiny local heuristic policy that requires no API key (for CI / smoke tests)
    - Detailed per-step trajectory record (prompt, raw_completion, parsed_action,
      reward, observation snapshot)

The agent talks to a `CompliancePatchEnv` from the existing `environment` package.
It consumes tasks in the format produced by `project.task_generator`.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .hidden_compliance import run_hidden_compliance_checks
from .utils import extract_json, get_logger

log = get_logger("agent")


# ─── Strict system prompt ────────────────────────────────────────────────────
#
# This prompt is intentionally short + emphatic. Every requirement that's
# critical for downstream training (single JSON, exact keys, no commentary)
# is repeated and bullet-listed. The model has fewer ways to go wrong.

SYSTEM_PROMPT = """You are CompliancePatchAgent, a code-fix agent operating inside a sandboxed CompliancePatchEnv.

GOAL
    Fix every violation in the codebase by emitting a sequence of single-action JSON commands.
    One JSON object per turn. Nothing else.

OUTPUT FORMAT — STRICT
    Reply with EXACTLY ONE JSON object. No prose, no markdown, no code fences,
    no leading/trailing text. The first character of your reply MUST be "{"
    and the last character MUST be "}". Any other format is treated as a crash.

ALLOWED ACTIONS
    {"action_type": "read_file",       "path": "<exact filename from observation>"}
    {"action_type": "write_patch",     "file": "<exact filename>", "line_start": <int>, "line_end": <int>, "new_code": "<replacement source>"}
    {"action_type": "run_ci"}
    {"action_type": "finalize_patch"}

HARD RULES
    1. "file" / "path" MUST appear verbatim in the observation's "available_files".
    2. "line_start"/"line_end" MUST come from the violations list — do NOT guess.
    3. "new_code" MUST keep the original Python indentation; never delete the
       function or replace it with `pass` (deletion = -1.0 cheat penalty).
    4. Always run_ci AFTER writing a patch so the env can score it.
    5. When all violations are PASS, emit finalize_patch.
    6. If the read budget is exhausted, do not call read_file — write the patch
       directly using the line numbers in the violations list.

THINK SILENTLY. RESPOND WITH ONE JSON OBJECT.
"""

# Predefined SAFE patches per rule, used by the heuristic fallback policy.
# Stored unindented — the heuristic re-applies the original line's indent
# at write time so patches always slot into the surrounding block correctly.
RULE_PATCHES: Dict[str, str] = {
    "GDPR-ART5-1A": "logger.info('event uid=%s', str(user.id))",
    "GDPR-ART5-1C": "return jsonify({'id': user.id, 'email': user.email})",
    "GDPR-ART25":   "@limiter.limit('5/minute')",
    "GDPR-ART32":   "DEBUG = False",
    "OWASP-A01":    "invoice = get_object_or_404(Invoice, pk=pk, owner=request.user)",
    "OWASP-A02":    "SECRET_KEY = os.environ['SECRET_KEY']",
    "OWASP-A03":    "cursor.execute('SELECT * FROM users WHERE name = %s', (name,))",
}

# CHEAT patches the heuristic occasionally chooses on HARD adversarial tasks.
# These pass the env's CI (regex/semantic check) but trip the hidden oracle,
# so dataset_builder rightly drops them and evaluate.py marks them PARTIAL/FAIL.
RULE_PATCHES_CHEAT: Dict[str, str] = {
    "GDPR-ART5-1A": "logger.info(f'login ok for {hashlib.sha256(user.email.encode()).hexdigest()}')",
    "OWASP-A02":    "SECRET_KEY = os.environ.get('SECRET_KEY', 'literal-fallback-secret')",
    "OWASP-A03":    "cursor.execute(f\"SELECT id FROM users WHERE name = '{name.replace(chr(39), '')}'\")",
    "GDPR-ART32":   "DEBUG = os.environ.get('DEBUG', True)",
}


# ─── Heuristic difficulty controls ───────────────────────────────────────────
#
# These knobs make the heuristic *intentionally imperfect* so the benchmark
# doesn't show unrealistic 100% success rates. Probabilities are deterministic
# functions of (task_id, rule_id) so reruns are reproducible.

# P(skip a single violation) by task difficulty. Tuned so that:
#   easy   → ~90% task success rate
#   medium → ~60-70% task success rate
#   hard   → ~40-60% task success rate
SKIP_PROB_BY_DIFFICULTY = {"easy": 0.10, "medium": 0.55, "hard": 0.45}
# P(use the cheat patch) for adversarial HARD tasks where a cheat exists.
CHEAT_PROB_HARD_ADVERSARIAL = 0.4


def _det_unit(*parts: str) -> float:
    """Deterministic float in [0,1) from string parts. Used for reproducible skipping.

    Uses md5 (cryptographic-grade uniformity) so skip/cheat decisions are well-distributed
    across the task space rather than clumping due to a poor polynomial hash.
    """
    import hashlib
    h = hashlib.md5("|".join(parts).encode()).digest()
    n = int.from_bytes(h[:4], "big")
    return n / 0xFFFFFFFF


def classify_failure_type(result: "TrajectoryResult") -> str:
    """Coarse failure label used for adaptive RL sampling and reporting."""
    if result.hidden_violation:
        return "hidden_violation"
    if 0 < result.violations_fixed < result.violations_total:
        return "partial_fix"
    if result.violations_total > 0 and result.violations_fixed == 0:
        return "no_fix"
    return "none"


def estimate_confidence(result: "TrajectoryResult") -> float:
    """
    Simple confidence score in [0, 1].

    It is intentionally process-based rather than correctness-based, so the
    interesting failure mode "high confidence but wrong" remains visible.
    """
    if not result.steps:
        return 0.0

    n = len(result.steps)
    retry_rate = sum(s.retries for s in result.steps) / max(1, n)
    fallback_rate = sum(1 for s in result.steps if s.used_fallback) / max(1, n)
    actions = [s.parsed_action.get("action_type", "unknown") for s in result.steps]
    repeated = sum(1 for a, b in zip(actions, actions[1:]) if a == b)
    consistency = 1.0 - min(1.0, repeated / max(1, n - 1))

    rewards = [float(s.reward) for s in result.steps]
    mean = sum(rewards) / max(1, len(rewards))
    variance = sum((r - mean) ** 2 for r in rewards) / max(1, len(rewards))
    stability = 1.0 - min(1.0, (variance ** 0.5) / (abs(mean) + 1.0))

    score = (
        0.35 * max(0.0, 1.0 - retry_rate)
        + 0.25 * max(0.0, 1.0 - fallback_rate)
        + 0.20 * consistency
        + 0.20 * stability
    )
    return round(max(0.0, min(1.0, score)), 4)


# ─── Data classes ────────────────────────────────────────────────────────────

@dataclass
class StepRecord:
    """One step in a trajectory — ready to be logged to JSONL."""
    step: int
    prompt: str                     # what we sent to the model (last user msg only — full history is heavy)
    raw_completion: str             # what the model returned
    parsed_action: Dict[str, Any]   # JSON we actually executed (after parse + fallback)
    reward: float
    observation: Dict[str, Any]     # observation AFTER the action
    used_fallback: bool             # True if we had to substitute a heuristic action
    retries: int                    # how many JSON re-prompts we used


@dataclass
class RLStepRecord:
    """One transition for policy-gradient training: (s, a, r, s', done)."""
    step: int
    state: Dict[str, Any]
    action: Dict[str, Any]
    reward: float
    next_state: Dict[str, Any]
    logprob: float = 0.0
    done: bool = False


@dataclass
class TrajectoryResult:
    """Outcome of one full episode."""
    task_id: str
    final_score: float
    violations_fixed: int
    violations_total: int
    steps: List[StepRecord] = field(default_factory=list)
    rl_trajectory: List[RLStepRecord] = field(default_factory=list)
    error: Optional[str] = None

    # Hidden compliance oracle output (populated post-rollout in agent.run)
    hidden_violation: bool = False
    hidden_reason: str = "ok"
    hidden_findings: List[Dict[str, Any]] = field(default_factory=list)

    # Task metadata propagated for downstream eval/dataset filters
    difficulty: str = "easy"
    adversarial: bool = False
    failure_type: str = "none"
    confidence: float = 0.0

    @property
    def success_rate(self) -> float:
        if self.violations_total == 0:
            return 0.0
        return self.violations_fixed / self.violations_total

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "final_score": self.final_score,
            "violations_fixed": self.violations_fixed,
            "violations_total": self.violations_total,
            "success_rate": self.success_rate,
            "error": self.error,
            "hidden_violation": self.hidden_violation,
            "hidden_reason": self.hidden_reason,
            "hidden_findings": self.hidden_findings,
            "difficulty": self.difficulty,
            "adversarial": self.adversarial,
            "failure_type": self.failure_type,
            "confidence": self.confidence,
            "steps": [s.__dict__ for s in self.steps],
            "rl_trajectory": [s.__dict__ for s in self.rl_trajectory],
            "cumulative_reward": round(sum(s.reward for s in self.rl_trajectory), 4),
        }


# ─── LLM backends ─────────────────────────────────────────────────────────────

LLMCallable = Callable[[List[Dict[str, str]]], str]
"""An LLM is just: messages -> raw text completion."""


def make_openai_backend(
    model: str = "gpt-4o-mini",
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> LLMCallable:
    """OpenAI-compatible chat completions (also works for vLLM/together/HF endpoints)."""
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError("Install openai>=1.0 to use the OpenAI backend") from e

    client = OpenAI(
        api_key=api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("HF_TOKEN"),
        base_url=base_url or os.environ.get("API_BASE_URL"),
    )

    def _call(messages: List[Dict[str, str]]) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    return _call


def make_hf_pipeline_backend(
    model_name_or_path: str,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
) -> LLMCallable:
    """Local HF model via transformers — used inside the Colab training notebook."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise RuntimeError("Install transformers + torch to use the HF backend") from e

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()

    def _call(messages: List[Dict[str, str]]) -> str:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0.0,
                temperature=max(temperature, 1e-5),
                pad_token_id=tokenizer.pad_token_id,
            )
        return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    return _call


def make_heuristic_backend() -> LLMCallable:
    """
    Tiny rule-based policy that requires NO model.
    Used for smoke tests, offline CI, and as a fallback.

    Reads the most-recent observation in the message history and emits a
    deterministic next action: read each unread file → write a patch per
    violation → run_ci → finalize.
    """

    def _call(messages: List[Dict[str, str]]) -> str:
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        obs = extract_json(last_user) or {}
        action = _heuristic_choose_action(
            obs,
            read_already=_read_files_seen(messages),
            patched_already=_patches_seen(messages),
            ci_runs=_ci_runs_seen(messages),
        )
        return json.dumps(action)

    return _call


def _read_files_seen(messages: List[Dict[str, str]]) -> set[str]:
    seen: set[str] = set()
    for m in messages:
        if m["role"] != "assistant":
            continue
        parsed = extract_json(m["content"]) or {}
        if parsed.get("action_type") == "read_file" and parsed.get("path"):
            seen.add(parsed["path"])
    return seen


def _patches_seen(messages: List[Dict[str, str]]) -> set[Tuple[str, int, int]]:
    """All (file, line_start, line_end) tuples we've already issued write_patch for."""
    seen: set[Tuple[str, int, int]] = set()
    for m in messages:
        if m["role"] != "assistant":
            continue
        parsed = extract_json(m["content"]) or {}
        if parsed.get("action_type") == "write_patch":
            seen.add((
                parsed.get("file", ""),
                int(parsed.get("line_start", 0)),
                int(parsed.get("line_end", 0)),
            ))
    return seen


def _ci_runs_seen(messages: List[Dict[str, str]]) -> int:
    """How many times we've called run_ci so far."""
    n = 0
    for m in messages:
        if m["role"] != "assistant":
            continue
        parsed = extract_json(m["content"]) or {}
        if parsed.get("action_type") == "run_ci":
            n += 1
    return n


def _heuristic_choose_action(
    obs: Dict[str, Any],
    read_already: set[str],
    patched_already: Optional[set] = None,
    ci_runs: int = 0,
) -> Dict[str, Any]:
    """
    Deterministic policy used by the offline backend AND the agent fallback.

    Order of operations:
        1. read each violated file (budget permitting)
        2. write a patch per violation — but on medium/hard tasks, deterministically
           SKIP some violations so the success rate is realistic
        3. on hard adversarial tasks, occasionally substitute the CHEAT patch
           (the hidden oracle catches it later)
        4. run_ci once to score
        5. finalize_patch
    """
    patched_already = patched_already or set()
    available = obs.get("available_files", [])
    violations = obs.get("violations", [])
    fixed = obs.get("violations_fixed", 0)
    total = obs.get("violations_total", len(violations))
    budget = obs.get("file_reads_remaining", 0)
    ci_results = obs.get("ci_results", []) or []
    fixed_keys = {(r.get("file"), r.get("rule_id")) for r in ci_results if r.get("ci") == "PASS"}

    difficulty = obs.get("task_difficulty", "easy")
    adversarial = bool(obs.get("task_adversarial", False))
    task_id = obs.get("task_id", "")

    if fixed >= total and total > 0:
        return {"action_type": "finalize_patch"}

    if budget > 0:
        for v in violations:
            if v.get("file") not in read_already and v.get("file") in available:
                return {"action_type": "read_file", "path": v["file"]}

    # Build skip + cheat decisions per violation, deterministic on (task_id, rule_id, file, line)
    file_view = obs.get("last_file_view") or ""
    skip_p = SKIP_PROB_BY_DIFFICULTY.get(difficulty, 0.0)

    for v in violations:
        key = (v.get("file"), v.get("rule_id"))
        if key in fixed_keys:
            continue
        loc = (v.get("file"), int(v.get("line_start", 0)), int(v.get("line_end", 0)))
        if loc in patched_already:
            continue

        seed_str = (task_id, v.get("rule_id", ""), v.get("file", ""), str(v.get("line_start", 0)))
        # Deterministic per-violation skip. Single-violation tasks can fully fail;
        # multi-violation tasks become PARTIAL — that's intentional.
        if _det_unit("skip", *seed_str) < skip_p:
            patched_already.add(loc)        # mark as "tried" so we don't loop
            continue

        # On hard adversarial tasks, sometimes pick the cheat patch.
        use_cheat = (
            difficulty == "hard"
            and adversarial
            and v["rule_id"] in RULE_PATCHES_CHEAT
            and _det_unit("cheat", *seed_str) < CHEAT_PROB_HARD_ADVERSARIAL
        )
        raw_patch = (RULE_PATCHES_CHEAT if use_cheat else RULE_PATCHES).get(
            v["rule_id"], "pass  # TODO: fix"
        )
        new_code = _apply_indent_from_view(file_view, v["line_start"], raw_patch)
        return {
            "action_type": "write_patch",
            "file": v["file"],
            "line_start": v["line_start"],
            "line_end": v["line_end"],
            "new_code": new_code,
        }

    if ci_runs < 2 and len(ci_results) == 0:
        return {"action_type": "run_ci"}
    return {"action_type": "finalize_patch"}


def _apply_indent_from_view(file_view: str, line_no: int, raw_patch: str) -> str:
    """
    Re-indent `raw_patch` to match the original line's indentation.

    `file_view` is the numbered file string we attach to observations after a
    read_file action — each line looks like "  42: <code>". We extract the
    target line's leading whitespace and prepend it to every line of the patch.
    Falls back to no-indent if we can't find the line.
    """
    if not file_view:
        return raw_patch
    needle = f"{line_no:3d}: "
    for line in file_view.split("\n"):
        if line.startswith(needle):
            original = line[len(needle):]
            indent = len(original) - len(original.lstrip())
            pad = " " * indent
            return "\n".join(pad + p for p in raw_patch.split("\n"))
    return raw_patch


# ─── Agent core ──────────────────────────────────────────────────────────────

@dataclass
class AgentConfig:
    max_steps: int = 16
    max_retries: int = 2          # # of times to re-prompt for valid JSON before falling back
    context_window: int = 6       # keep last N (user/assistant) messages, plus the system msg
    use_fallback: bool = True
    verbose: bool = False


class ComplianceAgent:
    """
    Wraps an LLM callable + CompliancePatchEnv into a self-healing agent loop.

    Usage:
        env = CompliancePatchEnv()
        agent = ComplianceAgent(llm=my_backend, config=AgentConfig())
        result = agent.run(env=env, task=task_dict)
    """

    def __init__(
        self,
        llm: Optional[LLMCallable] = None,
        config: Optional[AgentConfig] = None,
        system_prompt: str = SYSTEM_PROMPT,
    ):
        self.llm = llm or make_heuristic_backend()
        self.config = config or AgentConfig()
        self.system_prompt = system_prompt

    # ── Public API ────────────────────────────────────────────────────────

    def run(self, env, task: Dict[str, Any]) -> TrajectoryResult:
        """
        Roll out one full episode. Never crashes — exceptions become a
        TrajectoryResult with error=... so dataset_builder can keep going.

        After the env terminates we ALSO run a hidden compliance oracle on the
        final patched codebase (`hidden_compliance.run_hidden_compliance_checks`).
        Detected cheats reduce the final reward by 0.5.
        """
        result = TrajectoryResult(
            task_id=task["task_id"],
            final_score=0.0,
            violations_fixed=0,
            violations_total=len(task.get("violations", [])),
            difficulty=task.get("difficulty", "easy"),
            adversarial=bool(task.get("adversarial", False)),
        )
        # Make the task available to formatters / heuristics so they can branch
        # on difficulty without us refactoring every backend.
        self._current_task = task
        try:
            obs = env.reset(
                task_id=task["task_id"],
                codebase=task["codebase"],
                violations=task["violations"],
                max_steps=task.get("max_steps", self.config.max_steps),
                file_reads_remaining=task.get("file_reads_remaining", 5),
            )
            messages: List[Dict[str, str]] = [{"role": "system", "content": self.system_prompt}]
            self._push_user(messages, self._format_initial_user(task, obs))
            info: Dict[str, Any] = {}

            for step in range(1, self.config.max_steps + 1):
                state_snapshot = _obs_snapshot(obs, task=task)
                action, raw, used_fallback, retries, logprob = self._choose_action(messages, obs)

                obs, reward, done, info = env.step(action)
                next_state_snapshot = _obs_snapshot(obs, task=task)

                result.steps.append(StepRecord(
                    step=step,
                    prompt=messages[-1]["content"][:4000],
                    raw_completion=raw[:4000],
                    parsed_action=action,
                    reward=float(reward),
                    observation=next_state_snapshot,
                    used_fallback=used_fallback,
                    retries=retries,
                ))
                result.rl_trajectory.append(RLStepRecord(
                    step=step,
                    state=state_snapshot,
                    action=action,
                    reward=float(reward),
                    next_state=next_state_snapshot,
                    logprob=float(logprob),
                    done=bool(done),
                ))

                if done:
                    if (
                        action.get("action_type") != "finalize_patch"
                        and not getattr(env.state, "done", True)
                    ):
                        obs, reward, done, info = env.step({"action_type": "finalize_patch"})
                    break

                messages.append({"role": "assistant", "content": json.dumps(action)})
                self._push_user(messages, _format_observation(obs, task=task))

            crit = (info or {}).get("critique", {})
            critique = crit if crit else _critique_from_obs(obs)
            result.final_score = float(critique.get("final_score", obs.get("cumulative_reward", 0.0)))
            result.violations_fixed = int(critique.get("violations_fixed", obs.get("violations_fixed", 0)))
            result.violations_total = int(critique.get("violations_total", result.violations_total))

            # ── Hidden compliance oracle ──────────────────────────────────
            # The environment enforces this at finalize time for deployed
            # OpenEnv clients. Keep this fallback for older envs/tests that
            # may not return hidden-oracle fields in the critique.
            oracle_already_scored = "hidden_violation" in critique
            if oracle_already_scored:
                result.hidden_violation = bool(critique.get("hidden_violation"))
                result.hidden_reason = str(critique.get("hidden_reason", "ok"))
                result.hidden_findings = list(critique.get("hidden_findings", []))
            else:
                patched_codebase = dict(getattr(env.state, "patches", {}) or {})
                oracle = run_hidden_compliance_checks(patched_codebase)
                result.hidden_violation = bool(oracle.get("hidden_violation"))
                result.hidden_reason = str(oracle.get("reason", "ok"))
                result.hidden_findings = list(oracle.get("findings", []))
            terminal_penalty = 0.0
            if result.hidden_violation and not oracle_already_scored:
                terminal_penalty -= 1.0
                if self.config.verbose:
                    log.warning("Hidden violation on %s: %s",
                                task["task_id"], result.hidden_reason[:160])
            if (
                0 < result.violations_fixed < result.violations_total
                and not bool(critique.get("partial_fix"))
            ):
                terminal_penalty -= 0.4
            if (
                result.violations_total > 0
                and result.violations_fixed == 0
                and not result.hidden_violation
                and not bool(critique.get("no_fix"))
            ):
                terminal_penalty -= 0.1
            if terminal_penalty and result.rl_trajectory:
                result.final_score = round(result.final_score + terminal_penalty, 4)
                result.rl_trajectory[-1].reward = round(result.rl_trajectory[-1].reward + terminal_penalty, 4)
            result.failure_type = classify_failure_type(result)
            result.confidence = estimate_confidence(result)
        except Exception as e:
            log.exception("Rollout crashed for task %s", task.get("task_id"))
            result.error = f"{type(e).__name__}: {e}"
            result.failure_type = "no_fix"
            result.confidence = estimate_confidence(result)
        return result

    # ── Internal helpers ──────────────────────────────────────────────────

    def _choose_action(
        self,
        messages: List[Dict[str, str]],
        obs: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], str, bool, int, float]:
        """
        Try `max_retries+1` times to get valid JSON. On failure, fall back
        to the heuristic policy. Returns (action_dict, raw_completion, fallback_used, retries, logprob).
        """
        ctx = self._capped_context(messages)
        last_raw = ""
        for attempt in range(self.config.max_retries + 1):
            try:
                raw = self.llm(ctx)
            except Exception as e:
                log.warning("LLM call failed (attempt %d): %s", attempt + 1, e)
                last_raw = f"<llm_error: {e}>"
                time.sleep(0.5 * (attempt + 1))
                continue
            last_raw = raw
            parsed = extract_json(raw)
            if parsed and isinstance(parsed, dict) and parsed.get("action_type") in {
                "read_file", "write_patch", "run_ci", "finalize_patch",
            }:
                logprob = float(parsed.pop("_logprob", 0.0) or 0.0)
                parsed = {k: v for k, v in parsed.items() if not str(k).startswith("_")}
                if self.config.verbose:
                    log.info("step action: %s", parsed.get("action_type"))
                return parsed, raw, False, attempt, logprob
            log.info("Invalid JSON / unknown action_type (attempt %d). Re-prompting.", attempt + 1)
            ctx = ctx + [{
                "role": "user",
                "content": (
                    "Your previous reply was not valid JSON or used an invalid action_type. "
                    "Reply with ONE JSON object. Allowed action_type values: "
                    "read_file, write_patch, run_ci, finalize_patch."
                ),
            }]

        # All retries exhausted → fallback
        if self.config.use_fallback:
            action = _heuristic_choose_action(
                obs,
                read_already=self._files_already_read(messages),
                patched_already=_patches_seen(messages),
                ci_runs=_ci_runs_seen(messages),
            )
            log.warning("Falling back to heuristic action: %s", action.get("action_type"))
            return action, last_raw, True, self.config.max_retries, 0.0
        return {"action_type": "finalize_patch"}, last_raw, True, self.config.max_retries, 0.0

    def _capped_context(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Always keep the system prompt; truncate the rest to the last N msgs."""
        system = [m for m in messages if m["role"] == "system"][:1]
        body = [m for m in messages if m["role"] != "system"]
        return system + body[-self.config.context_window :]

    def _push_user(self, messages: List[Dict[str, str]], content: str) -> None:
        messages.append({"role": "user", "content": content})

    def _files_already_read(self, messages: List[Dict[str, str]]) -> set[str]:
        return _read_files_seen(messages)

    def _format_initial_user(self, task: Dict[str, Any], obs: Dict[str, Any]) -> str:
        return (
            f"TASK: {task.get('description', '(no description)')}\n"
            f"FRAMEWORK: {', '.join(task.get('framework', []))}\n"
            f"DIFFICULTY: {task.get('difficulty', 'easy')}\n"
            f"VIOLATIONS_TO_FIX: {len(task.get('violations', []))}\n"
            f"\nINITIAL_OBSERVATION:\n{json.dumps(_obs_snapshot(obs, task=task), ensure_ascii=False)}"
        )


# ─── Observation formatting ──────────────────────────────────────────────────

def _obs_snapshot(obs: Dict[str, Any], task: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Trim noisy/long fields and propagate task-level metadata into the obs."""
    if not isinstance(obs, dict):
        return {"raw": str(obs)[:1000]}
    snap = {
        "available_files": obs.get("available_files", []),
        "violations": obs.get("violations", []),
        "file_reads_remaining": obs.get("file_reads_remaining"),
        "step_count": obs.get("step_count"),
        "violations_fixed": obs.get("violations_fixed"),
        "violations_total": obs.get("violations_total"),
        "cumulative_reward": round(float(obs.get("cumulative_reward", 0.0)), 4),
        "ci_results": [
            {"file": r.get("file"), "rule_id": r.get("rule_id"), "ci": r.get("ci"),
             "reason": (r.get("reason") or "")[:120]}
            for r in (obs.get("ci_results") or [])
        ],
    }
    if task is not None:
        snap["task_id"] = task.get("task_id")
        snap["task_difficulty"] = task.get("difficulty", "easy")
        snap["task_adversarial"] = bool(task.get("adversarial", False))
    res = obs.get("action_result")
    if isinstance(res, str) and "\n" in res:
        numbered = []
        for i, line in enumerate(res.split("\n")[:60], 1):
            numbered.append(f"{i:3d}: {line}")
        snap["last_file_view"] = "\n".join(numbered)
    elif isinstance(res, str):
        snap["action_result"] = res[:200]
    return snap


def _format_observation(obs: Dict[str, Any], task: Optional[Dict[str, Any]] = None) -> str:
    return f"OBSERVATION:\n{json.dumps(_obs_snapshot(obs, task=task), ensure_ascii=False)}"


def _critique_from_obs(obs: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "final_score": obs.get("cumulative_reward", 0.0),
        "violations_fixed": obs.get("violations_fixed", 0),
        "violations_total": obs.get("violations_total", 0),
    }
