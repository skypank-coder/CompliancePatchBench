"""
GRPO Training Script for CompliancePatchBench patch agent.
Run on Google Colab with a T4 GPU.
Trains a configurable Qwen/Unsloth instruct model to write compliance patches via GRPO.

Usage:
  pip install unsloth trl requests
  export ENV_BASE_URL=https://your-space.hf.space  # or http://localhost:7860
  python tools/grpo_training.py
"""

import os
import sys
import json
import re
import requests
import torch
from pathlib import Path
from typing import List

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from environment.patch_env import CompliancePatchEnv
from project.agent import (
    GENERATION_MAX_NEW_TOKENS,
    SYSTEM_PROMPT,
    _safe_fallback_action,
    align_causal_lm_and_tokenizer,
    json_action_eos_token_ids,
)
from project.utils import clip_model_json_output
from environment.tasks.task1_single_file import get_task as get_task1
from environment.tasks.task2_django_app import get_task as get_task2

ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")
MODEL_NAME = os.environ.get("MODEL_NAME", "unsloth/Qwen2.5-3B-Instruct")
MAX_STEPS = 30
BATCH_SIZE = 4
TASKS = ["task1_single_file", "task2_django_app"]

TASK_CACHE = {
    "task1_single_file": get_task1(),
    "task2_django_app": get_task2(),
}


def call_env(endpoint: str, payload: dict) -> dict:
    r = requests.post(f"{ENV_BASE_URL}/{endpoint}", json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


def rollout(model, tokenizer, task_id: str) -> dict:
    """Run one episode and return (prompt, completion, reward)."""
    reset = call_env("patch/reset", {"task_id": task_id})
    session_id = reset["session_id"]
    obs = reset["observation"]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"Task: {task_id}\n"
            f"Files: {obs['available_files']}\n"
            f"Violations to fix: {json.dumps(obs['violations'], indent=2)}\n"
            f"Read budget: {obs['file_reads_remaining']}\n"
            "Begin. Read the relevant file first."
        )}
    ]

    final_reward = 0.0
    all_completions = []
    first_file = obs["available_files"][0] if obs.get("available_files") else None

    for step in range(20):
        # Format prompt
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        _eos = json_action_eos_token_ids(tokenizer)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=GENERATION_MAX_NEW_TOKENS,
                temperature=0.6,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=_eos if _eos else tokenizer.eos_token_id,
            )

        raw = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        completion = clip_model_json_output(raw)
        all_completions.append(completion)

        action = None
        retry_prompt = None
        for attempt in range(3):
            candidate = completion if attempt == 0 else retry_prompt or completion
            match = re.search(r'\{.*\}', candidate, re.DOTALL)
            try:
                parsed = json.loads(match.group(0)) if match else None
            except Exception:
                parsed = None
            if isinstance(parsed, dict) and parsed.get("action_type") in {"read_file", "write_patch", "run_ci", "finalize_patch"}:
                action = parsed
                break
            retry_prompt = "Your previous output was invalid JSON. Output ONLY valid JSON."

        if action is None:
            action = _safe_fallback_action({
                "available_files": list(obs.get("available_files") or ([first_file] if first_file else [])),
                "violations": list(obs.get("violations") or []),
                "file_reads_remaining": int(obs.get("file_reads_remaining", 0) or 0),
                "ci_results": list(obs.get("ci_results") or []),
                "last_file_view": obs.get("last_file_view", ""),
            })

        # Step env
        step_resp = call_env("patch/step", {"session_id": session_id, "action": action})
        obs = step_resp["observation"]
        reward = step_resp["reward"]
        reward_value = reward.get("value", 0.0) if isinstance(reward, dict) else reward
        done = step_resp["done"]
        final_reward = obs["cumulative_reward"]

        messages.append({"role": "assistant", "content": completion})
        messages.append({"role": "user", "content": f"Result: {obs['action_result'][:500]}\nCI: {obs['ci_results']}\nReward: {reward_value:+.4f}"})

        if done or action.get("action_type") == "finalize_patch":
            break

    return {
        "task_id": task_id,
        "prompt": messages[0]["content"] + "\n" + messages[1]["content"],
        "completion": "\n".join(all_completions),
        "reward": final_reward,
    }


def train():
    from unsloth import FastLanguageModel
    from trl import GRPOConfig, GRPOTrainer

    print("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=2048,
        load_in_4bit=True,
        dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "v_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    align_causal_lm_and_tokenizer(model, tokenizer, max_new_tokens=GENERATION_MAX_NEW_TOKENS)
    FastLanguageModel.for_training(model)

    # Presentation-only: per-trainer-step aggregates (does not change learning).
    reward_history: list[float] = []
    step_avg_rewards: list[float] = []
    total_generations = 0
    json_valid_count = 0
    best_batch: dict = {"success_num": 0, "success_den": 0, "avg": -1e9, "step": -1}
    best_reward_ever: dict = {"value": -1e9, "step": -1}
    grpo_step = [0]  # mutable counter for logging

    def _completion_has_valid_action_json(comp: str) -> bool:
        for match in re.finditer(r"\{[^{}]*\"action_type\"[^{}]*\}", comp, flags=re.DOTALL):
            try:
                action = json.loads(match.group(0))
            except json.JSONDecodeError:
                continue
            if isinstance(action, dict) and action.get("action_type"):
                return True
        return False

    def reward_fn(completions, **kwargs) -> List[float]:
        """Reward function called by GRPOTrainer."""
        nonlocal total_generations, json_valid_count, best_batch, best_reward_ever
        grpo_step[0] += 1
        step_n = grpo_step[0]
        rewards: List[float] = []
        task_ids = kwargs.get("task_id") or kwargs.get("task_ids") or ["task1_single_file"] * len(completions)
        if isinstance(task_ids, str):
            task_ids = [task_ids] * len(completions)

        ngen = len(completions)
        jok = 0
        for comp in completions:
            total_generations += 1
            if _completion_has_valid_action_json(comp):
                jok += 1
                json_valid_count += 1

        for i, comp in enumerate(completions):
            task_id = task_ids[i] if i < len(task_ids) else "task1_single_file"
            task = TASK_CACHE.get(task_id, TASK_CACHE["task1_single_file"])

            actions = []
            for match in re.finditer(r'\{[^{}]*"action_type"[^{}]*\}', comp, flags=re.DOTALL):
                try:
                    action = json.loads(match.group(0))
                except json.JSONDecodeError:
                    continue
                if isinstance(action, dict) and action.get("action_type"):
                    actions.append(action)

            if not actions:
                task = TASK_CACHE.get(task_id, TASK_CACHE["task1_single_file"])
                actions = [_safe_fallback_action({
                    "available_files": list(task["codebase"].keys()),
                    "violations": list(task["ground_truth"]),
                    "file_reads_remaining": int(task.get("file_reads_remaining", 5) or 0),
                    "ci_results": [],
                    "last_file_view": "",
                })]

            env = CompliancePatchEnv()
            env.reset(
                task_id=task_id,
                codebase=task["codebase"],
                violations=task["ground_truth"],
                max_steps=12,
                file_reads_remaining=task.get("file_reads_remaining", 5),
            )

            final_score = 0.0
            try:
                done = False
                info: dict = {}
                for action in actions[:12]:
                    obs, reward, done, info = env.step(action)
                    final_score = float(info.get("final_score", reward))
                    if done:
                        break

                if not done:
                    _obs, reward, done, info = env.step({"action_type": "finalize_patch"})
                    final_score = float(info.get("final_score", 0.0))

                rewards.append(final_score)
            except Exception:
                rewards.append(0.0)
        reward_history.extend(rewards)
        avg = sum(rewards) / max(1, len(rewards))
        step_avg_rewards.append(avg)
        n_ok = sum(1 for r in rewards if r > 0.0)
        if n_ok > best_batch["success_num"] or (
            n_ok == best_batch["success_num"] and avg > best_batch["avg"]
        ):
            best_batch = {"success_num": n_ok, "success_den": len(rewards), "avg": avg, "step": step_n}
        if avg > best_reward_ever["value"]:
            best_reward_ever = {"value": avg, "step": step_n}
        jrate = 100.0 * jok / max(1, ngen)
        invalid_rate = 1.0 - (jok / max(1, ngen))
        print("------------------------------------------------------------\n## GRPO BATCH (trainer step %d)\n------------------------------------------------------------" % step_n)
        print("  json=%d/%d  |  JSON validity (batch): %.1f%%" % (jok, ngen, jrate))
        print("  invalid_output_rate=%.1f%%" % (100.0 * invalid_rate))
        print("  success=%d/%d (reward>0)  |  batch_avg_reward=%+.3f" % (n_ok, len(rewards), avg))
        if invalid_rate > 0.20:
            print("High invalid output rate — training unstable")
        print("  → Model improves over time; watch batch_avg_reward trend, not a single step.")
        return rewards

    # Build training prompts from rollouts
    print("Collecting rollout prompts...")
    prompts = []
    for task_id in TASKS:
        for _ in range(8):   # 8 rollouts per task = 16 total
            try:
                result = rollout(model, tokenizer, task_id)
                prompts.append({"prompt": result["prompt"]})
                print(f"  {task_id}: reward={result['reward']:.4f}")
            except Exception as e:
                print(f"  Rollout failed: {e}")
                prompts.append({"prompt": f"Fix GDPR violations in {task_id}"})

    from datasets import Dataset
    dataset = Dataset.from_list(prompts)

    _extra_eos = json_action_eos_token_ids(tokenizer)
    _gen_kw = {"eos_token_id": _extra_eos} if _extra_eos else None
    if _extra_eos and getattr(model, "generation_config", None) is not None:
        model.generation_config.eos_token_id = _extra_eos

    config = GRPOConfig(
        output_dir="./patch_agent_model",
        max_steps=MAX_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=2,
        learning_rate=5e-6,
        logging_steps=5,
        save_steps=20,
        report_to="none",
        max_prompt_length=1536,
        max_completion_length=GENERATION_MAX_NEW_TOKENS,
        num_generations=4,
        generation_kwargs={**(_gen_kw or {}), "temperature": 0.6, "top_p": 0.9},
    )

    trainer = GRPOTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        reward_funcs=reward_fn,
        tokenizer=tokenizer,
    )

    print("------------------------------------------------------------\n## STARTING GRPO\n------------------------------------------------------------")
    print("  max_steps=%d  |  tasks=%s" % (MAX_STEPS, ", ".join(TASKS)))
    print("  Training uses multi-file scenarios: task2_django_app, task1_single_file (see TASKS in script).")
    print(f"  → Agent learns structured actions from env reward; json= line tracks parse reliability.")
    print(f"Starting GRPO training for {MAX_STEPS} steps...")
    trainer.train()

    with open("reward_curve.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "per_completion_rewards": reward_history,
                "per_step_avg_reward": step_avg_rewards,
                "json_valid_count": json_valid_count,
                "total_generations": max(1, total_generations),
            },
            f,
            indent=2,
        )
    last10 = sum(step_avg_rewards[-10:]) / min(10, len(step_avg_rewards)) if step_avg_rewards else 0.0
    jtot = max(1, total_generations)
    print("------------------------------------------------------------\n## BEST PERFORMANCE\n------------------------------------------------------------")
    print("  Success: %d/%d  (most successes in a batch during training)" % (best_batch["success_num"], best_batch["success_den"] or 1))
    print("  Reward : %+.3f  (highest batch average during training)\n" % best_reward_ever["value"])
    print("  Final performance (last 10 steps avg): %+.3f" % last10)
    print("  JSON validity rate: %.1f%%  (%d/%d)" % (100.0 * json_valid_count / jtot, json_valid_count, jtot))
    print("  → Outputs are reliable when JSON validity is high; compare batch_avg to baseline eval.")
    print("Reward curve saved: reward_curve.json  |  per-step avg points: %d" % len(step_avg_rewards))

    model.save_pretrained("patch_agent_model")
    tokenizer.save_pretrained("patch_agent_model")
    print("Model saved to patch_agent_model/")


if __name__ == "__main__":
    train()
