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
from project.agent import GENERATION_MAX_NEW_TOKENS, SYSTEM_PROMPT, align_causal_lm_and_tokenizer
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

    for step in range(20):
        # Format prompt
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=GENERATION_MAX_NEW_TOKENS,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        raw = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        completion = clip_model_json_output(raw)
        all_completions.append(completion)

        # Parse action
        try:
            # Extract JSON from completion
            import re
            match = re.search(r'\{.*\}', completion, re.DOTALL)
            action = json.loads(match.group(0)) if match else {"action_type": "finalize_patch"}
        except Exception:
            action = {"action_type": "finalize_patch"}

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

    reward_history = []

    def reward_fn(completions, **kwargs) -> List[float]:
        """Reward function called by GRPOTrainer."""
        rewards = []
        task_ids = kwargs.get("task_id") or kwargs.get("task_ids") or ["task1_single_file"] * len(completions)
        if isinstance(task_ids, str):
            task_ids = [task_ids] * len(completions)

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
                rewards.append(-0.3)
                continue

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
                info = {}
                for action in actions[:12]:
                    obs, reward, done, info = env.step(action)
                    final_score = float(info.get("final_score", reward))
                    if done:
                        break

                if not done:
                    obs, reward, done, info = env.step({"action_type": "finalize_patch"})
                    final_score = float(info.get("final_score", 0.0))

                rewards.append(final_score)
            except Exception:
                rewards.append(0.0)
        reward_history.extend(rewards)
        avg = sum(rewards) / len(rewards)
        print(f"  Batch reward avg: {avg:.4f} | history avg: {sum(reward_history)/len(reward_history):.4f}")
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
    )

    trainer = GRPOTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        reward_funcs=reward_fn,
        tokenizer=tokenizer,
    )

    print(f"Starting GRPO training for {MAX_STEPS} steps...")
    trainer.train()

    # Save reward curve data
    import json
    with open("reward_curve.json", "w") as f:
        json.dump(reward_history, f)
    print(f"Reward history saved. Final avg: {sum(reward_history)/len(reward_history):.4f}")

    # Save model
    model.save_pretrained("patch_agent_model")
    tokenizer.save_pretrained("patch_agent_model")
    print("Model saved to patch_agent_model/")


if __name__ == "__main__":
    train()
