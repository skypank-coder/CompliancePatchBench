"""
GRPO Training Script for CompliancePatchBench patch agent.
Run on Google Colab with a T4 GPU.
Trains Qwen2.5-1.5B-Instruct to write compliance patches via GRPO.

Usage:
  pip install unsloth trl requests
  export ENV_BASE_URL=https://your-space.hf.space  # or http://localhost:7860
  python tools/grpo_training.py
"""

import os
import json
import requests
import torch
from typing import List

ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")
MODEL_NAME = "unsloth/Qwen2.5-1.5B-Instruct"
MAX_STEPS = 60
BATCH_SIZE = 4
TASKS = ["task1_single_file", "task2_django_app"]


SYSTEM_PROMPT = """You are a security compliance engineer. You receive a Python codebase with known GDPR/OWASP violations.
Your job: write a minimal patch that fixes each violation without breaking existing code.

You interact via JSON actions:
{"action_type": "read_file", "path": "filename.py"}
{"action_type": "write_patch", "file": "filename.py", "line_start": 45, "line_end": 47, "new_code": "replacement code here"}
{"action_type": "run_ci"}
{"action_type": "finalize_patch"}

Rules:
- Read the file first to understand context
- Write the minimal fix — never delete the flagged line entirely
- run_ci after patching to check your work
- finalize_patch when done

Respond with ONE JSON action per turn."""


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
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        completion = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
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
        done = step_resp["done"]
        final_reward = obs["cumulative_reward"]

        messages.append({"role": "assistant", "content": completion})
        messages.append({"role": "user", "content": f"Result: {obs['action_result'][:500]}\nCI: {obs['ci_results']}\nReward: {reward:+.4f}"})

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
    FastLanguageModel.for_training(model)

    reward_history = []

    def reward_fn(completions, **kwargs) -> List[float]:
        """Reward function called by GRPOTrainer."""
        rewards = []
        for comp in completions:
            # Extract reward from the last env step embedded in completion metadata
            # For GRPO: reward is passed via the rollout above
            # Here we use a simple heuristic: if finalize_patch appears and CI PASS in text
            if "finalize_patch" in comp and "ci_pass" in comp.lower():
                rewards.append(1.5)
            elif "ci_pass" in comp.lower():
                rewards.append(0.8)
            elif "finalize_patch" in comp:
                rewards.append(0.1)
            else:
                rewards.append(-0.1)
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
        max_completion_length=256,
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
