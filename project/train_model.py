"""
train_model.py
==============

LoRA fine-tuning pipeline for CompliancePatchBench.

Default backend: Unsloth (fastest 4-bit LoRA on a single T4/A10/A100).
Falls back to vanilla `transformers + peft + trl` if Unsloth isn't installed —
useful on macOS / non-CUDA dev boxes for a dry-run check.

Inputs:
    project/data/dataset.jsonl   (output of dataset_builder.py)

Outputs:
    project/data/lora_adapter/   (the trained LoRA adapter)
    project/data/training_log.json

CLI:
    python -m project.train_model \\
        --dataset project/data/dataset.jsonl \\
        --base-model unsloth/mistral-7b-instruct-v0.3-bnb-4bit \\
        --epochs 2 --batch-size 2 --lr 2e-4

Budget notes (~$30 / single T4 Colab):
    * Mistral-7B-Instruct + LoRA r=16, alpha=32, 1 epoch on ~30 trajectories
      takes ≈ 5–10 min on T4. Two epochs ≈ 15 min.
    * If you only have CPU, the script will refuse to call SFTTrainer (it'd
      take days) and instead emit a tiny tokenisation-only dry-run so you can
      verify the dataset shape end-to-end.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .utils import DATA_DIR, DATASET_PATH, get_logger, read_jsonl, write_json

log = get_logger("train_model")

DEFAULT_BASE_MODEL = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
ADAPTER_DIR = DATA_DIR / "lora_adapter"
TRAINING_LOG = DATA_DIR / "training_log.json"


# ─── Config ───────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    base_model: str = DEFAULT_BASE_MODEL
    dataset_path: str = str(DATASET_PATH)
    output_dir: str = str(ADAPTER_DIR)
    epochs: int = 2
    batch_size: int = 2
    grad_accum: int = 4
    learning_rate: float = 2e-4
    max_seq_length: int = 4096
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    seed: int = 42
    warmup_ratio: float = 0.05
    save_steps: int = 50
    logging_steps: int = 5


# ─── Dataset loading ──────────────────────────────────────────────────────────

def load_sft_dataset(path: str):
    """
    Load the JSONL produced by dataset_builder and convert to a HF Dataset.
    Each row exposes both `messages` (chat format) and `text` (flat format)
    so trainers with different APIs can both consume it.
    """
    try:
        from datasets import Dataset
    except ImportError as e:
        raise RuntimeError("Install `datasets` to train.") from e

    rows = list(read_jsonl(Path(path)))
    if not rows:
        raise RuntimeError(f"No SFT rows found in {path}. Did dataset_builder produce examples?")
    log.info("Loaded %d SFT rows from %s", len(rows), path)

    keep_cols = ["task_id", "category", "input", "output", "messages", "text", "final_score", "quality"]
    cleaned = [{k: r.get(k) for k in keep_cols} for r in rows]
    return Dataset.from_list(cleaned)


# ─── Unsloth path (preferred) ─────────────────────────────────────────────────

def train_with_unsloth(cfg: TrainConfig) -> Dict:
    """Fastest path on a CUDA box. Returns a small JSON-serialisable summary."""
    from unsloth import FastLanguageModel  # noqa: F401  (heavy import — keep lazy)
    from trl import SFTConfig, SFTTrainer
    import torch

    log.info("Loading base model (4-bit): %s", cfg.base_model)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.base_model,
        max_seq_length=cfg.max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=cfg.seed,
    )

    ds = load_sft_dataset(cfg.dataset_path)

    # Render `text` per-row using the base model's chat template if available,
    # otherwise fall back to the flat <|role|> rendering already in the row.
    def _render(row):
        try:
            text = tokenizer.apply_chat_template(
                row["messages"], tokenize=False, add_generation_prompt=False,
            )
            return {"text": text}
        except Exception:
            return {"text": row["text"]}

    ds = ds.map(_render)

    sft_args = SFTConfig(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        warmup_ratio=cfg.warmup_ratio,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=cfg.logging_steps,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=cfg.seed,
        save_steps=cfg.save_steps,
        save_total_limit=2,
        report_to="none",
        max_seq_length=cfg.max_seq_length,
        packing=False,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        args=sft_args,
    )

    log.info("Starting training: %d examples × %d epochs", len(ds), cfg.epochs)
    train_result = trainer.train()

    log.info("Saving LoRA adapter → %s", cfg.output_dir)
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    return {
        "backend": "unsloth",
        "base_model": cfg.base_model,
        "examples": len(ds),
        "metrics": getattr(train_result, "metrics", {}),
        "config": asdict(cfg),
    }


# ─── Fallback: plain transformers + peft (CPU dry-run safe) ───────────────────

def train_with_peft(cfg: TrainConfig) -> Dict:
    """
    Slow but portable: works without Unsloth.
    On CPU, we just do a tokenisation pass + 1 forward step so callers can
    sanity-check the dataset shape end-to-end. On CUDA, we run a real (small)
    SFT loop.
    """
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    log.info("Loading base model (peft path): %s", cfg.base_model)
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(cfg.base_model, torch_dtype=dtype)
    if torch.cuda.is_available():
        model = model.to("cuda")

    peft_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    ds = load_sft_dataset(cfg.dataset_path)
    ds = ds.map(lambda r: {"text": r["text"]})

    if not torch.cuda.is_available():
        # Dry run only — make sure tokenisation works
        sample = tokenizer(ds[0]["text"], truncation=True, max_length=cfg.max_seq_length, return_tensors="pt")
        with torch.no_grad():
            _ = model(**sample)
        log.warning("CUDA not available — performed a dry-run forward only (no training).")
        return {
            "backend": "peft-dryrun",
            "base_model": cfg.base_model,
            "examples": len(ds),
            "note": "No CUDA detected; ran tokenisation + 1 forward pass for sanity-check.",
            "config": asdict(cfg),
        }

    sft_args = SFTConfig(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=2,
        warmup_ratio=cfg.warmup_ratio,
        report_to="none",
        max_seq_length=cfg.max_seq_length,
        dataset_text_field="text",
    )
    trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=ds, args=sft_args)
    train_result = trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    return {
        "backend": "peft",
        "base_model": cfg.base_model,
        "examples": len(ds),
        "metrics": getattr(train_result, "metrics", {}),
        "config": asdict(cfg),
    }


# ─── Entry point ──────────────────────────────────────────────────────────────

def train(cfg: TrainConfig) -> Dict:
    """Pick the best available backend and train."""
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    try:
        import unsloth  # noqa: F401
        summary = train_with_unsloth(cfg)
    except ImportError:
        log.warning("Unsloth not installed — falling back to peft+trl.")
        summary = train_with_peft(cfg)

    write_json(TRAINING_LOG, summary)
    log.info("Training complete. Summary → %s", TRAINING_LOG)
    return summary


def _parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="LoRA fine-tune a base model on the self-built SFT dataset.")
    p.add_argument("--base-model", type=str, default=DEFAULT_BASE_MODEL)
    p.add_argument("--dataset", type=str, default=str(DATASET_PATH))
    p.add_argument("--output", type=str, default=str(ADAPTER_DIR))
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-seq-length", type=int, default=4096)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    a = p.parse_args()
    return TrainConfig(
        base_model=a.base_model, dataset_path=a.dataset, output_dir=a.output,
        epochs=a.epochs, batch_size=a.batch_size, grad_accum=a.grad_accum,
        learning_rate=a.lr, max_seq_length=a.max_seq_length,
        lora_r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=a.lora_dropout,
    )


def main() -> None:
    cfg = _parse_args()
    summary = train(cfg)
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=2))


if __name__ == "__main__":
    main()
