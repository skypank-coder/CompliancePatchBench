"""
CompliancePatchBench — production-grade self-improving compliance patcher.

Modules:
    task_generator   — Procedurally generates 30+ diverse compliance tasks.
    agent            — Robust agent loop (strict JSON, retry, fallback).
    dataset_builder  — Self-learning core: rollouts -> trajectories -> SFT dataset.
    train_model      — LoRA fine-tuning pipeline (Unsloth backend).
    rl_trainer       — RL loop: rollout -> reward-to-go advantages -> policy update.
    evaluate         — Before/after benchmark with avg score, success rate.
    utils            — Shared helpers (logging, JSON parsing, file IO).
"""

__version__ = "1.0.0"
