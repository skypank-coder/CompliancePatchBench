"""
dataset_builder.py
==================

Self-learning core. Pipeline:

    tasks.json
        │
        ▼
    [ rollouts ]   ← `ComplianceAgent` (heuristic OR LLM) runs each task N times
        │
        ▼
    trajectories.jsonl   ← (prompt, completion, action, reward, final_score)
        │
        ▼
    [ reward filter ]    ← keep top-quality trajectories only
        │
        ▼
    dataset.jsonl        ← {"input": <task+violations prompt>,
                            "output": <chat-formatted action sequence>}
                           ready for SFT / DPO / GRPO

The output schema is a chat-style messages list, so it drops straight into
TRL's `SFTTrainer` (`dataset_text_field=None, formatting_func=...`) or you
can use the convenience `sft_examples` field for one-line preprocessing.

CLI:
    python -m project.dataset_builder --tasks project/data/tasks.json \\
        --rollouts 1 --min-success 0.66 --out project/data/dataset.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

# Allow running both as a module (python -m project.dataset_builder) and from
# the repo root with the existing environment package on PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from environment.patch_env import CompliancePatchEnv  # noqa: E402

from .agent import (  # noqa: E402
    AgentConfig,
    ComplianceAgent,
    LLMCallable,
    StepRecord,
    SYSTEM_PROMPT,
    TrajectoryResult,
    make_heuristic_backend,
    make_openai_backend,
)
from .utils import (  # noqa: E402
    DATASET_PATH,
    TASKS_PATH,
    TRAJECTORIES_PATH,
    TRAJECTORIES_RL_PATH,
    append_jsonl,
    get_logger,
    read_json,
    write_jsonl,
)

log = get_logger("dataset_builder")


# ─── Filtering ────────────────────────────────────────────────────────────────

def trajectory_quality(t: TrajectoryResult) -> float:
    """
    Heuristic 0-1 quality score blending env reward, success rate, and several
    *anti-cheat* signals. Higher = better example for SFT.

    Penalties:
      - hidden_violation     → heavy (0.5) — never want to teach a cheat
      - bloated diffs        → small      — discourages spurious write_patch calls
      - fallback / retry     → small      — discourages unstable rollouts
    """
    if t.error or not t.steps:
        return 0.0

    sr = t.success_rate
    fallback_penalty = sum(1 for s in t.steps if s.used_fallback) / max(len(t.steps), 1)
    retry_penalty = min(1.0, sum(s.retries for s in t.steps) / (3 * max(len(t.steps), 1)))

    # Penalise unnecessary write_patch calls beyond the number of violations.
    n_patches = sum(1 for s in t.steps if (s.parsed_action or {}).get("action_type") == "write_patch")
    bloat_ratio = max(0, n_patches - max(1, t.violations_total)) / max(1, t.violations_total)
    bloat_penalty = min(1.0, 0.25 * bloat_ratio)

    base = 0.7 * sr + 0.3 * min(1.0, max(0.0, t.final_score))
    score = base - 0.15 * fallback_penalty - 0.10 * retry_penalty - bloat_penalty

    # Heavy penalty for cheats — even a "successful" hidden-violation trajectory
    # should not look attractive to the dataset filter.
    if t.hidden_violation:
        score -= 0.5
    return max(0.0, score)


def keep_trajectory(
    t: TrajectoryResult,
    min_success: float = 0.66,
    min_quality: float = 0.4,
) -> bool:
    """Decide whether a trajectory is good enough to enter the SFT dataset.

    A trajectory is rejected if ANY of these are true:
      * the rollout errored
      * the task had no violations to fix
      * it triggered the hidden compliance oracle (cheat fix)
      * its violations_fixed/violations_total < `min_success`
      * its quality score (see trajectory_quality) < `min_quality`
    """
    if t.error:
        return False
    if t.violations_total == 0:
        return False
    # Hard reject any cheat trajectory — dataset must reflect ONLY correct fixes.
    if t.hidden_violation:
        return False
    if t.success_rate < min_success:
        return False
    if trajectory_quality(t) < min_quality:
        return False
    return True


# ─── Conversion ───────────────────────────────────────────────────────────────

def trajectory_to_sft_messages(t: TrajectoryResult) -> List[Dict[str, str]]:
    """
    Render a trajectory into a chat-style messages list suitable for SFT.

    The conversation is reconstructed from the stored prompts:
      [system, user, assistant(action_1), user(obs_1), assistant(action_2), ...]
    Only ACCEPTED actions (the ones we actually executed) appear as assistant
    turns — that's what we want the model to learn to emit.
    """
    if not t.steps:
        return []

    msgs: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    msgs.append({"role": "user", "content": t.steps[0].prompt})

    for i, s in enumerate(t.steps):
        msgs.append({"role": "assistant", "content": json.dumps(s.parsed_action, ensure_ascii=False)})
        # Insert observation as user message — except after the final step
        if i < len(t.steps) - 1:
            obs_str = json.dumps(s.observation, ensure_ascii=False)
            msgs.append({"role": "user", "content": f"OBSERVATION:\n{obs_str}"})
    return msgs


def trajectory_to_sft_text(t: TrajectoryResult) -> str:
    """
    Plain-text rendering of the same messages, for trainers that want a
    single `text` field. Uses a simple `<|role|>` delimiter that's tokenizer-agnostic.
    """
    pieces = []
    for m in trajectory_to_sft_messages(t):
        pieces.append(f"<|{m['role']}|>\n{m['content']}")
    pieces.append("<|end|>")
    return "\n".join(pieces)


def build_sft_record(task: Dict, t: TrajectoryResult) -> Dict:
    """Combine task metadata + chosen trajectory into one SFT row."""
    msgs = trajectory_to_sft_messages(t)
    return {
        "task_id": t.task_id,
        "category": task.get("category"),
        "framework": task.get("framework"),
        "difficulty": task.get("difficulty", "easy"),
        "adversarial": bool(task.get("adversarial", False)),
        "input": msgs[1]["content"] if len(msgs) > 1 else "",
        "output": "\n".join(m["content"] for m in msgs if m["role"] == "assistant"),
        "messages": msgs,
        "text": trajectory_to_sft_text(t),
        "final_score": t.final_score,
        "success_rate": t.success_rate,
        "quality": round(trajectory_quality(t), 4),
        "n_steps": len(t.steps),
        # Surface for audits — these will always be False/ok in the kept set
        # (hidden trajectories are filtered) but it makes downstream analysis easier.
        "hidden_violation": t.hidden_violation,
        "hidden_reason": t.hidden_reason,
    }


# ─── Pipeline ────────────────────────────────────────────────────────────────

def run_rollouts(
    tasks: List[Dict],
    n_rollouts_per_task: int = 1,
    llm: Optional[LLMCallable] = None,
    config: Optional[AgentConfig] = None,
    trajectories_path: Path = TRAJECTORIES_PATH,
    trajectories_rl_path: Path = TRAJECTORIES_RL_PATH,
) -> List[TrajectoryResult]:
    """Roll out the agent over every (task, sample) pair and persist trajectories."""
    agent = ComplianceAgent(llm=llm or make_heuristic_backend(), config=config)
    env = CompliancePatchEnv()

    # Wipe the trajectory log so reruns don't accumulate stale data
    if trajectories_path.exists():
        trajectories_path.unlink()
    if trajectories_rl_path.exists():
        trajectories_rl_path.unlink()

    trajectories: List[TrajectoryResult] = []
    t0 = time.time()
    total = len(tasks) * n_rollouts_per_task
    done = 0

    for task in tasks:
        for sample_idx in range(n_rollouts_per_task):
            done += 1
            t = agent.run(env, task)
            trajectories.append(t)
            row = t.to_dict()
            row["sample_idx"] = sample_idx
            row["quality"] = round(trajectory_quality(t), 4)
            append_jsonl(trajectories_path, row)
            append_jsonl(trajectories_rl_path, {
                "task_id": t.task_id,
                "sample_idx": sample_idx,
                "difficulty": t.difficulty,
                "adversarial": t.adversarial,
                "final_score": t.final_score,
                "success_rate": t.success_rate,
                "violations_fixed": t.violations_fixed,
                "violations_total": t.violations_total,
                "hidden_violation": t.hidden_violation,
                "hidden_reason": t.hidden_reason,
                "cumulative_reward": round(sum(s.reward for s in t.rl_trajectory), 4),
                "trajectory": [s.__dict__ for s in t.rl_trajectory],
            })
            log.info(
                "[%3d/%d] %-40s  score=%.3f  fixed=%d/%d  quality=%.2f  err=%s",
                done, total, task["task_id"][:40],
                t.final_score, t.violations_fixed, t.violations_total,
                trajectory_quality(t),
                t.error or "-",
            )
    log.info("Rollouts complete in %.1fs. Wrote %d trajectories → %s",
             time.time() - t0, len(trajectories), trajectories_path)
    log.info("RL trajectories written → %s", trajectories_rl_path)
    return trajectories


def filter_and_export(
    tasks: List[Dict],
    trajectories: List[TrajectoryResult],
    out_path: Path = DATASET_PATH,
    min_success: float = 0.66,
    min_quality: float = 0.4,
) -> Dict:
    """Filter trajectories by quality and export the SFT dataset."""
    by_task: Dict[str, Dict] = {t["task_id"]: t for t in tasks}

    kept: List[Dict] = []
    skipped = 0
    skipped_hidden = 0
    for t in trajectories:
        task = by_task.get(t.task_id)
        if task is None:
            continue
        if t.hidden_violation:
            skipped_hidden += 1
        if not keep_trajectory(t, min_success=min_success, min_quality=min_quality):
            skipped += 1
            continue
        kept.append(build_sft_record(task, t))

    # Best-of-N: keep only the top-quality trajectory per task to dedupe
    deduped: Dict[str, Dict] = {}
    for row in kept:
        existing = deduped.get(row["task_id"])
        if not existing or row["quality"] > existing["quality"]:
            deduped[row["task_id"]] = row
    rows = list(deduped.values())
    rows.sort(key=lambda r: -r["quality"])

    write_jsonl(out_path, rows)
    summary = {
        "total_trajectories": len(trajectories),
        "kept_after_filter": len(kept),
        "kept_after_dedupe": len(rows),
        "skipped": skipped,
        "skipped_hidden_violation": skipped_hidden,
        "min_success": min_success,
        "min_quality": min_quality,
        "out_path": str(out_path),
    }
    log.info("Dataset export: %s", json.dumps(summary, indent=2))
    return summary


# ─── CLI ─────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build an SFT dataset by running environment rollouts.")
    p.add_argument("--tasks", type=str, default=str(TASKS_PATH), help="Path to tasks.json")
    p.add_argument("--out", type=str, default=str(DATASET_PATH), help="Output JSONL path for the SFT dataset")
    p.add_argument("--trajectories", type=str, default=str(TRAJECTORIES_PATH),
                   help="Path to write the full trajectory log (JSONL)")
    p.add_argument("--trajectories-rl", type=str, default=str(TRAJECTORIES_RL_PATH),
                   help="Path to write RL transition trajectories (JSONL)")
    p.add_argument("--rollouts", type=int, default=1, help="Rollouts per task (>1 enables best-of-N)")
    p.add_argument("--min-success", type=float, default=0.66, help="Min violations-fixed ratio to keep")
    p.add_argument("--min-quality", type=float, default=0.4, help="Min blended quality score to keep")
    p.add_argument("--max-tasks", type=int, default=0, help="If >0, only run the first N tasks (debug)")
    p.add_argument("--llm", choices=["heuristic", "openai"], default="heuristic",
                   help="Backend for rollouts. 'openai' needs OPENAI_API_KEY/HF_TOKEN + MODEL_NAME.")
    p.add_argument("--model", type=str, default="gpt-4o-mini")
    p.add_argument("--max-steps", type=int, default=16)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    tasks = read_json(args.tasks)
    if args.max_tasks > 0:
        tasks = tasks[: args.max_tasks]
    log.info("Loaded %d tasks from %s", len(tasks), args.tasks)

    if args.llm == "openai":
        llm = make_openai_backend(model=args.model, temperature=0.0)
    else:
        llm = make_heuristic_backend()

    config = AgentConfig(max_steps=args.max_steps)
    trajectories = run_rollouts(
        tasks,
        n_rollouts_per_task=args.rollouts,
        llm=llm,
        config=config,
        trajectories_path=Path(args.trajectories),
        trajectories_rl_path=Path(args.trajectories_rl),
    )
    summary = filter_and_export(
        tasks,
        trajectories,
        out_path=Path(args.out),
        min_success=args.min_success,
        min_quality=args.min_quality,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
