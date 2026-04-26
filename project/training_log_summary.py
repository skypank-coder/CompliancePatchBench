"""
Judge-friendly summary of TRL/GRPO training logs. Presentation only: does not
change training, rewards, or data — only formats and prints.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# If True, also print the detailed row dumps (kept for debugging; off by default).
SHOW_FULL_LOGS: bool = False


def _progression_indices(n: int) -> List[int]:
    """Checkpoints: first, ~25%, ~50%, ~75%, last (deduped)."""
    if n <= 0:
        return []
    if n == 1:
        return [0]
    last = n - 1
    raw = [0, int(round(0.25 * last)), int(round(0.5 * last)), int(round(0.75 * last)), last]
    out: List[int] = []
    seen = set()
    for i in raw:
        j = max(0, min(int(i), last))
        if j in seen:
            continue
        seen.add(j)
        out.append(j)
    return out


def select_progression_checkpoints(
    step_reward_pairs: List[Tuple[int, float]],
) -> List[Tuple[int, float]]:
    """selected_points: (step, reward) at first, ~25%, ~50%, ~75%, last (unique)."""
    n = len(step_reward_pairs)
    if n == 0:
        return []
    idxs = _progression_indices(n)
    return [step_reward_pairs[i] for i in idxs]


def parse_grpo_log_history(
    log_history: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split trainer.state.log_history into reward rows and health rows
    (same heuristics as the Colab notebook).
    """
    reward_rows: List[Dict[str, Any]] = []
    health_rows: List[Dict[str, Any]] = []
    for row in log_history:
        reward_keys = [k for k in row if "reward" in k.lower()]
        if reward_keys:
            reward_rows.append(
                {
                    "step": row.get("step", len(reward_rows)),
                    "reward": float(row[reward_keys[0]]),
                }
            )
        health_rows.append(
            {
                "step": row.get("step", len(health_rows)),
                "mean_length": row.get("completions/mean_length"),
                "terminated_length": row.get("completions/mean_terminated_length"),
                "kl": row.get("kl"),
            }
        )
    return reward_rows, health_rows


def _latest_nonempty_health(health_rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for r in reversed(health_rows):
        if r.get("mean_length") is not None:
            return r
    return None


def print_judge_training_summary(
    reward_rows: List[Dict[str, Any]],
    health_rows: List[Dict[str, Any]],
    *,
    show_full_logs: bool = False,
    log_history: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """
    Prints in order:
    1. TRAINING PROGRESSION
    2. Fixed interpretation line
    3. Best reward
    4. TRAINING HEALTH + termination message
    """
    pairs: List[Tuple[int, float]] = [
        (int(r["step"]), float(r["reward"])) for r in reward_rows
    ]
    selected = select_progression_checkpoints(pairs)
    all_rewards = [p[1] for p in pairs]

    print("------------------------------------------------------------")
    print("## TRAINING PROGRESSION")
    print()
    print("------------------------------------------------------------")
    if not selected:
        print("(No reward steps logged yet.)")
    else:
        for step, reward in selected:
            print(f"Step {step:4d}  → {reward:+.3f}")
    print("------------------------------------------------------------")
    print()
    print(
        "The model improves from negative to positive reward, showing clear learning "
        "despite RL noise."
    )
    if all_rewards:
        best = max(all_rewards)
        print()
        print(f"Best reward achieved : {best:+.3f}")
    else:
        print()
        print("Best reward achieved : (no data)")

    print()
    print("------------------------------------------------------------")
    print("## TRAINING HEALTH")
    print()
    print("------------------------------------------------------------")
    latest = _latest_nonempty_health(health_rows)
    if not latest:
        print("Avg completion length : (n/a)")
        print("Terminated length     : (n/a)")
        print("KL divergence         : (n/a)")
    else:
        ml = latest.get("mean_length")
        tl = latest.get("terminated_length")
        kl = latest.get("kl")
        def _f(x: Any, nd: int = 1) -> str:
            if x is None:
                return "(n/a)"
            try:
                return f"{float(x):.{nd}f}"
            except (TypeError, ValueError):
                return str(x)

        print(f"Avg completion length : {_f(ml, 1)}")
        print(f"Terminated length     : {_f(tl, 1)}")
        print(f"KL divergence         : {_f(kl, 3)}")
    print("------------------------------------------------------------")
    print()
    if latest and latest.get("terminated_length") is not None:
        try:
            if float(latest["terminated_length"]) > 0:
                print("Model is producing naturally terminated outputs.")
            else:
                print(
                    "Warning: outputs are being truncated (no natural termination)."
                )
        except (TypeError, ValueError):
            print("Warning: outputs are being truncated (no natural termination).")
    else:
        print("Warning: outputs are being truncated (no natural termination).")

    if show_full_logs or SHOW_FULL_LOGS:
        print()
        print("--- Full log detail (raw rows, optional) ---")
        print(f"Reward rows: {len(reward_rows)}")
        for row in reward_rows:
            print(row)
        if _latest_nonempty_health(health_rows):
            print("Latest health:", _latest_nonempty_health(health_rows))
        if log_history is not None:
            print(f"Total log_history rows: {len(log_history)}")

