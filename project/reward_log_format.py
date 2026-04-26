"""
Read-only pretty-printing for per-batch reward lists (e.g. GRPO reward_fn logs).
Does not change reward values — only labels for human interpretation.
"""
from __future__ import annotations

from typing import List, Optional, Tuple


def describe_reward_label(r: float) -> str:
    """
    Map a scalar env reward to a short semantic tag (for logs only).
    """
    if r <= -0.999:
        return "invalid JSON / no valid action"
    if r < 0.0:
        return "penalty / bad patch"
    if r < 0.1:
        return "negligible / near-zero"
    if r < 0.5:
        return "partial fix"
    if r < 1.0:
        return "good fix"
    return "full fix"


def format_annotated_reward(
    r: float,
    *,
    decimals: int = 3,
    violation_pair: Optional[Tuple[int, int]] = None,
) -> str:
    """One reward, e.g. -1.000 (invalid JSON / no valid action)."""
    body = f"{r:+.{decimals}f} ({describe_reward_label(r)})"
    if violation_pair is not None:
        a, b = violation_pair
        return f"{r:+.{decimals}f} ({a}/{b} viol. fixed) ({describe_reward_label(r)})"
    return body


def format_grpo_batch_log_line(
    batch_avg: float,
    n_success: int,
    rewards: List[float],
    *,
    decimals: int = 3,
    violation_pairs: Optional[List[Optional[Tuple[int, int]]]] = None,
) -> str:
    """
    One log line: batch mean, success count, annotated reward list.
    (Optional violation_pairs aligned with rewards if available from the env.)
    """
    n = len(rewards)
    parts: List[str] = []
    for i, r in enumerate(rewards):
        vp: Optional[Tuple[int, int]] = None
        if violation_pairs and i < len(violation_pairs):
            vp = violation_pairs[i]
        parts.append(format_annotated_reward(r, decimals=decimals, violation_pair=vp))
    inner = ", ".join(parts)
    return (
        f"  Batch avg={batch_avg:+.{decimals}f} | success={n_success}/{n} | [{inner}]"
    )
