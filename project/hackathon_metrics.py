"""
Aggregates episode-level results for hackathon / README reporting.
Does not change the RL pipeline — only metrics derived from existing env critques.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def _row_status(e: Dict[str, Any]) -> str:
    """Match evaluate.task_status from episode dicts with critique fields."""
    vt = int(e.get("violations_total") or 0)
    if vt == 0:
        return "FAIL"
    vf = int(e.get("violations_fixed") or 0)
    hv = bool(e.get("hidden_violation"))
    if vf == vt and not hv:
        return "SUCCESS"
    if vf > 0 or hv:
        return "PARTIAL"
    return "FAIL"


def episode_summary(episodes: List[Dict[str, Any]]) -> Dict[str, float]:
    """From list of {final_score, violations_*, hidden_violation, task_id?}."""
    n = len(episodes)
    if n == 0:
        return {
            "n": 0.0,
            "success_rate": 0.0,
            "avg_reward": 0.0,
            "violations_fixed_pct": 0.0,
            "hidden_violation_rate": 0.0,
        }
    success = sum(1 for e in episodes if _row_status(e) == "SUCCESS")
    tot_v = sum(int(e.get("violations_total") or 0) for e in episodes)
    tot_f = sum(int(e.get("violations_fixed") or 0) for e in episodes)
    hvr = sum(1 for e in episodes if e.get("hidden_violation")) / n
    avg = sum(float(e.get("final_score", 0.0)) for e in episodes) / n
    vfp = (tot_f / tot_v) if tot_v else 0.0
    return {
        "n": float(n),
        "success_rate": round(success / n, 4),
        "avg_reward": round(avg, 4),
        "violations_fixed_pct": round(vfp, 4),
        "hidden_violation_rate": round(hvr, 4),
    }


def print_before_after(baseline: Dict[str, float], trained: Dict[str, float], title: str = "") -> None:
    if title:
        print(title)
    print()
    print("BEFORE (baseline):")
    print(f"  success_rate:          {baseline.get('success_rate', 0.0):.4f}")
    print(f"  avg_reward:            {baseline.get('avg_reward', 0.0):+.4f}")
    print(f"  violations_fixed_pct:  {baseline.get('violations_fixed_pct', 0.0):.4f}")
    print(f"  hidden_violation_rate: {baseline.get('hidden_violation_rate', 0.0):.4f}")
    print()
    print("AFTER (trained):")
    print(f"  success_rate:          {trained.get('success_rate', 0.0):.4f}")
    print(f"  avg_reward:            {trained.get('avg_reward', 0.0):+.4f}")
    print(f"  violations_fixed_pct:  {trained.get('violations_fixed_pct', 0.0):.4f}")
    print(f"  hidden_violation_rate: {trained.get('hidden_violation_rate', 0.0):.4f}")


def print_improvement(baseline: Dict[str, float], trained: Dict[str, float]) -> None:
    """Relative success_rate % and absolute avg_reward delta (same task set)."""
    b_sr = float(baseline.get("success_rate", 0.0) or 0.0)
    t_sr = float(trained.get("success_rate", 0.0) or 0.0)
    b_ar = float(baseline.get("avg_reward", 0.0) or 0.0)
    t_ar = float(trained.get("avg_reward", 0.0) or 0.0)
    if b_sr and b_sr > 0:
        d_sr_pct = (t_sr - b_sr) / b_sr * 100.0
    else:
        d_sr_pct = 0.0
    d_ar = t_ar - b_ar
    print()
    print("IMPROVEMENT (trained vs baseline, same task set):")
    sign_sr = "+" if d_sr_pct >= 0 else ""
    sign_ar = "+" if d_ar >= 0 else ""
    print(f"  success_rate: {sign_sr}{d_sr_pct:.1f}%  (relative)")
    print(f"  avg_reward:   {sign_ar}{d_ar:.4f}  (absolute)")


def print_multi_task_block(tasks: List[Dict[str, Any]]) -> None:
    print()
    print("=" * 60)
    n = len(tasks)
    mf = [t["task_id"] for t in tasks if len(t.get("codebase", {})) > 1]
    print(f"Training uses {n} tasks including multi-file scenarios ({len(mf)} with 2+ source files):")
    for t in tasks:
        tid = t.get("task_id", "?")
        mark = " (multi-file)" if len(t.get("codebase", {})) > 1 else ""
        print(f"  * {tid}{mark}")
    print("=" * 60)


def print_final_summary(
    n_tasks: int,
    base_sr: float,
    train_sr: float,
    base_ar: Optional[float] = None,
    train_ar: Optional[float] = None,
    gen_base_sr: Optional[float] = None,
    gen_train_sr: Optional[float] = None,
) -> None:
    print()
    print("=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    print(f"  total_tasks (main eval):  {n_tasks}")
    print(f"  baseline success_rate:    {base_sr:.4f}")
    print(f"  trained success_rate:     {train_sr:.4f}")
    if base_sr and base_sr > 0:
        imp = (train_sr - base_sr) / base_sr * 100.0
    else:
        imp = 0.0
    print(f"  improvement (success_rate):  {imp:+.1f}%  (relative vs baseline)")
    if base_ar is not None and train_ar is not None:
        print(f"  baseline avg_reward:     {base_ar:+.4f}")
        print(f"  trained avg_reward:      {train_ar:+.4f}")
        print(f"  improvement (avg_reward):  {train_ar - base_ar:+.4f}")
    if gen_base_sr is not None and gen_train_sr is not None:
        print(f"  generalization (held-out):  baseline SR {gen_base_sr:.4f}  |  trained SR {gen_train_sr:.4f}")
    print("=" * 60)
