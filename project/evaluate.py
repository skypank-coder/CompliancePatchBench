"""
evaluate.py
===========

Run the agent over the full task suite and report:
    * average env reward
    * success rate (tasks with violations_fixed == violations_total)
    * % of violations fixed across the whole suite
    * BEFORE-vs-AFTER / hackathon-style summaries when given two report JSONs

CLI:
    # Single run (heuristic baseline)
    python -m project.evaluate run --tasks project/data/tasks.json --tag baseline

    # Compare two saved eval reports (prints BEFORE/AFTER, IMPROVEMENT, FINAL SUMMARY)
    python -m project.evaluate compare --before project/data/eval_baseline.json \\
        --after project/data/eval_trained.json --format both

    # Full submission flow: main split + hold-out generalization + optional curve PNGs
    python -m project.evaluate submission --trained-llm hf --model <adapter_or_hf_id> \\
        --write-curves

    # Plot reward, success, JSON validity from project/data/learning_curve.json
    python -m project.evaluate curves --out project/data/figures
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from environment.patch_env import CompliancePatchEnv  # noqa: E402

from .agent import (  # noqa: E402
    AgentConfig,
    ComplianceAgent,
    LLMCallable,
    TrajectoryResult,
    classify_failure_type,
    make_heuristic_backend,
    make_hf_pipeline_backend,
    make_openai_backend,
)
from .hackathon_metrics import (  # noqa: E402
    learning_curve_derivatives,
    print_baseline_trained_core,
    print_final_summary,
    print_generalization_test,
    print_improvement,
    print_interpretation_curves,
    print_interpretation_major,
    print_learning_curve_footer,
    print_multi_task_block,
)
from .utils import DATA_DIR, LEARNING_CURVE_PATH, TASKS_PATH, get_logger, read_json, write_json

log = get_logger("evaluate")


# ─── Metrics ──────────────────────────────────────────────────────────────────

def task_status(r: TrajectoryResult) -> str:
    """
    Per-task status used for failure-case visibility.

        SUCCESS — all violations fixed AND no hidden compliance violation
        PARTIAL — at least one violation fixed, OR hidden violation present
        FAIL    — no meaningful fix landed
    """
    if r.violations_total == 0:
        return "FAIL"
    fixed_all = r.violations_fixed == r.violations_total
    if fixed_all and not r.hidden_violation:
        return "SUCCESS"
    if r.violations_fixed > 0 or r.hidden_violation:
        return "PARTIAL"
    return "FAIL"


def _success_rate_for(results: List[TrajectoryResult]) -> float:
    if not results:
        return 0.0
    return round(
        sum(1 for r in results if task_status(r) == "SUCCESS") / len(results),
        4,
    )


def aggregate(results: List[TrajectoryResult]) -> Dict:
    """Compute the headline numbers from a list of trajectories.

    Includes difficulty-aware breakdowns and the cheat-resistance score so the
    benchmark can demonstrate where the agent actually fails.
    """
    n = len(results)
    if n == 0:
        return {
            "n": 0, "avg_score": 0.0, "overall_success_rate": 0.0,
            "easy_success_rate": 0.0, "medium_success_rate": 0.0, "hard_success_rate": 0.0,
            "hidden_violation_rate": 0.0, "partial_fix_rate": 0.0, "cheat_resistance": 1.0,
            "violations_fixed_pct": 0.0, "errors": 0,
        }

    avg_score = sum(r.final_score for r in results) / n
    total_v = sum(r.violations_total for r in results)
    fixed_v = sum(r.violations_fixed for r in results)
    err_count = sum(1 for r in results if r.error)
    hidden_violations = sum(1 for r in results if r.hidden_violation)
    failure_stats = {"hidden_violation": 0, "partial_fix": 0, "no_fix": 0}
    high_conf_wrong = 0
    reward_components = {
        "reward_ci": 0.0,
        "reward_minimal": 0.0,
        "reward_regression": 0.0,
        "reward_penalty": 0.0,
    }

    by_diff = {"easy": [], "medium": [], "hard": []}
    statuses = {"SUCCESS": 0, "PARTIAL": 0, "FAIL": 0}
    for r in results:
        by_diff.setdefault(r.difficulty, []).append(r)
        status = task_status(r)
        statuses[status] = statuses.get(status, 0) + 1
        failure_type = r.failure_type if r.failure_type != "none" else classify_failure_type(r)
        if failure_type in failure_stats:
            failure_stats[failure_type] += 1
        if status != "SUCCESS" and r.confidence >= 0.75:
            high_conf_wrong += 1
        for key in reward_components:
            reward_components[key] += float((r.reward_components or {}).get(key, 0.0))

    return {
        "n": n,
        "avg_score": round(avg_score, 4),
        "overall_success_rate": _success_rate_for(results),
        "easy_success_rate": _success_rate_for(by_diff.get("easy", [])),
        "medium_success_rate": _success_rate_for(by_diff.get("medium", [])),
        "hard_success_rate": _success_rate_for(by_diff.get("hard", [])),
        "hidden_violation_rate": round(hidden_violations / n, 4),
        "partial_fix_rate": round(statuses["PARTIAL"] / n, 4),
        "fail_rate": round(statuses["FAIL"] / n, 4),
        "no_fix_rate": round(failure_stats["no_fix"] / n, 4),
        "cheat_resistance": round(1 - hidden_violations / n, 4),
        "violations_total": total_v,
        "violations_fixed": fixed_v,
        "violations_fixed_pct": round((fixed_v / total_v) if total_v else 0.0, 4),
        "errors": err_count,
        "by_difficulty_counts": {d: len(rs) for d, rs in by_diff.items() if rs},
        "status_counts": statuses,
        "failure_stats": failure_stats,
        "reward_components": {k: round(v, 4) for k, v in reward_components.items()},
        "avg_confidence": round(sum(r.confidence for r in results) / n, 4),
        "high_confidence_wrong": high_conf_wrong,
        # legacy alias so existing notebooks/scripts don't break
        "success_rate": _success_rate_for(results),
    }


def summary_to_headline(s: Dict[str, Any]) -> Dict[str, float]:
    """Map evaluate() aggregate summary to hackathon headline metrics."""
    return {
        "success_rate": float(s.get("overall_success_rate", s.get("success_rate", 0.0))),
        "avg_reward": float(s.get("avg_score", 0.0)),
        "violations_fixed_pct": float(s.get("violations_fixed_pct", 0.0)),
        "hidden_violation_rate": float(s.get("hidden_violation_rate", 0.0)),
    }


def split_main_holdout(tasks: List[Dict], seed: int, holdout_frac: float) -> Tuple[List[Dict], List[Dict]]:
    """
    Disjoint main vs held-out task lists from the same JSON (generalization test).
    holdout_frac is the fraction of tasks reserved for the generalization set.
    """
    if not tasks:
        return [], []
    n = len(tasks)
    if n < 2 or holdout_frac <= 0.0:
        return list(tasks), []
    rng = random.Random(int(seed))
    order = list(range(n))
    rng.shuffle(order)
    n_hold = max(1, int(round(n * float(holdout_frac))))
    n_hold = min(n_hold, n - 1)
    hold_idx = order[:n_hold]
    main_idx = order[n_hold:]
    hold_tasks = [tasks[i] for i in hold_idx]
    main_tasks = [tasks[i] for i in main_idx]
    return main_tasks, hold_tasks


def _optional_curve_extras() -> Tuple[Optional[float], Optional[float]]:
    """valid_json_rate from last learning-curve point + last-10 avg reward (read-only)."""
    p = Path(LEARNING_CURVE_PATH)
    if not p.is_file():
        return None, None
    curve = read_json(str(p))
    if not isinstance(curve, list) or not curve:
        return None, None
    last = curve[-1]
    vj = float(last.get("valid_json_rate", 0.0) or 0.0)
    d = learning_curve_derivatives(curve)
    l10 = float(d.get("last_10_avg_reward", 0.0))
    return vj, l10


def print_hackathon_from_reports(before: Dict, after: Dict, *, same_task_set_note: str) -> None:
    """BEFORE/AFTER/IMPROVEMENT from two full evaluate() report dicts."""
    bh = summary_to_headline(before["summary"])
    ah = summary_to_headline(after["summary"])
    print_baseline_trained_core(bh, ah, same_task_set_note=same_task_set_note)
    print_improvement(bh, ah)
    print_interpretation_major()


def print_hackathon_from_compare_diff(diff: Dict) -> None:
    """Use compare() output dict (before/after summaries in diff)."""
    bh = summary_to_headline(diff["before"])
    ah = summary_to_headline(diff["after"])
    print_baseline_trained_core(
        bh, ah, same_task_set_note="(baseline and trained both evaluated on the same ordered task set)"
    )
    print_improvement(bh, ah)
    print_interpretation_major()


# ─── Single-run evaluation ────────────────────────────────────────────────────

def evaluate(
    tasks: List[Dict],
    llm: Optional[LLMCallable] = None,
    config: Optional[AgentConfig] = None,
    print_per_task: bool = True,
) -> Dict:
    """Run the agent on every task and return both per-task + aggregate metrics."""
    agent = ComplianceAgent(llm=llm or make_heuristic_backend(), config=config)
    env = CompliancePatchEnv()

    per_task: List[Dict] = []
    trajectories: List[TrajectoryResult] = []
    t0 = time.time()

    for i, task in enumerate(tasks, 1):
        t = agent.run(env, task)
        trajectories.append(t)
        status = task_status(t)
        row = {
            "task_id": t.task_id,
            "category": task.get("category"),
            "framework": task.get("framework"),
            "difficulty": t.difficulty,
            "adversarial": t.adversarial,
            "status": status,
            "final_score": round(t.final_score, 4),
            "violations_fixed": t.violations_fixed,
            "violations_total": t.violations_total,
            "success_rate": round(t.success_rate, 4),
            "hidden_violation": t.hidden_violation,
            "hidden_reason": t.hidden_reason,
            "failure_type": t.failure_type if t.failure_type != "none" else classify_failure_type(t),
            "confidence": t.confidence,
            "reward_components": t.reward_components,
            "n_steps": len(t.steps),
            "fallback_steps": sum(1 for s in t.steps if s.used_fallback),
            "actions": [s.parsed_action.get("action_type") for s in t.steps],
            "error": t.error,
        }
        per_task.append(row)
        if print_per_task:
            log.info(
                "[%3d/%d] %-40s [%s] %-7s  score=%+.3f  fixed=%d/%d  hidden=%s  steps=%d",
                i, len(tasks), t.task_id[:40], t.difficulty, status, t.final_score,
                t.violations_fixed, t.violations_total,
                "Y" if t.hidden_violation else "n", len(t.steps),
            )
            if t.hidden_violation:
                log.info("        ↳ hidden_reason: %s", t.hidden_reason[:200])

    summary = aggregate(trajectories)
    summary["wall_time_s"] = round(time.time() - t0, 2)

    return {"summary": summary, "per_task": per_task}


# ─── Comparison ───────────────────────────────────────────────────────────────

_COMPARE_KEYS = (
    "avg_score",
    "overall_success_rate",
    "easy_success_rate",
    "medium_success_rate",
    "hard_success_rate",
    "violations_fixed_pct",
    "hidden_violation_rate",
    "partial_fix_rate",
    "cheat_resistance",
)


def compare(before: Dict, after: Dict) -> Dict:
    """Pretty diff of two evaluate() outputs."""
    bs, as_ = before["summary"], after["summary"]
    delta = {k: round(as_.get(k, 0.0) - bs.get(k, 0.0), 4) for k in _COMPARE_KEYS}

    rows = []
    bm = {r["task_id"]: r for r in before["per_task"]}
    am = {r["task_id"]: r for r in after["per_task"]}
    for tid in sorted(set(bm) | set(am)):
        b, a = bm.get(tid), am.get(tid)
        rows.append({
            "task_id": tid,
            "difficulty": (b or a or {}).get("difficulty"),
            "before_score": (b or {}).get("final_score"),
            "after_score":  (a or {}).get("final_score"),
            "before_status": (b or {}).get("status"),
            "after_status":  (a or {}).get("status"),
            "before_hidden": (b or {}).get("hidden_violation"),
            "after_hidden":  (a or {}).get("hidden_violation"),
        })

    return {"before": bs, "after": as_, "delta": delta, "per_task": rows}


def print_comparison(diff: Dict) -> None:
    print("\n---\n## DETAILED METRIC TABLE (run compare)\n---")
    print(f"{'metric':<28}  {'before':>10}  {'after':>10}  {'delta':>10}")
    print("-" * 64)
    for k in _COMPARE_KEYS:
        b = float(diff["before"].get(k, 0.0))
        a = float(diff["after"].get(k, 0.0))
        d = float(diff["delta"].get(k, 0.0))
        print(f"{k:<28}  {b:>10.3f}  {a:>10.3f}  {d:>+10.3f}")
    bs = diff["before"].get("status_counts", {})
    as_ = diff["after"].get("status_counts", {})
    if bs or as_:
        print("\n---\n## STATUS COUNTS\n---")
        print(f"{'status':<12}  {'before':>8}  {'after':>8}  {'delta':>8}")
        for s in ("SUCCESS", "PARTIAL", "FAIL"):
            b = int(bs.get(s, 0))
            a = int(as_.get(s, 0))
            print(f"{s:<12}  {b:>8d}  {a:>8d}  {a - b:>+8d}")
    bsum, asum = diff["before"], diff["after"]
    before_success = float(bsum.get("overall_success_rate", 0.0))
    after_success = float(asum.get("overall_success_rate", 0.0))
    before_hidden = float(bsum.get("hidden_violation_rate", 0.0))
    after_hidden = float(asum.get("hidden_violation_rate", 0.0))
    print(
        f"\n  success_rate: {before_success:.3f} → {after_success:.3f}  |  "
        f"hidden_violation_rate: {before_hidden:.3f} → {after_hidden:.3f}"
    )


def print_summary(report: Dict) -> None:
    """Console summary of a single evaluate() report (used by CLI default path)."""
    s = report["summary"]
    print("\n------------------------------------------------------------\n## EVALUATION SUMMARY\n------------------------------------------------------------")
    n = s.get("n", 0)
    print(f"  n                      :  {n}")
    print(f"  success_rate           :  {s.get('overall_success_rate', 0.0):.3f}")
    print(f"  avg_reward (avg_score) :  {s.get('avg_score', 0.0):+.3f}")
    vfp = float(s.get("violations_fixed_pct", 0.0) or 0.0)
    print(f"  violations_fixed_pct   :  {100.0 * vfp:.1f}%")
    print(f"  hidden_violation_rate  :  {100.0 * float(s.get('hidden_violation_rate', 0.0) or 0.0):.1f}%")
    print(f"  easy_success_rate      :  {s.get('easy_success_rate', 0.0):.3f}")
    print(f"  medium_success_rate    :  {s.get('medium_success_rate', 0.0):.3f}")
    print(f"  hard_success_rate      :  {s.get('hard_success_rate', 0.0):.3f}")
    sc = s.get("status_counts", {})
    if sc:
        print(
            f"  status (S/P/F)         :  "
            f"{sc.get('SUCCESS', 0)}/{sc.get('PARTIAL', 0)}/{sc.get('FAIL', 0)}"
        )
    print(f"  errors                  :  {s.get('errors', 0)}")
    rc = s.get("reward_components", {})
    if rc:
        print(
            f"  reward_components       :  "
            f"ci={rc.get('reward_ci', 0.0):+.3f}  "
            f"minimal={rc.get('reward_minimal', 0.0):+.3f}  "
            f"regression={rc.get('reward_regression', 0.0):+.3f}  "
            f"penalty={rc.get('reward_penalty', 0.0):+.3f}"
        )
    print("  → These numbers are per-task; compare two runs (baseline vs trained) to see improvement.")
    print("------------------------------------------------------------")


def compare_iterations(metrics: List[Dict]) -> Dict:
    """Summarise RL learning-curve metrics across iterations."""
    if not metrics:
        return {"iterations": 0, "delta": {}}
    first, last = metrics[0], metrics[-1]
    delta = {
        "avg_reward": round(last.get("avg_reward", 0.0) - first.get("avg_reward", 0.0), 4),
        "success_rate": round(last.get("success_rate", 0.0) - first.get("success_rate", 0.0), 4),
        "test_success_rate": round(last.get("test_success_rate", 0.0) - first.get("test_success_rate", 0.0), 4),
        "hidden_violation_rate": round(
            last.get("hidden_violation_rate", 0.0) - first.get("hidden_violation_rate", 0.0),
            4,
        ),
        "partial_fix_rate": round(last.get("partial_fix_rate", 0.0) - first.get("partial_fix_rate", 0.0), 4),
        "recovered_tasks": int(last.get("total_recovered_tasks", 0)),
    }
    return {
        "iterations": len(metrics),
        "start": first,
        "end": last,
        "delta": delta,
        "history": metrics,
    }


def print_iteration_comparison(metrics: List[Dict], *, smooth_window: int = 6) -> None:
    """Print compact RL improvement table with smoothed reward column."""
    print("\n------------------------------------------------------------\n## RL LEARNING CURVE\n------------------------------------------------------------")
    d = learning_curve_derivatives(metrics, smooth_window=smooth_window) if metrics else {}
    sm = d.get("avg_reward_smoothed", [])
    vj = [float(m.get("valid_json_rate", 0.0) or 0.0) for m in metrics] if metrics else []
    print(
        f"{'iter':>4}  "
        f"{'avg_rew':>9}  "
        f"{'smoth':>9}  "
        f"{'vjson':>7}  "
        f"{'success':>8}  "
        f"{'test_sr':>8}  "
        f"{'hvr':>8}  "
        f"{'recov':>6}"
    )
    print("-" * 72)
    for i, m in enumerate(metrics):
        srm = sm[i] if i < len(sm) else float(m.get("avg_reward", 0.0))
        vj_i = vj[i] if i < len(vj) else 0.0
        print(
            f"{int(m.get('iteration', 0)):>4}  "
            f"{float(m.get('avg_reward', 0.0)):>+9.3f}  "
            f"{float(srm):>+9.3f}  "
            f"{vj_i:>7.1%}  "
            f"{float(m.get('train_success_rate', m.get('success_rate', 0.0))):>8.3f}  "
            f"{float(m.get('test_success_rate', 0.0)):>8.3f}  "
            f"{float(m.get('hidden_violation_rate', 0.0)):>8.3f}  "
            f"{int(m.get('recovered_tasks', 0)):>6d}"
        )
    if metrics:
        first, last = metrics[0], metrics[-1]
        bsr = float(first.get("success_rate", first.get("train_success_rate", 0.0)))
        tsr = float(last.get("success_rate", last.get("train_success_rate", 0.0)))
        bhr = float(first.get("hidden_violation_rate", 0.0))
        thr = float(last.get("hidden_violation_rate", 0.0))
        print(
            f"  first→last: success_rate {bsr:.3f} → {tsr:.3f}  |  "
            f"hidden_violation_rate {bhr:.3f} → {thr:.3f}  |  "
            f"recovered total={int(last.get('total_recovered_tasks', 0))}  |  "
            f"test_success_rate (last)={float(last.get('test_success_rate', 0.0)):.3f}"
        )
    print("  → Model improves over time: compare smoth to avg_rew; smoth is the fair trend line.")
    print("  → Agent learns structured actions when the vjson column trends up (valid JSON rate).")
    print("  Tip: run `python -m project.evaluate curves --window 6` for judge-ready PNGs.")
    print("------------------------------------------------------------")
    if metrics:
        print_learning_curve_footer(metrics, window=smooth_window)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _build_llm(args) -> LLMCallable:
    if args.llm == "openai":
        return make_openai_backend(model=args.model, temperature=0.0)
    if args.llm == "hf":
        return make_hf_pipeline_backend(args.model, temperature=0.0)
    return make_heuristic_backend()


def _make_llm(kind: str, model: str) -> LLMCallable:
    if kind == "openai":
        return make_openai_backend(model=model, temperature=0.0)
    if kind == "hf":
        return make_hf_pipeline_backend(model, temperature=0.0)
    return make_heuristic_backend()


def _write_learning_curve_figures(curve_path: Path, out_dir: Path, window: int) -> List[Path]:
    from . import plot_submission_figures as _psf  # local import: matplotlib

    data = read_json(str(curve_path)) if curve_path.is_file() else []
    if not isinstance(data, list) or not data:
        log.warning("No data at %s — skipping curve figures", curve_path)
        _psf.print_graph_summary(out_dir)
        return []
    win: Optional[int] = None if int(window) <= 0 else int(window)
    a, b, c = _psf.plot_from_learning_curve(data, out_dir, window=win, skip_existing=True)
    out = [p for p in (a, b, c) if p]
    if out:
        print()
        print_interpretation_curves()
    _psf.print_graph_summary(out_dir)
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate the compliance agent.")
    sub = p.add_subparsers(dest="cmd")

    run_p = sub.add_parser("run", help="Evaluate a single configuration")
    run_p.add_argument("--tasks", type=str, default=str(TASKS_PATH))
    run_p.add_argument("--tag", type=str, default="baseline", help="Filename tag for the saved report")
    run_p.add_argument("--llm", choices=["heuristic", "openai", "hf"], default="heuristic")
    run_p.add_argument("--model", type=str, default="gpt-4o-mini")
    run_p.add_argument("--max-steps", type=int, default=16)
    run_p.add_argument("--max-tasks", type=int, default=0)
    run_p.add_argument(
        "--list-tasks",
        action="store_true",
        help="Print task_id list and (multi-file) tags before running (readability).",
    )
    run_p.add_argument(
        "--demo-trace",
        action="store_true",
        help="Per-step judge-style log (multi-file task recommended).",
    )

    cmp_p = sub.add_parser("compare", help="Compare two saved evaluation reports")
    cmp_p.add_argument("--before", type=str, required=True)
    cmp_p.add_argument("--after", type=str, required=True)
    cmp_p.add_argument("--out", type=str, default=str(DATA_DIR / "comparison.json"))
    cmp_p.add_argument(
        "--format",
        choices=("full", "hackathon", "both"),
        default="both",
        help="full: metric table; hackathon: BEFORE/AFTER/IMPROVEMENT; both: hackathon then full",
    )
    cmp_p.add_argument(
        "--gen-before",
        type=str,
        default="",
        help="Optional: eval report JSON for generalization baseline (held-out set).",
    )
    cmp_p.add_argument(
        "--gen-after",
        type=str,
        default="",
        help="Optional: eval report JSON for generalization trained (held-out set).",
    )
    cmp_p.add_argument(
        "--gen-n-tasks",
        type=int,
        default=0,
        help="Only for printing: n tasks in the gen split (if you know it).",
    )
    cmp_p.add_argument(
        "--gen-seed",
        type=int,
        default=0,
        help="Only for printing: seed used to build the hold-out split (for labels).",
    )

    iter_p = sub.add_parser("iterations", help="Compare RL learning-curve iterations")
    iter_p.add_argument("--metrics", type=str, default=str(LEARNING_CURVE_PATH))
    iter_p.add_argument("--out", type=str, default=str(DATA_DIR / "iteration_comparison.json"))
    iter_p.add_argument(
        "--plot-out",
        type=str,
        default="",
        help="If set, also write reward/success/JSON-validity PNGs to this directory (matplotlib).",
    )
    iter_p.add_argument(
        "--window",
        type=int,
        default=0,
        help="Smoothing window for --plot-out (0 = auto: 5 or 7 from series length).",
    )

    cur_p = sub.add_parser("curves", help="Plot reward, success rate, and JSON validity from learning_curve.json")
    cur_p.add_argument("--input", type=str, default=str(LEARNING_CURVE_PATH))
    cur_p.add_argument(
        "--out",
        type=str,
        default=str(DATA_DIR / "figures"),
        help="Output directory (reward_curve.png, success_curve.png, json_validity_curve.png; skips existing).",
    )
    cur_p.add_argument(
        "--window",
        type=int,
        default=0,
        help="Moving-average window (0 = auto: 5 if n<14 else 7).",
    )

    sub_p = sub.add_parser(
        "submission",
        help="Run baseline + trained on a main task set, optional hold-out generalization, then print a summary.",
    )
    sub_p.add_argument("--tasks", type=str, default=str(TASKS_PATH))
    sub_p.add_argument("--max-steps", type=int, default=16)
    sub_p.add_argument("--max-tasks", type=int, default=0)
    sub_p.add_argument("--holdout-frac", type=float, default=0.25, help="Fraction of tasks in generalization set.")
    sub_p.add_argument("--gen-seed", type=int, default=2025, help="Random seed for main vs hold-out split.")
    sub_p.add_argument(
        "--skip-gen",
        action="store_true",
        help="Do not run the held-out generalization evaluation (faster).",
    )
    sub_p.add_argument(
        "--baseline-llm",
        choices=["heuristic", "openai", "hf"],
        default="heuristic",
    )
    sub_p.add_argument(
        "--trained-llm",
        choices=["heuristic", "openai", "hf"],
        required=True,
    )
    sub_p.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Model id for --trained-llm openai|hf (and --baseline-llm if not heuristic).",
    )
    sub_p.add_argument(
        "--baseline-model",
        type=str,
        default="",
        help="If set, overrides --model for baseline when baseline is openai|hf.",
    )
    sub_p.add_argument(
        "--tag-prefix",
        type=str,
        default="submission",
        help="eval_{prefix}_main_baseline.json, etc.",
    )
    sub_p.add_argument(
        "--learning-curve",
        type=str,
        default=str(LEARNING_CURVE_PATH),
        help="Path to learning_curve.json for optional PNG export.",
    )
    sub_p.add_argument(
        "--write-curves",
        action="store_true",
        help="Write reward / success / JSON validity PNGs to project/data/figures (needs curve file).",
    )
    sub_p.add_argument(
        "--curve-window",
        type=int,
        default=0,
        help="Smoothing for --write-curves (0 = auto: 5 or 7 from series length).",
    )
    sub_p.add_argument(
        "--curve-out",
        type=str,
        default=str(DATA_DIR / "figures"),
    )

    # Default subcommand for convenience
    p.set_defaults(cmd="run", tasks=str(TASKS_PATH), tag="baseline", llm="heuristic",
                   model="gpt-4o-mini", max_steps=16, max_tasks=0, list_tasks=False, demo_trace=False)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.cmd == "compare":
        before = read_json(args.before)
        after = read_json(args.after)
        diff = compare(before, after)
        write_json(args.out, diff)
        if args.format in ("hackathon", "both"):
            print_hackathon_from_compare_diff(diff)
        gen_bh: Optional[Dict[str, float]] = None
        gen_ah: Optional[Dict[str, float]] = None
        n_gen: Optional[int] = None
        if args.gen_before and args.gen_after:
            gbf = read_json(args.gen_before)
            gaf = read_json(args.gen_after)
            gen_bh = summary_to_headline(gbf["summary"])
            gen_ah = summary_to_headline(gaf["summary"])
            n_gen = int(args.gen_n_tasks or 0) or len(gbf.get("per_task", []) or [])
            print_generalization_test(
                gen_bh, gen_ah, n_tasks=n_gen, gen_seed=int(args.gen_seed)
            )
        s0, s1 = summary_to_headline(before["summary"]), summary_to_headline(after["summary"])
        vj_opt, l10_opt = _optional_curve_extras()
        print_final_summary(
            n_tasks=int(before["summary"].get("n", 0)),
            base_sr=s0["success_rate"],
            train_sr=s1["success_rate"],
            base_vfp=s0["violations_fixed_pct"],
            train_vfp=s1["violations_fixed_pct"],
            base_ar=s0["avg_reward"],
            train_ar=s1["avg_reward"],
            base_hvr=s0["hidden_violation_rate"],
            train_hvr=s1["hidden_violation_rate"],
            gen_base_sr=gen_bh["success_rate"] if gen_bh is not None else None,
            gen_train_sr=gen_ah["success_rate"] if gen_ah is not None else None,
            gen_base_vfp=gen_bh["violations_fixed_pct"] if gen_bh is not None else None,
            gen_train_vfp=gen_ah["violations_fixed_pct"] if gen_ah is not None else None,
            n_gen_tasks=(n_gen if (gen_bh is not None) else None),
            json_validity_pct=vj_opt,
            last_10_trained_reward=l10_opt,
        )
        if args.format in ("full", "both"):
            print_comparison(diff)
        log.info("Comparison written → %s", args.out)
        return

    if args.cmd == "iterations":
        metrics = read_json(args.metrics)
        diff = compare_iterations(metrics)
        write_json(args.out, diff)
        print_iteration_comparison(metrics, smooth_window=int(args.window))
        log.info("Iteration comparison written → %s", args.out)
        if getattr(args, "plot_out", None) and str(args.plot_out).strip():
            _write_learning_curve_figures(Path(args.metrics), Path(args.plot_out), int(args.window))
        return

    if args.cmd == "curves":
        _ = _write_learning_curve_figures(Path(args.input), Path(args.out), int(args.window))
        return

    if args.cmd == "submission":
        tasks: List[Dict] = read_json(args.tasks)
        if int(args.max_tasks) > 0:
            tasks = tasks[: int(args.max_tasks)]
        print_multi_task_block(tasks, header="Evaluation")
        main_tasks, hold_tasks = split_main_holdout(
            list(tasks), int(args.gen_seed), float(args.holdout_frac)
        )
        main_tasks = sorted(main_tasks, key=lambda t: str(t.get("task_id", "")))
        hold_tasks = sorted(hold_tasks, key=lambda t: str(t.get("task_id", "")))
        pfx = str(args.tag_prefix).replace("..", "_")
        b_model = str(args.baseline_model or args.model)
        t_model = str(args.model)
        llm_b = _make_llm(str(args.baseline_llm), b_model)
        llm_t = _make_llm(str(args.trained_llm), t_model)
        cfg = AgentConfig(max_steps=int(args.max_steps))

        log.info("MAIN eval: %d tasks (same set for baseline and trained).", len(main_tasks))
        rep_mb = evaluate(main_tasks, llm=llm_b, config=cfg, print_per_task=True)
        rep_mt = evaluate(main_tasks, llm=llm_t, config=cfg, print_per_task=True)
        write_json(DATA_DIR / f"eval_{pfx}_main_baseline.json", rep_mb)
        write_json(DATA_DIR / f"eval_{pfx}_main_trained.json", rep_mt)
        print_hackathon_from_reports(
            rep_mb, rep_mt, same_task_set_note="(baseline and trained both evaluated on the same ordered main task set)"
        )

        gen_base_h: Optional[Dict[str, float]] = None
        gen_train_h: Optional[Dict[str, float]] = None
        if not bool(args.skip_gen) and hold_tasks:
            log.info("GENERALIZATION eval: %d held-out tasks (seed=%s).", len(hold_tasks), args.gen_seed)
            rep_gb = evaluate(hold_tasks, llm=llm_b, config=cfg, print_per_task=True)
            rep_gt = evaluate(hold_tasks, llm=llm_t, config=cfg, print_per_task=True)
            write_json(DATA_DIR / f"eval_{pfx}_gen_baseline.json", rep_gb)
            write_json(DATA_DIR / f"eval_{pfx}_gen_trained.json", rep_gt)
            gen_base_h = summary_to_headline(rep_gb["summary"])
            gen_train_h = summary_to_headline(rep_gt["summary"])
            print_generalization_test(
                gen_base_h, gen_train_h, n_tasks=len(hold_tasks), gen_seed=int(args.gen_seed)
            )
        elif not bool(args.skip_gen):
            log.info("Not enough tasks for a disjoint hold-out; skipping generalization block.")

        sm = summary_to_headline(rep_mb["summary"])
        smt = summary_to_headline(rep_mt["summary"])
        vj_opt, l10_opt = _optional_curve_extras()
        print_final_summary(
            n_tasks=len(main_tasks),
            base_sr=sm["success_rate"],
            train_sr=smt["success_rate"],
            base_vfp=sm["violations_fixed_pct"],
            train_vfp=smt["violations_fixed_pct"],
            base_ar=sm["avg_reward"],
            train_ar=smt["avg_reward"],
            base_hvr=sm["hidden_violation_rate"],
            train_hvr=smt["hidden_violation_rate"],
            gen_base_sr=gen_base_h["success_rate"] if gen_base_h is not None else None,
            gen_train_sr=gen_train_h["success_rate"] if gen_train_h is not None else None,
            gen_base_vfp=gen_base_h["violations_fixed_pct"] if gen_base_h is not None else None,
            gen_train_vfp=gen_train_h["violations_fixed_pct"] if gen_train_h is not None else None,
            n_gen_tasks=(len(hold_tasks) if (hold_tasks and gen_base_h is not None) else None),
            json_validity_pct=vj_opt,
            last_10_trained_reward=l10_opt,
        )
        if bool(args.write_curves) and str(args.learning_curve):
            _write_learning_curve_figures(
                Path(args.learning_curve), Path(str(args.curve_out)), int(args.curve_window)
            )
        log.info("Saved eval_%s_main_{baseline,trained}.json", pfx)
        return

    tasks = read_json(args.tasks)
    if getattr(args, "max_tasks", 0) > 0:
        tasks = tasks[: args.max_tasks]
    if getattr(args, "list_tasks", False):
        print_multi_task_block(tasks, header="Evaluation")
    log.info("Evaluating on %d tasks (%s backend)", len(tasks), args.llm)

    llm = _build_llm(args)
    config = AgentConfig(max_steps=args.max_steps, demo_trace=bool(getattr(args, "demo_trace", False)))
    report = evaluate(tasks, llm=llm, config=config)

    out_path = DATA_DIR / f"eval_{args.tag}.json"
    write_json(out_path, report)
    log.info("Saved report → %s", out_path)
    print_summary(report)


if __name__ == "__main__":
    main()
