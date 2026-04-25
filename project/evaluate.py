"""
evaluate.py
===========

Run the agent over the full task suite and report:
    * average env reward
    * success rate (tasks with violations_fixed == violations_total)
    * % of violations fixed across the whole suite
    * BEFORE-vs-AFTER comparison when given two runs (e.g. base model vs LoRA)

CLI:
    # Single run (heuristic baseline)
    python -m project.evaluate --tasks project/data/tasks.json --tag baseline

    # Compare two runs
    python -m project.evaluate compare \\
        --before project/data/eval_baseline.json \\
        --after  project/data/eval_lora.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

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
    print("\n=== BEFORE vs AFTER ===")
    print(f"{'metric':<28}{'before':>12}{'after':>12}{'delta':>12}")
    print("-" * 64)
    for k in _COMPARE_KEYS:
        b = diff["before"].get(k, 0.0)
        a = diff["after"].get(k, 0.0)
        d = diff["delta"].get(k, 0.0)
        sign = "+" if d >= 0 else ""
        print(f"{k:<28}{b:>12.4f}{a:>12.4f}{sign + str(round(d, 4)):>12}")
    # Show how the SUCCESS / PARTIAL / FAIL split shifted.
    bs = diff["before"].get("status_counts", {})
    as_ = diff["after"].get("status_counts", {})
    if bs or as_:
        print("\n=== STATUS COUNTS ===")
        print(f"{'status':<12}{'before':>12}{'after':>12}{'delta':>12}")
        for s in ("SUCCESS", "PARTIAL", "FAIL"):
            b = bs.get(s, 0)
            a = as_.get(s, 0)
            print(f"{s:<12}{b:>12d}{a:>12d}{(a-b):>+12d}")

    bsum, asum = diff["before"], diff["after"]
    before_success = float(bsum.get("overall_success_rate", 0.0))
    after_success = float(asum.get("overall_success_rate", 0.0))
    before_hidden = float(bsum.get("hidden_violation_rate", 0.0))
    after_hidden = float(asum.get("hidden_violation_rate", 0.0))
    print(f"\nSuccess: {before_success:.0%} → {after_success:.0%}")
    print(f"Hidden violations: {before_hidden:.0%} → {after_hidden:.0%}")


def print_summary(report: Dict) -> None:
    """Console summary of a single evaluate() report (used by CLI default path)."""
    s = report["summary"]
    print("\n=== EVALUATION SUMMARY ===")
    print(f"  tasks evaluated:        {s.get('n', 0)}")
    print(f"  avg score:              {s.get('avg_score', 0.0):+.4f}")
    print(f"  overall success rate:   {s.get('overall_success_rate', 0.0):.2%}")
    print(f"    easy   success rate:  {s.get('easy_success_rate', 0.0):.2%}")
    print(f"    medium success rate:  {s.get('medium_success_rate', 0.0):.2%}")
    print(f"    hard   success rate:  {s.get('hard_success_rate', 0.0):.2%}")
    print(f"  partial fix rate:       {s.get('partial_fix_rate', 0.0):.2%}")
    print(f"  hidden violation rate:  {s.get('hidden_violation_rate', 0.0):.2%}")
    print(f"  no-fix rate:            {s.get('no_fix_rate', 0.0):.2%}")
    print(f"  cheat resistance:       {s.get('cheat_resistance', 0.0):.2%}")
    print(f"  violations fixed pct:   {s.get('violations_fixed_pct', 0.0):.2%}")
    print(f"  avg confidence:         {s.get('avg_confidence', 0.0):.2%}")
    print(f"  high-conf wrong:        {s.get('high_confidence_wrong', 0)}")
    print(f"  errors:                 {s.get('errors', 0)}")
    sc = s.get("status_counts", {})
    if sc:
        print(f"  status counts:          SUCCESS={sc.get('SUCCESS', 0)}  "
              f"PARTIAL={sc.get('PARTIAL', 0)}  FAIL={sc.get('FAIL', 0)}")
    fs = s.get("failure_stats", {})
    if fs:
        print(f"  failure stats:          hidden={fs.get('hidden_violation', 0)}  "
              f"partial={fs.get('partial_fix', 0)}  no_fix={fs.get('no_fix', 0)}")
    rc = s.get("reward_components", {})
    if rc:
        print(f"  reward components:      ci={rc.get('reward_ci', 0.0):+.3f}  "
              f"minimal={rc.get('reward_minimal', 0.0):+.3f}  "
              f"regression={rc.get('reward_regression', 0.0):+.3f}  "
              f"penalty={rc.get('reward_penalty', 0.0):+.3f}")


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


def print_iteration_comparison(metrics: List[Dict]) -> None:
    """Print compact RL improvement table."""
    print("\n=== RL LEARNING CURVE ===")
    print(f"{'iter':>4}  {'reward':>10}  {'train':>10}  {'test':>10}  {'hidden':>10}  {'partial':>10}  {'recovered':>10}")
    print("-" * 82)
    for m in metrics:
        print(
            f"{int(m.get('iteration', 0)):>4}  "
            f"{m.get('avg_reward', 0.0):>+10.4f}  "
            f"{m.get('train_success_rate', m.get('success_rate', 0.0)):>9.2%}  "
            f"{m.get('test_success_rate', 0.0):>9.2%}  "
            f"{m.get('hidden_violation_rate', 0.0):>9.2%}  "
            f"{m.get('partial_fix_rate', 0.0):>9.2%}  "
            f"{int(m.get('recovered_tasks', 0)):>10d}"
        )
    if metrics:
        first, last = metrics[0], metrics[-1]
        before_success = float(first.get("success_rate", first.get("train_success_rate", 0.0)))
        after_success = float(last.get("success_rate", last.get("train_success_rate", 0.0)))
        before_hidden = float(first.get("hidden_violation_rate", 0.0))
        after_hidden = float(last.get("hidden_violation_rate", 0.0))
        print(f"Success: {before_success:.0%} → {after_success:.0%}")
        print(f"Hidden violations: {before_hidden:.0%} → {after_hidden:.0%}")
        print(f"Recovered {int(last.get('total_recovered_tasks', 0))} previously failed tasks")
        print(f"Final test_success_rate: {last.get('test_success_rate', 0.0):.0%}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _build_llm(args) -> LLMCallable:
    if args.llm == "openai":
        return make_openai_backend(model=args.model, temperature=0.0)
    if args.llm == "hf":
        return make_hf_pipeline_backend(args.model, temperature=0.0)
    return make_heuristic_backend()


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

    cmp_p = sub.add_parser("compare", help="Compare two saved evaluation reports")
    cmp_p.add_argument("--before", type=str, required=True)
    cmp_p.add_argument("--after", type=str, required=True)
    cmp_p.add_argument("--out", type=str, default=str(DATA_DIR / "comparison.json"))

    iter_p = sub.add_parser("iterations", help="Compare RL learning-curve iterations")
    iter_p.add_argument("--metrics", type=str, default=str(LEARNING_CURVE_PATH))
    iter_p.add_argument("--out", type=str, default=str(DATA_DIR / "iteration_comparison.json"))

    # Default subcommand for convenience
    p.set_defaults(cmd="run", tasks=str(TASKS_PATH), tag="baseline", llm="heuristic",
                   model="gpt-4o-mini", max_steps=16, max_tasks=0)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.cmd == "compare":
        before = read_json(args.before)
        after = read_json(args.after)
        diff = compare(before, after)
        write_json(args.out, diff)
        print_comparison(diff)
        log.info("Comparison written → %s", args.out)
        return

    if args.cmd == "iterations":
        metrics = read_json(args.metrics)
        diff = compare_iterations(metrics)
        write_json(args.out, diff)
        print_iteration_comparison(metrics)
        log.info("Iteration comparison written → %s", args.out)
        return

    tasks = read_json(args.tasks)
    if getattr(args, "max_tasks", 0) > 0:
        tasks = tasks[: args.max_tasks]
    log.info("Evaluating on %d tasks (%s backend)", len(tasks), args.llm)

    llm = _build_llm(args)
    config = AgentConfig(max_steps=args.max_steps)
    report = evaluate(tasks, llm=llm, config=config)

    out_path = DATA_DIR / f"eval_{args.tag}.json"
    write_json(out_path, report)
    log.info("Saved report → %s", out_path)
    print_summary(report)


if __name__ == "__main__":
    main()
