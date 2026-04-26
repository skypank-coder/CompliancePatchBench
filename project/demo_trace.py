"""
Hackathon step-by-step demo: task selection and trace formatting only.
Does not change env, model, or reward math — only I/O and task picking.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

# Match notebook loop budget (keeps output readable)
DEMO_MAX_STEPS = 12
_EXPLAIN_MAX = 80


def _n_violations(task: Dict[str, Any]) -> int:
    v = task.get("violations")
    if isinstance(v, list):
        return len(v)
    gt = task.get("ground_truth")
    if isinstance(gt, list) and not v:
        return len(gt)
    return 0


def _n_codebase_files(task: Dict[str, Any]) -> int:
    return len(task.get("codebase") or {})


def select_demo_task(tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Prefer multi-file tasks; among those, pick the one with the most violations.
    If there are no multi-file tasks, pick the max-violations task in the list.
    """
    if not tasks:
        raise ValueError("select_demo_task: empty task list")
    multi = [t for t in tasks if _n_codebase_files(t) > 1]
    if multi:
        return max(multi, key=_n_violations)
    return max(tasks, key=_n_violations)


def print_demo_task_header(task: Dict[str, Any]) -> None:
    """Prints ## DEMO TASK block — context before a trace run."""
    tid = task.get("task_id", "—")
    n_files = _n_codebase_files(task)
    n_v = _n_violations(task)
    print("------------------------------------------------------------")
    print("## DEMO TASK")
    print("------------------------------------------------------------")
    print(f"  Task ID       : {tid}")
    print(f"  Files         : {n_files}")
    print(f"  Violations    : {n_v}")
    if n_files > 1:
        print("  This is a multi-file task requiring cross-file reasoning.")
    print("------------------------------------------------------------")


def interpret_demo_step_reward(r: float) -> str:
    """
    Text label for a single env step reward (per hackathon spec).
    """
    if r <= -0.999:
        return "invalid JSON / no valid action"
    if r < 0.0:
        return "setback (small env penalty / error)"
    if 0.0 <= r < 0.5:
        return "minor improvement / partial fix"
    if 0.5 <= r < 1.0:
        return "good fix (multiple violations resolved)"
    return "successful step / strong progress"


def _trim(s: str, n: int = _EXPLAIN_MAX) -> str:
    s = s.replace("\n", " ").strip()
    if len(s) <= n:
        return s
    return s[: n - 1] + "…"


def _demo_explain(action: Dict[str, Any], obs: Dict[str, Any], r: float) -> str:
    """Short natural-language line from action + env observation (no new facts)."""
    at = str(action.get("action_type", "?"))
    ar = str(obs.get("action_result", "") or "")

    if at == "read_file":
        path = action.get("path", "?")
        if "ERROR" in ar or ar.upper().startswith("ERROR"):
            return _trim(f"read_file error: {ar}")
        return _trim(f"Loaded file {path}")

    if at == "write_patch":
        fn = action.get("file", "?")
        if "REJECTED" in ar or "ERROR" in ar[:120]:
            return _trim(f"{fn}: {ar}")
        # reward hint from CI is often 0; rely on result text
        if "Patch applied" in ar:
            return _trim(f"Applied change in {fn} — {ar}")
        return _trim(f"write_patch {fn} — {ar}")

    if at == "run_ci":
        m = re.search(r"CI complete:\s*(\d+)\s*/\s*(\d+)", ar, re.I)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if a == b and b > 0:
                return "All CI checks passed (violations addressed in this run)"
            return _trim(f"CI result {a}/{b} on violation checks — {ar}")
        return _trim(f"CI: {ar}")

    if at == "finalize_patch":
        return _trim("finalize_patch (session end)")

    return _trim(ar)


def format_demo_step_line(
    step_1: int, action: Dict[str, Any], r: float, obs: Dict[str, Any]
) -> str:
    at = (action.get("action_type") or "?")[:15]
    expl = _demo_explain(action, obs, r)
    kind = interpret_demo_step_reward(r)
    return f"Step {step_1:2d} | {at:15s} | r={r:+.3f} | {expl} · {kind}"


def print_final_demo_block(info: Dict[str, Any]) -> None:
    """After finalize, print ## FINAL RESULT with score + CI list."""
    crit = info.get("critique") or {}
    final = float(
        info.get("final_score", crit.get("final_score", 0.0)) or 0.0
    )
    print("------------------------------------------------------------")
    print("## FINAL RESULT")
    print("------------------------------------------------------------")
    print(f"  Final score: {final:.4f}")
    rows: List[Dict[str, Any]] = crit.get("ci_results") or []
    if rows:
        print("  CI Results:")
        for c in rows:
            rid = str(c.get("rule_id", "rule"))
            ok = c.get("ci") == "PASS"
            mark = "✔" if ok else "✖"
            state = "passed" if ok else "failed"
            print(f"    {mark} {rid} {state}")
    else:
        print("  CI Results: (no per-rule rows in critique)")
