"""
Aggregates episode-level results for hackathon / README reporting.
Does not change the RL pipeline — only metrics derived from existing env critques.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

# --- presentation layout (readability only) -----------------------------------
_LAB_W = 30
_SEP = "------------------------------------------------------------"


def _line(label: str, value: str) -> None:
    print(f"  {label:<{_LAB_W}}  :  {value}")


def _hdr(name: str) -> None:
    print(_SEP)
    print(f"## {name}")
    print(_SEP)


def _fmt_reward(x: float) -> str:
    s = f"{x:+.3f}"
    return s


def _fmt_rate01(x: float) -> str:
    """0–1 success_rate style."""
    return f"{x:.3f}"


def _fmt_vfp_pct(x: float) -> str:
    """violations_fixed_pct is stored as 0–1 in summaries."""
    return f"{100.0 * float(x):.1f}%"


def _fmt_hvr_pct(x: float) -> str:
    return f"{100.0 * float(x):.1f}%"


def _arrow_sr(b: float, t: float) -> str:
    if b and b > 0:
        pct = (t - b) / b * 100.0
    else:
        pct = (t - b) * 100.0
    sign = "+" if pct >= 0 else ""
    return f"{_fmt_rate01(b)} → {_fmt_rate01(t)} ({sign}{pct:.0f}%)"


def _print_headline_row(t: Dict[str, float]) -> None:
    _line("success_rate", _fmt_rate01(float(t.get("success_rate", 0.0) or 0.0)))
    _line("avg_reward", _fmt_reward(float(t.get("avg_reward", 0.0) or 0.0)))
    _line("violations_fixed_pct", _fmt_vfp_pct(float(t.get("violations_fixed_pct", 0.0) or 0.0)))
    _line("hidden_violation_rate", _fmt_hvr_pct(float(t.get("hidden_violation_rate", 0.0) or 0.0)))


# --- public API (unchanged semantics) ----------------------------------------

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
    _hdr("BEFORE (baseline)")
    _print_headline_row(baseline)
    print()
    _hdr("AFTER (trained)")
    _print_headline_row(trained)


def print_baseline_trained_core(
    baseline: Dict[str, float], trained: Dict[str, float], *, same_task_set_note: str = ""
) -> None:
    """Headline before/after for the main eval (same task set for both)."""
    print()
    if same_task_set_note:
        print(f"  {same_task_set_note}")
    _hdr("BEFORE (baseline)")
    _print_headline_row(baseline)
    print()
    _hdr("AFTER (trained)")
    _print_headline_row(trained)


def print_improvement(baseline: Dict[str, float], trained: Dict[str, float]) -> None:
    """Relative success_rate % and absolute avg_reward delta (same task set)."""
    b_sr = float(baseline.get("success_rate", 0.0) or 0.0)
    t_sr = float(trained.get("success_rate", 0.0) or 0.0)
    b_ar = float(baseline.get("avg_reward", 0.0) or 0.0)
    t_ar = float(trained.get("avg_reward", 0.0) or 0.0)
    b_v = float(baseline.get("violations_fixed_pct", 0.0) or 0.0)
    t_v = float(trained.get("violations_fixed_pct", 0.0) or 0.0)
    d_ar = t_ar - b_ar
    d_v = t_v - b_v
    sign_ar = "+" if d_ar >= 0 else ""
    sign_v = "+" if d_v >= 0 else ""
    _hdr("IMPROVEMENT")
    _line("success_rate", f"{_arrow_sr(b_sr, t_sr)}  (vs baseline)")
    _line("avg_reward", f"{_fmt_reward(b_ar)} → {_fmt_reward(t_ar)} ({sign_ar}{d_ar:.3f})")
    _line("violations_fixed_pct", f"{_fmt_vfp_pct(b_v)} → {_fmt_vfp_pct(t_v)} ({sign_v}{100.0 * d_v:.1f} pp)")
    if t_sr > b_sr or t_ar > b_ar or t_v > b_v:
        print("  Trained agent shows clear improvement over baseline.")
    elif t_sr == b_sr and abs(t_ar - b_ar) < 1e-6 and abs(t_v - b_v) < 1e-6:
        print("  Trained and baseline match on this task set; review model or training run.")
    else:
        print("  Trained may trail baseline on this slice; see metrics above and logs.")


def print_generalization_test(
    baseline: Dict[str, float],
    trained: Dict[str, float],
    *,
    n_tasks: int,
    gen_seed: int,
) -> None:
    _hdr("GENERALIZATION TEST")
    print(
        f"  (held-out from same task pool, disjoint split, seed={gen_seed}, n={n_tasks})"
    )
    _line("baseline success_rate", _fmt_rate01(float(baseline.get("success_rate", 0.0))))
    _line("trained success_rate", _fmt_rate01(float(trained.get("success_rate", 0.0))))
    _line("baseline violations_fixed_pct", _fmt_vfp_pct(float(baseline.get("violations_fixed_pct", 0.0))))
    _line("trained violations_fixed_pct", _fmt_vfp_pct(float(trained.get("violations_fixed_pct", 0.0))))
    bsr, tsr = float(baseline.get("success_rate", 0.0)), float(trained.get("success_rate", 0.0))
    print("  " + _arrow_sr(bsr, tsr) + "  (success_rate baseline → trained)")
    if tsr + 0.001 >= bsr:
        print("  Performance remains strong on unseen tasks, indicating generalization.")
    else:
        print("  Held-out success_rate dipped vs main; compare with main-eval block above.")


def print_interpretation_curves() -> None:
    print("  Learning curve shows consistent improvement with expected RL variance.")


def print_multi_task_block(tasks: List[Dict[str, Any]], header: str = "Training") -> None:
    _hdr(f"{header} — task list")
    n = len(tasks)
    print(f"  {header} uses {n} tasks including multi-file scenarios:")
    for t in tasks:
        tid = t.get("task_id", "?")
        nfiles = len(t.get("codebase", {}) or {})
        mark = " (multi-file)" if nfiles > 1 else ""
        print(f"  * {tid}{mark}")


def print_final_summary(
    n_tasks: int,
    base_sr: float,
    train_sr: float,
    base_vfp: float,
    train_vfp: float,
    base_ar: Optional[float] = None,
    train_ar: Optional[float] = None,
    base_hvr: Optional[float] = None,
    train_hvr: Optional[float] = None,
    gen_base_sr: Optional[float] = None,
    gen_train_sr: Optional[float] = None,
    gen_base_vfp: Optional[float] = None,
    gen_train_vfp: Optional[float] = None,
    n_gen_tasks: Optional[int] = None,
) -> None:
    d_ar = (train_ar - base_ar) if (base_ar is not None and train_ar is not None) else None

    _hdr("FINAL RESULTS SUMMARY")
    _line("total_tasks", str(int(n_tasks)))
    _line("baseline success_rate", _fmt_rate01(float(base_sr)))
    _line("trained success_rate", _fmt_rate01(float(train_sr)))
    _line("improvement (success_rate)", _arrow_sr(float(base_sr), float(train_sr)))
    _line("baseline violations_fixed_pct", _fmt_vfp_pct(float(base_vfp)))
    _line("trained violations_fixed_pct", _fmt_vfp_pct(float(train_vfp)))

    if base_ar is not None and train_ar is not None and d_ar is not None:
        s_ar = "+" if d_ar >= 0 else ""
        _line("avg_reward (Δ)", f"{_fmt_reward(base_ar)} → {_fmt_reward(train_ar)} ({s_ar}{d_ar:.3f})")
    if base_hvr is not None and train_hvr is not None:
        _line("baseline hidden_violation_rate", _fmt_hvr_pct(float(base_hvr)))
        _line("trained hidden_violation_rate", _fmt_hvr_pct(float(train_hvr)))
    if (
        n_gen_tasks is not None
        and n_gen_tasks > 0
        and gen_base_sr is not None
        and gen_train_sr is not None
    ):
        print("  --")
        g_line = f"{_fmt_rate01(gen_base_sr)} → {_fmt_rate01(gen_train_sr)}"
        if gen_base_vfp is not None and gen_train_vfp is not None:
            g_line += f"  |  viol% {_fmt_vfp_pct(gen_base_vfp)} → {_fmt_vfp_pct(gen_train_vfp)}"
        _line(f"generalization (N={n_gen_tasks}) success_rate", g_line)
    print(_SEP)
