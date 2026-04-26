"""
Generate exactly three PNG learning-curve figures (judge view).

Reads project/data/learning_curve.json — does not retrain.
Does not write loss/KL or extra subplots.

If a target PNG already exists, it is not recreated (idempotent for CI and local runs).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

_REPO_DATA = Path(__file__).resolve().parent / "data"
RL_TRAINING_LOG = _REPO_DATA / "rl_training_log.json"
LEARNING_CURVE_PATH = _REPO_DATA / "learning_curve.json"
FIGURES_DIR = _REPO_DATA / "figures"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Readability: judge-friendly text sizes
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }
)


def _smooth(y: List[float], window: int) -> List[float]:
    if not y or window < 2:
        return list(y)
    w = min(window, len(y))
    out: List[float] = []
    for i in range(len(y)):
        lo = max(0, i - w // 2)
        hi = min(len(y), lo + w)
        lo = max(0, hi - w)
        chunk = y[lo:hi]
        out.append(sum(chunk) / len(chunk))
    return out


def _iter_axis(rows: List[Dict[str, Any]]) -> List[float]:
    return [float(r.get("iteration", i)) for i, r in enumerate(rows)]


def _choose_window(n: int, explicit: Optional[int]) -> int:
    if explicit is not None and explicit >= 2:
        return min(explicit, max(2, n))
    return 7 if n >= 14 else 5


def _write_smoothed_curve(
    it: List[float],
    y_raw: List[float],
    *,
    title: str,
    ylabel: str,
    out_path: Path,
    window: int,
    skip_if_exists: bool,
) -> Tuple[Path, bool]:
    """
    Returns (path, created_this_run) where created_this_run is False if skipped or unchanged on disk.
    """
    out_path = Path(out_path)
    if skip_if_exists and out_path.is_file():
        return out_path, False

    ys = _smooth(y_raw, window)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(it, y_raw, "o--", color="0.6", alpha=0.5, markersize=4, label="raw", linewidth=1)
    ax.plot(
        it,
        ys,
        "-",
        color="C0",
        linewidth=2.5,
        label=f"smoothed (window={window})",
    )
    ax.set_xlabel("RL Iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(bottom=None)
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle=":")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path, True


def plot_from_learning_curve(
    rows: List[Dict[str, Any]],
    out_dir: Path,
    window: Optional[int] = None,
    *,
    skip_existing: bool = True,
) -> Tuple[Path, Path, Path]:
    """
    Produce (reward_curve.png, success_curve.png, json_validity_curve.png).
    Reuses existing files when skip_existing=True and the file is present.
    """
    if not rows:
        raise ValueError("rows must be non-empty")
    w = _choose_window(len(rows), window)
    it = _iter_axis(rows)
    out_dir = Path(out_dir)

    y_r = [float(r.get("avg_reward", 0.0) or 0.0) for r in rows]
    y_s = [float(r.get("success_rate", r.get("train_success_rate", 0.0)) or 0.0) for r in rows]
    y_j = [float(r.get("valid_json_rate", 0.0) or 0.0) for r in rows]

    re_path, c1 = _write_smoothed_curve(
        it,
        y_r,
        title="Reward vs RL Iteration",
        ylabel="avg_reward",
        out_path=out_dir / "reward_curve.png",
        window=w,
        skip_if_exists=skip_existing,
    )
    su_path, c2 = _write_smoothed_curve(
        it,
        y_s,
        title="Success Rate vs RL Iteration",
        ylabel="success_rate",
        out_path=out_dir / "success_curve.png",
        window=w,
        skip_if_exists=skip_existing,
    )
    js_path, c3 = _write_smoothed_curve(
        it,
        y_j,
        title="JSON Validity Rate vs RL Iteration",
        ylabel="JSON validity rate",
        out_path=out_dir / "json_validity_curve.png",
        window=w,
        skip_if_exists=skip_existing,
    )
    for p, created in ((re_path, c1), (su_path, c2), (js_path, c3)):
        if created:
            log.info("Wrote %s", p)
        else:
            log.info("Unchanged (already exists, skipped): %s", p)
    return (re_path, su_path, js_path)


def _extract_valid_curve_entries(raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep only points with all required training metrics (no synthetic fill)."""
    out: List[Dict[str, Any]] = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        if "iteration" not in row or "avg_reward" not in row or "success_rate" not in row or "valid_json_rate" not in row:
            continue
        out.append(row)
    return out


def rebuild_learning_curve_from_rl_log(
    rl_log_path: Path = RL_TRAINING_LOG,
    out_path: Path = LEARNING_CURVE_PATH,
) -> tuple[List[Dict[str, Any]], int]:
    """
    Load learning_curve from rl_training_log.json; require >= 10 valid entries.
    Returns (entries, n) or raises ValueError.
    """
    p = Path(rl_log_path)
    if not p.is_file():
        raise ValueError("missing_rl_log")
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("invalid_rl_log")
    lc = data.get("learning_curve")
    if not isinstance(lc, list) or len(lc) < 10:
        raise ValueError("insufficient_raw")
    valid = _extract_valid_curve_entries(lc)
    if len(valid) < 10:
        raise ValueError("insufficient_valid")
    return valid, len(valid)


def _write_learning_curve_figures(
    curve_path: Path = LEARNING_CURVE_PATH,
    out_dir: Path = FIGURES_DIR,
    window: Optional[int] = None,
) -> None:
    """Regenerate the three judge PNGs from on-disk learning_curve.json (overwrites)."""
    curve_path = Path(curve_path)
    if not curve_path.is_file():
        log.warning("No curve at %s — skipping figure regeneration", curve_path)
        return
    rows: List[Dict[str, Any]] = json.loads(curve_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        log.warning("Empty curve file — skipping figures")
        return
    w = None if (window is None or int(window) <= 0) else int(window)
    plot_from_learning_curve(rows, Path(out_dir), window=w, skip_existing=False)
    print_graph_summary(Path(out_dir))


# Backward-compatible name
regenerate_figures_from_curve_file = _write_learning_curve_figures


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot the three learning-curve PNGs (reward, success, JSON validity).")
    ap.add_argument(
        "--rebuild-curve",
        action="store_true",
        help="Rebuild project/data/learning_curve.json from rl_training_log.json (>=10 valid points), then replot.",
    )
    ap.add_argument(
        "--rl-log",
        type=Path,
        default=RL_TRAINING_LOG,
        help="Path to rl_training_log.json (for --rebuild-curve).",
    )
    ap.add_argument(
        "--curve-out",
        type=Path,
        default=LEARNING_CURVE_PATH,
        help="Output path for rebuilt learning_curve.json.",
    )
    ap.add_argument(
        "--input",
        type=Path,
        default=LEARNING_CURVE_PATH,
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=FIGURES_DIR,
    )
    ap.add_argument(
        "--window",
        type=int,
        default=0,
        help="Smoothing window (0 = auto: 5 or 7 by series length).",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Regenerate all three PNGs even if they already exist.",
    )
    args = ap.parse_args()

    if bool(args.rebuild_curve):
        try:
            valid, n = rebuild_learning_curve_from_rl_log(rl_log_path=Path(args.rl_log), out_path=Path(args.curve_out))
        except ValueError:
            print(
                "WARNING: rl_training_log.json has insufficient data.\n"
                "   Run GRPO training first and export the log. NOT generating synthetic data."
            )
            sys.exit(1)
        out_curve = Path(args.curve_out)
        out_curve.parent.mkdir(parents=True, exist_ok=True)
        out_curve.write_text(json.dumps(valid, indent=2) + "\n", encoding="utf-8")
        print(f"Rebuilt learning_curve.json from rl_training_log.json: {n} iterations")
        _write_learning_curve_figures(curve_path=out_curve, out_dir=Path(args.out), window=0)
        return

    rows: List[Dict[str, Any]] = []
    if args.input.exists():
        raw = json.loads(args.input.read_text())
        if isinstance(raw, list):
            rows = raw
    if not rows:
        print("No data at", args.input, "— nothing to plot.")
        print_graph_summary(args.out)
        return
    win = None if int(args.window) <= 0 else int(args.window)
    plot_from_learning_curve(rows, args.out, window=win, skip_existing=not bool(args.force))
    print_graph_summary(args.out)


if __name__ == "__main__":
    main()
