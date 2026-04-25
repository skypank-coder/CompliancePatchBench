"""
Generate PNG learning-curve figures for README / hackathon.
Reads project/data/learning_curve.json (iterative RL) — does not retrain.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _smooth(y: List[float], window: int) -> List[float]:
    if not y or window < 2:
        return y
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


def _save_metric(
    it: List[float],
    y: List[float],
    ylabel: str,
    out_path: Path,
    window: int,
) -> None:
    ys = _smooth(y, window)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(it, y, "o--", alpha=0.4, label="raw")
    ax.plot(it, ys, "-", linewidth=2, label=f"smoothed (window={window})")
    ax.set_xlabel("RL Iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs RL Iteration")
    ax.legend()
    ax.grid(True, alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_from_learning_curve(
    rows: List[Dict[str, Any]],
    out_dir: Path,
    window: int = 4,
) -> Tuple[Path, Path, Path]:
    if not rows:
        raise ValueError("rows must be non-empty")
    it = _iter_axis(rows)
    out_dir = Path(out_dir)

    re_path = out_dir / "reward_curve.png"
    y_r = [float(r.get("avg_reward", 0.0) or 0.0) for r in rows]
    _save_metric(it, y_r, "avg_reward", re_path, window)

    y_s = [float(r.get("success_rate", 0.0) or 0.0) for r in rows]
    su_path = out_dir / "success_curve.png"
    _save_metric(it, y_s, "success_rate", su_path, window)

    y_h = [float(r.get("hidden_violation_rate", 0.0) or 0.0) for r in rows]
    hi_path = out_dir / "hidden_violation_curve.png"
    _save_metric(it, y_h, "hidden_violation_rate", hi_path, window)

    return (re_path, su_path, hi_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot learning-curve figures for submission.")
    ap.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "learning_curve.json",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "figures",
    )
    ap.add_argument("--window", type=int, default=4)
    args = ap.parse_args()
    rows: List[Dict[str, Any]] = []
    if args.input.exists():
        raw = json.loads(args.input.read_text())
        if isinstance(raw, list):
            rows = raw
    if not rows:
        print("No data at", args.input, "— nothing to plot.")
        return
    a, b, c = plot_from_learning_curve(rows, args.out, window=args.window)
    for p in (a, b, c):
        if p:
            print("Wrote", p)


if __name__ == "__main__":
    main()
