"""
One-command competition smoke test.

Runs the minimum proof that the project is alive:
  - FastAPI imports and exposes deployment endpoints
  - task generation works
  - RL loop writes a learning curve
  - evaluation can compare iterations
  - RL transitions contain state/action/reward/next_state/logprob/done
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

from api.server import app  # noqa: E402

from .evaluate import compare_iterations  # noqa: E402
from .rl_trainer import RLConfig, train_rl  # noqa: E402
from .task_generator import generate_tasks  # noqa: E402
from .utils import DATA_DIR, TASKS_PATH, write_json  # noqa: E402


def main() -> None:
    routes = {r.path for r in app.routes}
    for required in {"/health", "/project", "/rl/learning-curve", "/tasks", "/benchmark"}:
        assert required in routes, f"Missing API route: {required}"

    original_tasks = TASKS_PATH.read_text(encoding="utf-8") if TASKS_PATH.exists() else None
    try:
        tasks = generate_tasks(num=12, seed=42)
        write_json(TASKS_PATH, tasks)

        result = train_rl(RLConfig(iterations=1, max_tasks=8, max_steps=8, dry_run=True))
    finally:
        if original_tasks is not None:
            TASKS_PATH.write_text(original_tasks, encoding="utf-8")

    curve = result["learning_curve"]
    assert len(curve) == 2, "Expected iteration 0 and 1 in learning curve"
    for row in curve:
        for key in ("iteration", "avg_reward", "success_rate", "hidden_violation_rate"):
            assert key in row, f"Missing learning-curve metric: {key}"

    diff = compare_iterations(curve)
    assert diff["iterations"] == len(curve)

    traj_path = DATA_DIR / "trajectories_rl.jsonl"
    rows = [json.loads(line) for line in traj_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows, "No RL trajectories written"
    transition = rows[0]["trajectory"][0]
    for key in ("state", "action", "reward", "next_state", "logprob", "done"):
        assert key in transition, f"Missing RL transition key: {key}"

    print("competition smoke test passed")
    # `train_rl` already printed Success / Hidden lines above.


if __name__ == "__main__":
    main()
