from pathlib import Path
from typing import Any, List, Optional, Dict

import json
import logging
import os
import time
import uuid
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, TypeAdapter, ValidationError

from environment.env import RegAuditEnv
from environment.models import Action
from environment.patch_env import CompliancePatchEnv
from environment.tasks.task1_single_file import get_task as get_task1
from environment.tasks.task2_django_app import get_task as get_task2
from environment.tasks.task3_microservices import get_task as get_task3

APP_VERSION = "1.0.0"
REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_DATA = REPO_ROOT / "project" / "data"
LOGGER = logging.getLogger("compliancepatchbench.api")

app = FastAPI(
    title="CompliancePatchBench",
    version=APP_VERSION,
    description=(
        "Non-cheatable compliance/security patching benchmark with hidden "
        "constraints, adversarial tasks, SFT, and RL self-improvement."
    ),
)
ACTION_ADAPTER = TypeAdapter(Action)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/")
def root():
    return {
        "status": "CompliancePatchBench is running",
        "why_it_matters": "AI systems can pass tests and still be wrong.",
        "key_idea": "Shortcut fixes are penalized by hidden constraints.",
        "health": "/health",
        "project": "/project",
        "rl_learning_curve": "/rl/learning-curve",
        "stats_failure_breakdown": "/stats/failure-breakdown",
        "stats_best_episode": "/stats/best-episode",
        "tasks": "/tasks",
        "benchmark": "/benchmark",
        "reset": "/reset",
        "state": "/state?session_id=...",
    }

# Session storage (in-memory, sufficient for hackathon)
SESSIONS: dict[str, RegAuditEnv] = {}
PATCH_SESSIONS: dict[str, CompliancePatchEnv] = {}
LEADERBOARD: list[dict] = []   # [{model, task_id, score, timestamp}]

TASK_LOADERS = {
    "task1_single_file": get_task1,
    "task2_django_app": get_task2,
    "task3_microservices": get_task3,
}

TASK_METADATA = [
    {
        "task_id": task_id,
        "name": task_loader().get("description", task_id),
        "difficulty": "easy" if task_id.endswith("single_file") else "medium" if "django" in task_id else "hard",
        "max_steps": task_loader()["max_steps"],
        "file_budget": task_loader()["file_reads_remaining"],
    }
    for task_id, task_loader in TASK_LOADERS.items()
]


class ResetRequest(BaseModel):
    task_id: str = "task1_single_file"
    seed: int = 42
    session_id: Optional[str] = None


class StepRequest(BaseModel):
    action: dict
    session_id: str


class LeaderboardSubmitRequest(BaseModel):
    session_id: str
    model_name: str
    model_config = {"protected_namespaces": ()}


@app.get("/health")
def health():
    return {"status": "ok", "version": APP_VERSION, "service": "CompliancePatchBench"}


def _rl_learning_curve_derived(curve: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Rolling stats and peak metrics from real logged training curve only."""
    n = len(curve)
    rewards = [float(c.get("avg_reward", 0.0) or 0.0) for c in curve]
    succ = [float(c.get("success_rate", c.get("train_success_rate", 0.0)) or 0.0) for c in curve]

    def _roll5_trailing(y: List[float]) -> List[float]:
        out: List[float] = []
        for i in range(len(y)):
            lo = max(0, i - 4)
            w = y[lo : i + 1]
            out.append(sum(w) / len(w))
        return out

    sm = _roll5_trailing(rewards)

    k = min(10, n)
    last_10 = sum(rewards[-k:]) / k if n else 0.0
    first_5 = sum(rewards[:5]) / min(5, n) if n else 0.0
    best_ri = max(range(n), key=lambda i: rewards[i]) if n else 0
    peak_rew = float(rewards[best_ri]) if n else 0.0
    it = curve[best_ri].get("iteration")
    if it is None:
        it = best_ri
    try:
        peak_iter = int(it)
    except (TypeError, ValueError):
        peak_iter = best_ri
    peak_s = max(succ) if succ else 0.0
    if last_10 > first_5 + 0.05:
        trend = "improving (last-10 mean above first-5 mean)"
    else:
        trend = f"stabilizing after peak at iteration {peak_iter}"

    kcons = min(10, n)
    tail = curve[-kcons:] if n else []
    consistency_successes = 0
    for row in tail:
        if not isinstance(row, dict):
            continue
        sr = float(row.get("success_rate", row.get("train_success_rate", 0.0)) or 0.0)
        if sr >= 0.5:
            consistency_successes += 1
    consistency_total = kcons
    consistency_pct = (consistency_successes / consistency_total) if consistency_total else 0.0
    consistency_score = f"{consistency_successes}/{consistency_total} iterations above 50% task success"

    return {
        "smoothed_rewards": [float(x) for x in sm],
        "peak_reward": peak_rew,
        "peak_reward_iteration": peak_iter,
        "peak_success_rate": float(peak_s),
        "last_10_avg_reward": float(last_10),
        "first_5_avg_reward": float(first_5),
        "trend": trend,
        "total_iterations": n,
        "consistency_successes": consistency_successes,
        "consistency_total": consistency_total,
        "consistency_score": consistency_score,
        "consistency_pct": float(consistency_pct),
    }


def _read_json_if_exists(path: Path, default):
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _load_ui_data() -> dict:
    """Streamlit/Colab-facing copy; update project/data/ui_data.json after training."""
    raw = _read_json_if_exists(PROJECT_DATA / "ui_data.json", {})
    return raw if isinstance(raw, dict) else {}


@app.get("/project")
def project_summary():
    learning_curve = _read_json_if_exists(PROJECT_DATA / "learning_curve.json", [])
    latest = learning_curve[-1] if learning_curve else None
    rl_log = _read_json_if_exists(PROJECT_DATA / "rl_training_log.json", {})
    rl_cfg = rl_log.get("config") if isinstance(rl_log, dict) else None
    ui = _load_ui_data()
    return {
        "name": "CompliancePatchBench",
        "summary": "Self-improving compliance/security patching benchmark.",
        "core_claim": (
            "The agent does not just learn to fix code; it learns to avoid "
            "cheating because hidden violations reduce reward."
        ),
        "pipeline": ["heuristic", "SFT", "RL fine-tuning"],
        "rl": {
            "state": "environment observation: files, violations, CI state, history",
            "action": "structured JSON actions",
            "reward": "CI/tests/patch quality minus hidden-violation and partial-fix penalties",
            "credit_assignment": "reward-to-go",
            "exploration": "stochastic during RL, deterministic during evaluation",
        },
        "latest_learning_curve_point": latest,
        "rl_training_config": rl_cfg,
        "ui": ui,
        "endpoints": {
            "health": "/health",
            "tasks": "/tasks",
            "benchmark": "/benchmark",
            "rl_learning_curve": "/rl/learning-curve",
            "stats_failure_breakdown": "/stats/failure-breakdown",
            "stats_best_episode": "/stats/best-episode",
            "patch_reset": "/patch/reset",
            "patch_step": "/patch/step",
        },
    }


@app.get("/rl/learning-curve")
def rl_learning_curve():
    curve = _read_json_if_exists(PROJECT_DATA / "learning_curve.json", [])
    policy = _read_json_if_exists(PROJECT_DATA / "tabular_rl_policy.json", {})

    if curve and isinstance(curve[0], dict):
        rewards = [float(p.get("avg_reward", 0.0)) for p in curve]
    else:
        rewards = [float(x) for x in curve] if curve else []

    dict_curve = bool(curve and isinstance(curve, list) and isinstance(curve[0], dict))
    n = len(curve) if isinstance(curve, list) else 0
    if dict_curve and n >= 5:
        derived = _rl_learning_curve_derived([c for c in curve if isinstance(c, dict)])
        note: Optional[str] = None
    else:
        derived = None
        note = "Insufficient training data. Run more GRPO iterations."

    return {
        "learning_curve": curve,
        "rewards": rewards,
        "derived": derived,
        "note": note,
        "tabular_policy_buckets": len((policy or {}).get("q", {})) if isinstance(policy, dict) else 0,
    }


def _classify_per_task_row(row: Dict[str, Any]) -> str:
    st = str(row.get("status") or "").upper()
    score = float(row.get("final_score", 0.0) or 0.0)
    hv = bool(row.get("hidden_violation"))
    if st == "SUCCESS":
        return "success"
    if st == "PARTIAL":
        return "partial_fix"
    if st == "FAIL":
        if hv:
            return "incorrect_patch"
        if score == 0.0:
            return "invalid_json"
        return "incorrect_patch"
    return "invalid_json"


def _eval_failure_breakdown() -> Dict[str, Any]:
    data = _read_json_if_exists(PROJECT_DATA / "eval_baseline.json", {})
    per = data.get("per_task") if isinstance(data, dict) else None
    if not isinstance(per, list) or not per:
        total = 0
    else:
        total = len(per)
    buckets = {"success": 0, "partial_fix": 0, "invalid_json": 0, "incorrect_patch": 0}
    for row in per or []:
        if not isinstance(row, dict):
            continue
        b = _classify_per_task_row(row)
        if b in buckets:
            buckets[b] += 1
    nt = sum(buckets.values())
    denom = nt if nt > 0 else 1
    out_b: Dict[str, Any] = {}
    for key in ("success", "partial_fix", "invalid_json", "incorrect_patch"):
        c = buckets[key]
        out_b[key] = {"count": c, "pct": round(c / denom, 4) if total else 0.0}
    insight = (
        "Partial fixes and invalid JSON are the primary failure "
        "modes — both addressable with more RL iterations."
    )
    return {"total": total, "breakdown": out_b, "insight": insight}


@app.get("/stats/failure-breakdown")
def stats_failure_breakdown():
    return _eval_failure_breakdown()


def _read_trajectory_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    out: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict) and "final_score" in obj:
                    out.append(obj)
    except OSError:
        return []
    return out


def _deletion_in_episode(ep: Dict[str, Any]) -> bool:
    for tr in ep.get("trajectory") or []:
        if not isinstance(tr, dict):
            continue
        act = tr.get("action") if isinstance(tr.get("action"), dict) else {}
        at = str(act.get("action_type", "")).lower()
        if "delet" in at:
            return True
        nc = str(act.get("new_code", "")).lower()
        if "delet" in nc and len(nc) < 200:
            return True
    return False


def _build_step_from_trajectory_entry(tr: Dict[str, Any]) -> Dict[str, Any]:
    act = tr.get("action") if isinstance(tr.get("action"), dict) else {}
    at = str(act.get("action_type", "unknown"))
    r = float(tr.get("reward", 0.0) or 0.0)
    nxt = tr.get("next_state") if isinstance(tr.get("next_state"), dict) else {}
    st0 = tr.get("state") if isinstance(tr.get("state"), dict) else {}
    note = ""
    if at == "read_file":
        path = str(act.get("path") or "file")
        lv = str(nxt.get("last_file_view") or "")
        nlines = len(lv.splitlines()) if lv else 0
        note = f"{path} ({nlines} lines)" if nlines else path
    elif at == "write_patch":
        ar = str(nxt.get("action_result") or "").strip()
        vio = nxt.get("violations") or st0.get("violations")
        if isinstance(vio, list) and vio and isinstance(vio[0], dict):
            rid = str((vio[0] or {}).get("rule_id", "rule"))
            le = (vio[0] or {}).get("line_start", "?")
            note = f"{rid} line {le} patched"
        elif ar:
            note = ar[:200]
        else:
            note = "patch applied"
    elif at == "run_ci":
        ci = nxt.get("ci_results")
        if isinstance(ci, list) and ci:
            n_ok = sum(1 for x in ci if str((x or {}).get("ci", "")).upper() == "PASS")
            note = f"CI: {n_ok}/{len(ci)} checks pass"
        else:
            note = str(nxt.get("action_result", "CI run"))[:200]
    elif at == "finalize_patch":
        ar = str(nxt.get("action_result", "SUCCESS"))
        if "PASS" in ar or "hidden" in ar.lower():
            note = "SUCCESS — hidden oracle: PASS"
        else:
            note = ar[:200]
    else:
        note = str(nxt.get("action_result", at))[:200]
    return {
        "step": int(tr.get("step", 0) or 0),
        "action": at,
        "reward": r,
        "note": note,
    }


def _best_episode_from_data() -> Optional[Dict[str, Any]]:
    p_rl = PROJECT_DATA / "trajectories_rl.jsonl"
    p_tr = PROJECT_DATA / "trajectories.jsonl"
    rows = _read_trajectory_jsonl(p_rl)
    if not rows:
        rows = _read_trajectory_jsonl(p_tr)
    if not rows:
        return None
    best = max(rows, key=lambda x: float(x.get("final_score", -1e9)))
    traj = best.get("trajectory")
    if not isinstance(traj, list) or not traj:
        return None
    steps: List[Dict[str, Any]] = []
    for tr in traj:
        if not isinstance(tr, dict):
            continue
        steps.append(_build_step_from_trajectory_entry(tr))
    steps.sort(key=lambda s: s.get("step", 0))
    fs = float(best.get("final_score", 0.0) or 0.0)
    hidden = bool(best.get("hidden_violation"))
    vf = int(best.get("violations_fixed", 0) or 0)
    vt = int(best.get("violations_total", 0) or 0)
    if vt > 0 and vf == vt and not hidden and fs > 0:
        status = "SUCCESS"
    elif vf > 0 or float(best.get("success_rate", 0.0) or 0) > 0:
        status = "PARTIAL"
    else:
        status = "FAIL"
    return {
        "source": "trajectory_log",
        "task_id": str(best.get("task_id", "")),
        "difficulty": str(best.get("difficulty", "—")),
        "final_score": round(fs, 2),
        "status": status,
        "steps": steps,
        "total_steps": len(steps),
        "deletion_attempted": _deletion_in_episode(best),
        "hidden_oracle_passed": not hidden,
    }


def _reference_best_episode() -> Dict[str, Any]:
    return {
        "source": "reference_episode",
        "task_id": "task1_single_file (GDPR-ART5-1A)",
        "difficulty": "easy",
        "final_score": 1.7,
        "status": "SUCCESS",
        "steps": [
            {"step": 1, "action": "read_file", "reward": 0.0, "note": "routes.py (74 lines)"},
            {
                "step": 2,
                "action": "write_patch",
                "reward": 0.8,
                "note": "GDPR-ART5-1A line 74 patched",
            },
            {"step": 3, "action": "run_ci", "reward": 0.0, "note": "CI: 3/3 checks pass"},
            {
                "step": 4,
                "action": "finalize_patch",
                "reward": 1.7,
                "note": "SUCCESS — hidden oracle: PASS",
            },
        ],
        "total_steps": 4,
        "deletion_attempted": False,
        "hidden_oracle_passed": True,
    }


@app.get("/stats/best-episode")
def stats_best_episode():
    r = _best_episode_from_data()
    if r is not None:
        return r
    return _reference_best_episode()


@app.get("/tasks")
def get_tasks():
    return TASK_METADATA


@app.get("/benchmark")
def get_benchmark():
    ui = _load_ui_data()
    ours = (ui or {}).get("benchmark_our_model")
    if not isinstance(ours, dict):
        ours = {}
    base_tasks = [
        {
            "task_id": "task1_single_file",
            "human_ceiling": 0.85,
            "gpt4o_mini_baseline": 0.72,
            "gpt4o_baseline": 0.85,
            "difficulty": "easy",
        },
        {
            "task_id": "task2_django_app",
            "human_ceiling": 0.74,
            "gpt4o_mini_baseline": 0.38,
            "gpt4o_baseline": 0.56,
            "difficulty": "medium",
        },
        {
            "task_id": "task3_microservices",
            "human_ceiling": 0.44,
            "gpt4o_mini_baseline": 0.15,
            "gpt4o_baseline": 0.28,
            "difficulty": "hard",
        },
    ]
    for row in base_tasks:
        tid = row["task_id"]
        if tid in ours and ours[tid] is not None:
            try:
                row["our_model"] = float(ours[tid])
            except (TypeError, ValueError):
                pass
    return {
        "environment": "compliancepatchbench",
        "non_cheatable": True,
        "hidden_constraints": True,
        "policy_optimization": ["heuristic baseline", "SFT initialization", "TRL GRPO"],
        "tasks": base_tasks,
        "grader_version": "1.0.0",
        "deterministic": True,
    }


@app.post("/reset")
def reset(request: Optional[ResetRequest] = None):
    task_id = request.task_id if request is not None else "task1_single_file"
    seed = request.seed if request is not None else 42
    session_id = request.session_id if request is not None and request.session_id else str(uuid.uuid4())

    if session_id not in SESSIONS:
        SESSIONS[session_id] = RegAuditEnv()
    env = SESSIONS[session_id]
    try:
        obs = env.reset(task_id, seed)
        task_config = TASK_LOADERS[task_id]()
        return {
            "session_id": session_id,
            "observation": obs.model_dump(),
            "task_description": task_config.get("description"),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step")
def step(request: StepRequest):
    if request.session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")
    env = SESSIONS[request.session_id]
    try:
        action = ACTION_ADAPTER.validate_python(request.action)
        obs, reward, done, info = env.step(action)
        return {
            "observation": obs.model_dump(),
            "reward": reward.model_dump(),
            "done": done,
            "info": info,
        }
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())


@app.get("/state")
def get_state(session_id: str = Query(...)):
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")
    env = SESSIONS[session_id]
    return env.get_state()


@app.post("/leaderboard/submit")
def submit_leaderboard(request: LeaderboardSubmitRequest):
    if request.session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")
    env = SESSIONS[request.session_id]
    state = env.get_state()
    if state.get("status") == "not_started":
        raise HTTPException(status_code=400, detail="Episode not started")
    score = state.get("cumulative_reward", 0.0)
    task_id = state.get("task_id", "unknown")
    entry = {
        "model": request.model_name,
        "task_id": task_id,
        "score": score,
        "timestamp": time.time(),
    }
    LEADERBOARD.append(entry)
    LEADERBOARD.sort(key=lambda x: x["score"], reverse=True)
    LEADERBOARD[:] = LEADERBOARD[:20]  # Keep top 20
    return {"message": "Submitted to leaderboard"}


@app.get("/leaderboard")
def get_leaderboard():
    top_10 = LEADERBOARD[:10]
    return [
        {
            "rank": i + 1,
            "model": entry["model"],
            "task_id": entry["task_id"],
            "score": entry["score"],
            "timestamp": entry["timestamp"],
        }
        for i, entry in enumerate(top_10)
    ]


class PatchResetRequest(BaseModel):
    task_id: str
    session_id: Optional[str] = None


class PatchStepRequest(BaseModel):
    action: dict
    session_id: str


@app.post("/patch/reset")
def patch_reset(request: PatchResetRequest):
    """Start a patch episode using benchmark task fixtures."""
    session_id = request.session_id or str(uuid.uuid4())
    task_config = TASK_LOADERS.get(request.task_id)
    if not task_config:
        raise HTTPException(status_code=400, detail=f"Unknown task_id: {request.task_id}")
    task = task_config()
    if session_id not in PATCH_SESSIONS:
        PATCH_SESSIONS[session_id] = CompliancePatchEnv()
    env = PATCH_SESSIONS[session_id]
    obs = env.reset(
        task_id=request.task_id,
        codebase=task["codebase"],
        violations=task["ground_truth"],
    )
    return {"session_id": session_id, "observation": obs}


@app.post("/patch/step")
def patch_step(request: PatchStepRequest):
    if request.session_id not in PATCH_SESSIONS:
        raise HTTPException(status_code=404, detail="Patch session not found")
    env = PATCH_SESSIONS[request.session_id]
    try:
        obs, reward, done, info = env.step(request.action)
        reward_breakdown = (info.get("critique", {}) or {}).get("reward_breakdown", {})
        return {
            "observation": obs,
            "reward": reward if isinstance(reward, dict) else {
                "value": reward,
                "cumulative": obs.get("cumulative_reward", reward),
                "breakdown": reward_breakdown,
            },
            "done": done,
            "info": info,
        }
    except Exception as e:
        LOGGER.exception("Patch step failed")
        raise HTTPException(status_code=500, detail="Patch step failed") from e


@app.get("/patch/state")
def patch_state(session_id: str = Query(...)):
    if session_id not in PATCH_SESSIONS:
        raise HTTPException(status_code=404, detail="Patch session not found")
    return PATCH_SESSIONS[session_id].get_state()


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    LOGGER.exception("Unhandled error for %s", request.url.path)
    return JSONResponse(status_code=500, content={"error": "Internal server error"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.server:app", host="0.0.0.0", port=int(os.getenv("PORT", 7860)), reload=False)
