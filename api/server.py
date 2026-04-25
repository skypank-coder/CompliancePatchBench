from pathlib import Path
from typing import Optional

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


def _read_json_if_exists(path: Path, default):
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


@app.get("/project")
def project_summary():
    learning_curve = _read_json_if_exists(PROJECT_DATA / "learning_curve.json", [])
    latest = learning_curve[-1] if learning_curve else None
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
        "endpoints": {
            "health": "/health",
            "tasks": "/tasks",
            "benchmark": "/benchmark",
            "rl_learning_curve": "/rl/learning-curve",
            "patch_reset": "/patch/reset",
            "patch_step": "/patch/step",
        },
    }


@app.get("/rl/learning-curve")
def rl_learning_curve():
    curve = _read_json_if_exists(PROJECT_DATA / "learning_curve.json", [])
    policy = _read_json_if_exists(PROJECT_DATA / "tabular_rl_policy.json", {})
    return {
        "learning_curve": curve,
        "tabular_policy_buckets": len((policy or {}).get("q", {})) if isinstance(policy, dict) else 0,
        "note": (
            "Curves show reward, success, and hidden-violation rate by RL iteration. "
            "The loop is designed to scale; demo runs may use a small subset for runtime."
        ),
    }


@app.get("/tasks")
def get_tasks():
    return TASK_METADATA


@app.get("/benchmark")
def get_benchmark():
    return {
        "environment": "compliancepatchbench",
        "non_cheatable": True,
        "hidden_constraints": True,
        "self_learning": ["SFT", "tabular RL", "LoRA policy-gradient RL"],
        "tasks": [
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
        ],
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
