"""
Shared utilities for the CompliancePatchBench project package.

Provides:
    - structured logging (so every module logs the same way)
    - robust JSON extraction (handles models that wrap JSON in prose / code fences)
    - JSONL read/write helpers
    - deterministic seeding
    - file IO helpers used by task_generator + dataset_builder
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional


# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

TASKS_PATH = DATA_DIR / "tasks.json"
DATASET_PATH = DATA_DIR / "dataset.jsonl"
TRAJECTORIES_PATH = DATA_DIR / "trajectories.jsonl"
TRAJECTORIES_RL_PATH = DATA_DIR / "trajectories_rl.jsonl"
LEARNING_CURVE_PATH = DATA_DIR / "learning_curve.json"


# ── Logging ───────────────────────────────────────────────────────────────────

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s"


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Return a configured logger that writes to stdout exactly once."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt="%H:%M:%S"))
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False
    return logger


# ── Seeding ───────────────────────────────────────────────────────────────────

def seed_everything(seed: int) -> None:
    """Make task generation + dataset building reproducible."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np  # optional dep
        np.random.seed(seed)
    except ImportError:
        pass


# ── JSON parsing ──────────────────────────────────────────────────────────────

# Match a balanced-looking { ... } block — first attempt
_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}")
# Strip ```json fences
_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)


def clip_model_json_output(text: str) -> str:
    """
    Keep a single JSON object span: from first '{' to last '}', then trim at last '}'.
    Use before json.loads / extract_json so multi-line "new_code" is not cut mid-brace
    (avoid STOP_TOKENS on '}' that break valid JSON).
    """
    if not text:
        return ""
    t = str(text)
    i = t.find("{")
    if i < 0:
        return t.strip()
    j = t.rfind("}")
    if j < i:
        return t[i:].strip()
    out = t[i : j + 1]
    if "}" in out:
        out = out[: out.rfind("}") + 1]
    return out.strip()


def clip_reward_value(value: float) -> float:
    """Bound RL step / GRPO rewards to [-1.0, 1.5]."""
    return max(min(float(value), 1.5), -1.0)


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Robustly extract the first JSON object from a possibly-noisy string.

    Strategy (each fallback is tried in order):
      1. parse the raw string as JSON
      2. strip a ```json ... ``` code fence if present
      3. find the first {...} block via regex and parse it
      4. find the *largest balanced* {...} substring and parse it

    Returns None if no parse succeeds — callers must handle this.
    """
    if not text:
        return None
    text = clip_model_json_output(text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fence_match = _FENCE_RE.search(text)
    if fence_match:
        candidate = fence_match.group(1).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    block_match = _JSON_BLOCK_RE.search(text)
    if block_match:
        candidate = block_match.group(0)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    return _largest_balanced_json(text)


def _largest_balanced_json(text: str) -> Optional[Dict[str, Any]]:
    """Find the longest balanced {...} substring and try to parse it."""
    best: Optional[Dict[str, Any]] = None
    stack: List[int] = []
    in_string = False
    string_quote = ""
    escape = False

    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if in_string:
            if ch == string_quote:
                in_string = False
            continue
        if ch in ('"', "'"):
            in_string = True
            string_quote = ch
            continue
        if ch == "{":
            stack.append(i)
        elif ch == "}" and stack:
            start = stack.pop()
            if not stack:
                candidate = text[start : i + 1]
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict):
                        best = parsed  # keep the latest (largest) one
                except json.JSONDecodeError:
                    continue
    return best


# ── JSONL helpers ─────────────────────────────────────────────────────────────

def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    """Write rows to a JSONL file. Returns number of rows written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    """Append a single row to a JSONL file (creates parent dirs)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    """Iterate over rows in a JSONL file. Skips blank lines + bad rows."""
    path = Path(path)
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


# ── JSON file helpers ─────────────────────────────────────────────────────────

def write_json(path: Path, obj: Any, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)


def read_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


# ── Misc ──────────────────────────────────────────────────────────────────────

def short_hash(text: str, n: int = 8) -> str:
    """Stable short hash, useful for generating deterministic task IDs."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


def chunk(seq: List[Any], size: int) -> Iterator[List[Any]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]
