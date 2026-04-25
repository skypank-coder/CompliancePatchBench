"""
Hugging Face Docker Streamlit Spaces run: streamlit run streamlit_app.py
Real UI lives in app.py (single source of truth).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_app = Path(__file__).resolve().parent / "app.py"
spec = importlib.util.spec_from_file_location("_compliancepatchbench_ui", _app)
if not spec or not spec.loader:
    raise RuntimeError(f"Could not load {_app}")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
