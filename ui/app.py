"""
CompliancePatchBench — Streamlit demo (Meta OpenEnv Hackathon 2026)
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st

# --- Global setup -----------------------------------------------------------------
# Backend (FastAPI) base URL. On Hugging Face: set env ENV_BASE_URL to the
# *deployed* API Space public URL (https://<…>.hf.space), not localhost — the UI
# container's localhost is this Streamlit app, not the separate API Space.
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")
GITHUB_URL = os.environ.get(
    "CPB_GITHUB_URL",
    "https://github.com/skypank-coder/CompliancePatchBench",
)
HF_API_SPACE_URL = os.environ.get(
    "CPB_HF_SPACE_URL",
    "https://huggingface.co/spaces/rachana05/Compliance-patch-bench",
)
HF_UI_SPACE_URL = os.environ.get(
    "CPB_HF_UI_SPACE_URL",
    "https://huggingface.co/spaces/rachana05/CompliancePatchBench-UI",
)

def _safe_get(url: str, timeout: float = 12.0) -> Optional[dict]:
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def _safe_post(url: str, json_body: dict, timeout: float = 30.0) -> Optional[dict]:
    try:
        r = requests.post(url, json=json_body, timeout=timeout)
        if r.status_code < 400:
            return r.json()
    except Exception:
        pass
    return None


def check_health(base: str) -> bool:
    data = _safe_get(f"{base.rstrip('/')}/health")
    return bool(data and data.get("status") == "ok")


def fetch_project_api(base: str) -> Dict[str, Any]:
    data = _safe_get(f"{base.rstrip('/')}/project")
    return data if isinstance(data, dict) else {}


def _monorepo_learning_curve_paths() -> List[str]:
    p = Path(__file__).resolve()
    cands: List[str] = []
    if p.parent.name == "ui":
        cands.append(str(p.parents[1] / "project" / "data" / "learning_curve.json"))
    if p.parent.name == "src":
        try:
            cands.append(str(p.parents[3] / "project" / "data" / "learning_curve.json"))
        except IndexError:
            pass
    return cands


def _read_learning_curve_from_disk() -> Tuple[List[float], str]:
    data_dir = os.environ.get("CPB_DATA_DIR", "").strip()
    paths: List[str] = []
    if data_dir:
        paths.append(os.path.join(data_dir, "project", "data", "learning_curve.json"))
    paths.extend(_monorepo_learning_curve_paths())
    paths.extend(
        [
            "project/data/learning_curve.json",
            "reward_curve.json",
            "/content/CompliancePatchBench/project/data/learning_curve.json",
            "/content/CompliancePatchBench/reward_curve.json",
        ]
    )
    seen: set = set()
    for raw in paths:
        if not raw or raw in seen:
            continue
        seen.add(raw)
        ap = os.path.abspath(os.path.normpath(raw))
        if not os.path.isfile(ap):
            continue
        try:
            with open(ap, encoding="utf-8") as f:
                rawj = json.load(f)
        except Exception:
            continue
        if isinstance(rawj, list) and rawj:
            if isinstance(rawj[0], dict):
                return [float(x.get("avg_reward", 0.0)) for x in rawj], ap
            return [float(x) for x in rawj], ap
    return [], ""


def fetch_benchmark_table(base: str) -> Tuple[pd.DataFrame, bool]:
    data = _safe_get(f"{base.rstrip('/')}/benchmark")
    rows = []
    if data and isinstance(data.get("tasks"), list):
        for t in data["tasks"]:
            diff = str(t.get("difficulty", "")).lower()
            om = t.get("our_model")
            rows.append(
                {
                    "Task": t.get("task_id", "—"),
                    "Difficulty": diff or "—",
                    "GPT-4o": float(t.get("gpt4o_baseline", 0.0)),
                    "GPT-4o-mini": float(t.get("gpt4o_mini_baseline", 0.0)),
                    "Our Model": float(om) if om is not None else None,
                }
            )
        return pd.DataFrame(rows), True
    return pd.DataFrame(), False


def run_live_episode(base: str) -> Tuple[Optional[float], bool]:
    try:
        base = base.rstrip("/")
        r = _safe_post(f"{base}/patch/reset", {"task_id": "task1_single_file"})
        if not r or "session_id" not in r:
            return None, False
        sid = r["session_id"]
        obs = r.get("observation") or {}
        files = obs.get("available_files") or ["routes.py"]
        path0 = files[0]
        r2 = _safe_post(
            f"{base}/patch/step",
            {"session_id": sid, "action": {"action_type": "read_file", "path": path0}},
        )
        if not r2:
            return None, False
        r3 = _safe_post(
            f"{base}/patch/step",
            {"session_id": sid, "action": {"action_type": "finalize_patch"}},
        )
        if not r3:
            return None, False
        info = r3.get("info") or {}
        fin = info.get("final_score")
        if fin is not None:
            return float(fin), True
        return None, True
    except Exception:
        return None, False


# --- Page -------------------------------------------------------------------------
st.set_page_config(
    page_title="CompliancePatchBench",
    page_icon="🛡️",
    layout="wide",
)

st.markdown(
    """
    <style>
    div[data-testid="stMetricValue"] { font-size: 1.35rem; }
    .our-model { color: #0a0; font-weight: 600; }
    /* Prevent code blocks from being cut off */
    .stCode code {
        white-space: pre-wrap !important;
        word-break: break-word !important;
    }
    pre {
        overflow-x: auto !important;
        white-space: pre-wrap !important;
    }
    /* Make the 3-column comparison table equal width */
    [data-testid="column"] {
        min-width: 0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---- Sidebar ---------------------------------------------------------------------
with st.sidebar:
    st.title("🛡️ CompliancePatchBench")
    st.caption("Meta OpenEnv Hackathon 2026")
    st.divider()
    if check_health(ENV_BASE_URL):
        st.success("🟢 Environment Online")
    else:
        st.warning("🟡 Using demo data (backend not reachable)")

    st.link_button("GitHub repo", GITHUB_URL, use_container_width=True)
    st.link_button("Hugging Face — API (FastAPI)", HF_API_SPACE_URL, use_container_width=True)
    st.link_button("Hugging Face — Streamlit UI", HF_UI_SPACE_URL, use_container_width=True)

# ---- Session defaults ------------------------------------------------------------
if "using_demo" not in st.session_state:
    st.session_state.using_demo = not check_health(ENV_BASE_URL)
if "final_score_demo" not in st.session_state:
    st.session_state.final_score_demo = None
if "last_live_ok" not in st.session_state:
    st.session_state.last_live_ok = False

_project_payload: Dict[str, Any] = fetch_project_api(ENV_BASE_URL)
_ui_block: Dict[str, Any] = _project_payload.get("ui") if isinstance(_project_payload.get("ui"), dict) else {}
_ld = _ui_block.get("live_demo") if isinstance(_ui_block.get("live_demo"), dict) else {}
_reward_legend: Dict[str, Any] = (
    _ui_block.get("reward_legend") if isinstance(_ui_block.get("reward_legend"), dict) else {}
)
_rl_cfg: Dict[str, Any] = (
    _project_payload.get("rl_training_config")
    if isinstance(_project_payload.get("rl_training_config"), dict)
    else {}
)

# ---- Tabs ------------------------------------------------------------------------
tab_live, tab_train, tab_bench = st.tabs(
    ["🎯 Live Demo", "📈 Training Progress", "🏆 Benchmark"]
)

# === Tab 1: Live Demo ==============================================================
with tab_live:
    st.subheader("Why this benchmark matters")
    st.caption(
        "Baselines can pass tests yet break compliance. A trained RL agent fixes code "
        "without deleting audit trails or hiding PII from hidden checks."
    )
    vcode = str(_ld.get("violation_code") or "— (set project/data/ui_data.json and redeploy API)")
    bcode = str(_ld.get("baseline_code") or "—")
    rcode = str(_ld.get("rl_code") or "—")
    rule = str(_ld.get("rule") or "—")
    sev = str(_ld.get("severity") or "—")
    br = _reward_legend.get("deletion_fail", -1.0)
    gr = _reward_legend.get("good_patch_example")
    try:
        br_f = float(br)
    except (TypeError, ValueError):
        br_f = -1.0
    try:
        gr_f = float(gr) if gr is not None else None
    except (TypeError, ValueError):
        gr_f = None

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("#### 🚨 Violation")
        st.code(vcode, language="python")
        st.markdown(f"**Rule:** {rule}  \n**Severity:** {sev}")

    with c2:
        st.markdown("#### ❌ Baseline Agent")
        st.code(bcode, language="python")
        st.error("Fails hidden constraint (loses auditability)")
        st.metric("Reward (env)", f"{br_f:+.2f}")

    with c3:
        st.markdown("#### ✅ RL Agent")
        st.code(rcode, language="python")
        st.success("Passes CI + compliance")
        st.metric(
            "Reward (reference)",
            f"{gr_f:+.2f}" if gr_f is not None else "—",
        )

    st.divider()
    if st.button("▶ Run Live Episode", type="primary", use_container_width=False):
        with st.spinner("Running episode…"):
            time.sleep(0.15)
            score, live_ok = run_live_episode(ENV_BASE_URL)
            st.session_state.last_live_ok = live_ok
            st.session_state.final_score_demo = score
            if not live_ok:
                st.session_state.using_demo = True
                st.warning("Could not complete live episode — check API or env.")
            else:
                st.session_state.using_demo = False
                st.toast("Episode finished.", icon="✅")

    fs = st.session_state.final_score_demo
    if fs is not None:
        st.metric("Final Score (live)", f"{fs:+.2f}")
    else:
        st.metric("Final Score (live)", "—")
        st.caption("Run an episode to record `final_score` from the API.")


# === Tab 2: Training Progress ======================================================
with tab_train:
    # Load reward curve
    curve_data: List[float] = []
    from_api = False
    data_source_note = ""

    try:
        r = requests.get(f"{ENV_BASE_URL}/rl/learning-curve", timeout=5)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, list) and len(data) > 0:
                curve_data = data
                from_api = True
            elif isinstance(data, dict):
                curve_data = data.get("rewards", data.get("reward_history", data.get("learning_curve", [])))
                from_api = bool(curve_data)
    except Exception:
        pass

    if curve_data and len(curve_data) > 0 and isinstance(curve_data[0], dict):
        curve_data = [float(p.get("avg_reward", 0.0)) for p in curve_data]
    elif curve_data:
        curve_data = [float(x) for x in curve_data]

    # Local / Colab-exported files (same schema as project/data/learning_curve.json)
    if not curve_data or len(curve_data) < 2:
        disk_vals, disk_path = _read_learning_curve_from_disk()
        if disk_vals and len(disk_vals) >= 2:
            curve_data = disk_vals
            from_api = False
            data_source_note = disk_path

    if not from_api and not check_health(ENV_BASE_URL):
        st.warning("Using demo data")
    elif from_api:
        st.caption("Loaded from live API")
    elif data_source_note:
        st.caption(f"Source: {data_source_note}")

    if not curve_data:
        smoothed: List[float] = []
        initial_reward = 0.0
        peak_reward = 0.0
        final_reward = 0.0
        improvement_pct = 0.0
        step_numbers: List[int] = []
    else:
        smoothed = pd.Series(curve_data).rolling(window=5, min_periods=1).mean().tolist()
        initial_reward = smoothed[0]
        peak_reward = max(smoothed)
        final_reward = smoothed[-1]
        improvement_pct = ((final_reward - initial_reward) / max(abs(initial_reward), 0.001)) * 100
        step_numbers = list(range(5, len(curve_data) * 5 + 1, 5))

    col1, col2, m3 = st.columns(3)
    col1.metric("Initial Reward", f"{initial_reward:.2f}")
    col2.metric("Peak Reward", f"{peak_reward:.2f}", f"↑ from {initial_reward:.2f}")
    m3.metric("Final Reward (smoothed)", f"{final_reward:.2f}", f"{improvement_pct:+.0f}%")

    if len(curve_data) >= 2:
        # Build correct DataFrame with step numbers as index
        df_curve = pd.DataFrame(
            {
                "Raw Reward": curve_data,
                "Smoothed (5-step avg)": smoothed,
            },
            index=step_numbers,
        )

        st.subheader("Reward curve (raw + smoothed)")
        st.line_chart(df_curve, color=["#5B9BD5", "#E8703A"])
        st.caption(
            f"Training steps: {step_numbers[0]} → {step_numbers[-1]} | "
            f"{len(curve_data)} logged batches | "
            f"Smoothed trend: {smoothed[0]:.2f} → {smoothed[-1]:.2f} | "
            f"Peak: {peak_reward:.2f} at step {step_numbers[smoothed.index(peak_reward)]}"
        )
    else:
        st.warning(
            "No learning curve data. Run Colab training, commit `project/data/learning_curve.json`, "
            "redeploy the API, or set CPB_DATA_DIR / place the file under project/data."
        )

    del_fail = _reward_legend.get("deletion_fail", -1.0)
    good_ex = _reward_legend.get("good_patch_example", 1.7)
    try:
        df_s = float(del_fail)
    except (TypeError, ValueError):
        df_s = -1.0
    try:
        gf_s = float(good_ex)
    except (TypeError, ValueError):
        gf_s = 1.7
    b1, b2 = st.columns(2)
    with b1:
        st.error(f"❌ Deletes line → FAIL → Reward **{df_s:+.1f}**")
    with b2:
        st.success(f"✅ Proper patch → PASS → Reward **{gf_s:+.1f}** (env-dependent)")

    st.info("🔒 Deleting code is penalized. The model learned to fix, not cheat.")


# === Tab 3: Benchmark ==============================================================
with tab_bench:
    df, ok = fetch_benchmark_table(ENV_BASE_URL)
    if not ok or df.empty:
        st.warning("Backend offline or no benchmark rows — set ENV_BASE_URL to the API Space.")
    else:

        def _green_our_model_col(col: pd.Series) -> List[str]:
            return ["color: #0a0; font-weight: 600"] * len(col)

        try:
            st.dataframe(
                df.style.apply(_green_our_model_col, axis=0, subset=["Our Model"]),
                use_container_width=True,
                hide_index=True,
            )
        except Exception:
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Our Model": st.column_config.NumberColumn("Our Model", format="%.2f")
                },
            )

    st.divider()
    t1, t2, t3 = st.columns(3)
    with t1:
        st.markdown("**Themes**")
        st.markdown("- World Modeling\n- Self-Improvement")
    with t2:
        st.markdown("**Stack (from API / Colab config)**")
        bm = _rl_cfg.get("base_model") or _ui_block.get("base_model")
        lines = []
        if bm:
            lines.append(f"- **Base model:** `{bm}`")
        lines.append("- TRL GRPO / iterative RL (see `project/data/rl_training_log.json`)")
        lines.append("- FastAPI + CompliancePatchEnv")
        st.markdown("\n".join(lines) if lines else "— (train in Colab to populate `rl_training_log.json`)")
    with t3:
        st.markdown("**Why not cheatable**")
        bullets = _ui_block.get("why_not_cheatable_bullets")
        if isinstance(bullets, list) and bullets:
            st.markdown("\n".join(f"- {b}" for b in bullets))
        else:
            st.markdown(
                f"- deletion penalty ≈ **{_reward_legend.get('deletion_fail', -1.0)}**\n"
                "- hidden constraints in env\n- adversarial tasks in pool"
            )


# ---- Footer ----------------------------------------------------------------------
st.divider()
st.caption(
    f"[GitHub]({GITHUB_URL}) · [HF API Space]({HF_API_SPACE_URL}) · [HF UI Space]({HF_UI_SPACE_URL}) · Meta OpenEnv Hackathon 2026"
)
