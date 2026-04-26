"""
CompliancePatchBench — Streamlit demo (Meta OpenEnv Hackathon 2026)
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
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

def _streamlit_is_dark() -> bool:
    try:
        th = st.context.theme  # type: ignore[attr-defined]
        if th is not None:
            b = getattr(th, "base", None)
            if b is not None:
                return str(b).lower() == "dark"
    except Exception:
        pass
    return False


def _safe_get(url: str, timeout: float = 12.0) -> Optional[dict]:
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def _fetch_training_curve(base: str) -> Dict[str, Any]:
    """
    Real training data from API: GET /training-curve (or /rl/learning-curve), no dummy fill.
    Response includes `steps` and `rewards` from `project/data/learning_curve.json`.
    """
    data = None
    last_err: Optional[str] = None
    for path in ("/training-curve", "/rl/learning-curve"):
        url = f"{base.rstrip('/')}{path}"
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                data = r.json()
                break
            last_err = f"HTTP {r.status_code} from {path}"
        except Exception as e:
            last_err = str(e)
    if data is None:
        return {
            "ok": False,
            "err": last_err or "request failed",
            "learning_curve": [],
            "derived": None,
            "rewards": [],
            "steps": [],
            "note": None,
        }
    if not isinstance(data, dict):
        return {
            "ok": True,
            "err": None,
            "learning_curve": [],
            "derived": None,
            "rewards": [],
            "steps": [],
            "note": None,
        }
    lc = data.get("learning_curve") or []
    return {
        "ok": True,
        "err": None,
        "learning_curve": lc if isinstance(lc, list) else [],
        "derived": data.get("derived") if isinstance(data.get("derived"), dict) else None,
        "rewards": list(data.get("rewards") or []),
        "steps": list(data.get("steps") or []),
        "note": data.get("note"),
    }


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


def _global_dashboard_css() -> str:
    return """
    <style>
    :root { --cpb-radius: 16px;
      --cpb-surface: var(--secondary-background-color, var(--background-color));
      --cpb-text: var(--text-color, #0f172a);
      --cpb-border: rgba(15, 23, 42, 0.12);
      --cpb-shadow: 0 6px 24px rgba(15, 23, 42, 0.07);
    }
    [data-color-scheme="dark"] {
      --cpb-border: rgba(255, 255, 255, 0.12);
      --cpb-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    }
    .stApp { color: var(--text-color) !important; }
    div[data-testid="stAppViewContainer"] { background: var(--background-color) !important; }
    [data-testid="stHeader"] { background: var(--background-color) !important; }
    .main .block-container {
        max-width: 1080px !important;
        padding: 1.75rem 2rem 3rem 2rem !important;
    }
    h1, h2, h3, h4 { color: var(--text-color) !important; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px; background: var(--secondary-background-color, var(--background-color)); border-radius: 14px; padding: 8px;
        box-shadow: var(--cpb-shadow);
        border: 1px solid var(--cpb-border);
    }
    [data-baseweb="tab-panel"] {
        background: var(--cpb-surface) !important;
        border-radius: var(--cpb-radius) !important;
        padding: 1.5rem 1.5rem 1.75rem 1.5rem !important;
        margin-top: 0.5rem;
        box-shadow: var(--cpb-shadow) !important;
        border: 1px solid var(--cpb-border) !important;
        animation: cpb-fadein 0.4s ease-out;
    }
    @keyframes cpb-fadein { from { opacity: 0.92; } to { opacity: 1; } }
    [data-testid="stMetricValue"] { font-size: 1.3rem !important; font-weight: 700 !important; }
    [data-testid="stCodeBlock"] {
        background: var(--secondary-background-color, var(--background-color)) !important;
        color: var(--text-color) !important;
        border-radius: 12px !important;
        box-shadow: inset 0 0 0 1px var(--cpb-border) !important;
    }
    [data-testid="stCodeBlock"] pre, [data-testid="stCodeBlock"] code {
        white-space: pre-wrap !important; word-break: break-word !important; font-size: 0.8rem !important;
    }
    [data-testid="column"] { min-width: 0 !important; }
    .cpb-pill { display: inline-block; padding: 0.22rem 0.6rem; border-radius: 999px; font-size: 0.7rem; font-weight: 800; }
    .cpb-pill-bad { background: rgba(220, 38, 38, 0.2); color: #fecaca; }
    .cpb-pill-good { background: rgba(5, 150, 105, 0.25); color: #6ee7b7; }
    [data-testid="stAppViewContainer"][data-color-scheme="light"] .cpb-pill-bad { background: #fee2e2; color: #b91c1c; }
    [data-testid="stAppViewContainer"][data-color-scheme="light"] .cpb-pill-good { background: #d1fae5; color: #047857; }
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:has([data-testid="stCodeBlock"]):nth-child(1) {
        align-self: stretch; min-height: 360px;
        background: color-mix(in srgb, var(--secondary-background-color) 90%, #fecaca) !important;
        border: 1px solid var(--cpb-border); border-radius: 14px; padding: 0.75rem 0.6rem 1rem;
    }
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:has([data-testid="stCodeBlock"]):nth-child(2) {
        align-self: stretch; min-height: 360px;
        background: color-mix(in srgb, var(--secondary-background-color) 88%, #fda4af) !important;
        border: 1px solid var(--cpb-border); border-radius: 14px; padding: 0.75rem 0.6rem 1rem;
    }
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:has([data-testid="stCodeBlock"]):nth-child(3) {
        align-self: stretch; min-height: 360px;
        background: color-mix(in srgb, var(--secondary-background-color) 90%, #6ee7b7) !important;
        border: 1px solid var(--cpb-border); border-radius: 14px; padding: 0.75rem 0.6rem 1rem;
    }
    @supports not (background: color-mix(in srgb, red, blue)) {
      div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:has([data-testid="stCodeBlock"]):nth-child(1) {
        background: rgba(254, 202, 202, 0.2) !important; }
      div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:has([data-testid="stCodeBlock"]):nth-child(2) {
        background: rgba(252, 165, 165, 0.18) !important; }
      div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:has([data-testid="stCodeBlock"]):nth-child(3) {
        background: rgba(110, 231, 183, 0.15) !important; }
    }
    button[kind="primary"] {
        background: linear-gradient(180deg, #ef4444, #dc2626) !important; border: none !important;
        font-weight: 700 !important; border-radius: 12px !important; padding: 0.6rem 1.25rem !important;
        box-shadow: 0 4px 14px rgba(220, 38, 38, 0.35) !important;
    }
    .cpb-muted { color: var(--cpb-muted) !important; }
    .cpb-heading { color: var(--text-color) !important; }
    .cpb-hero {
        text-align: center; border-radius: 16px; padding: 1.1rem 1.25rem; margin-bottom: 0.75rem;
        background: var(--background-color) !important;
        box-shadow: var(--cpb-shadow) !important;
        border: 1px solid var(--cpb-border) !important;
    }
    .cpb-hero .cpb-hero-title {
        font-size: 1.85rem; font-weight: 800; letter-spacing: -0.03em; color: var(--text-color) !important; display: block;
    }
    .cpb-hero .cpb-hero-sub {
        color: var(--text-color) !important; opacity: 0.72; font-size: 0.95rem; margin: 0.35rem 0 0 0;
    }
    .cpb-pill-warn { background: var(--secondary-background-color) !important; color: var(--text-color) !important; border: 1px solid var(--cpb-border) !important; }
    </style>
    """


def _build_reward_plotly(
    step_numbers: List[Any],
    curve_data: List[float],
    smoothed: List[float],
    *,
    dark: bool = False,
    xaxis_title: str = "Trainer step",
    yaxis_title: str = "Reward",
    smooth_name: str = "Smoothed (rolling)",
    peak_step: Optional[int] = None,
    peak_reward: Optional[float] = None,
) -> go.Figure:
    if peak_step is not None and peak_reward is not None:
        peak_x, peak_y = int(peak_step), float(peak_reward)
    else:
        peak_i = int(np.argmax(curve_data)) if curve_data else 0
        peak_x = step_numbers[peak_i] if step_numbers and peak_i < len(step_numbers) else 0
        peak_y = float(curve_data[peak_i]) if curve_data and peak_i < len(curve_data) else 0.0
    fg = "#e2e8f0" if dark else "#0f172a"
    ann_bg = "rgba(30,41,59,0.95)" if dark else "rgba(255, 255, 255, 0.95)"
    ann_brd = "rgba(148,163,184,0.35)" if dark else "#e2e8f0"
    p_bg = "rgba(15,23,42,0.5)" if dark else "rgba(255,255,255,0.72)"
    leg_bg = "rgba(15,23,42,0.75)" if dark else "rgba(255,255,255,0.85)"
    grid = "rgba(148,163,184,0.2)" if dark else "rgba(148,163,184,0.12)"
    raw_line = "rgba(96, 165, 250, 0.55)" if dark else "rgba(37, 99, 235, 0.4)"
    m_line = "#fb923c" if dark else "rgb(234, 88, 12)"
    m_peak = "#fbbf24" if dark else "#b45309"
    m_ring = "rgba(15,23,42,0.9)" if dark else "#fff"

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=step_numbers,
            y=curve_data,
            mode="lines",
            name="Raw",
            line=dict(color=raw_line, width=1),
            hovertemplate="Step %{x}<br>Raw reward %{y:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=step_numbers,
            y=smoothed,
            mode="lines",
            name=smooth_name,
            line=dict(color=m_line, width=3.2),
            hovertemplate="Step %{x}<br>Smoothed %{y:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[peak_x],
            y=[peak_y],
            mode="markers",
            name="Peak (raw reward max)",
            marker=dict(size=14, color=m_peak, line=dict(color=m_ring, width=2)),
            hovertemplate="Peak → step %{x}<br>reward %{y:.3f}<extra></extra>",
        )
    )
    fig.add_vline(
        x=peak_x,
        line_width=1,
        line_dash="dash",
        line_color="rgba(148, 163, 184, 0.45)" if dark else "rgba(100, 116, 139, 0.55)",
    )
    fig.add_annotation(
        x=peak_x,
        y=peak_y,
        xanchor="left",
        text=f"Peak {peak_y:.2f} @ step {peak_x}",
        showarrow=True,
        arrowhead=1,
        ax=32,
        ay=-36,
        bgcolor=ann_bg,
        bordercolor=ann_brd,
        font=dict(size=12, color=fg),
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=p_bg,
        margin=dict(l=60, r=28, t=52, b=52),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor=leg_bg,
        ),
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        xaxis=dict(
            gridcolor=grid,
            zerolinecolor=grid,
            color=fg,
        ),
        yaxis=dict(
            gridcolor=grid,
            zerolinecolor=grid,
            color=fg,
        ),
        hovermode="x unified",
        font=dict(family="system-ui, sans-serif", color=fg),
        uirevision="cpb-train",
        transition=dict(duration=450, easing="cubic-in-out"),
    )
    return fig


def _add_benchmark_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    badges: List[str] = []
    for _, r in out.iterrows():
        try:
            om = r.get("Our Model")
            g4 = r.get("GPT-4o")
            if pd.notna(om) and pd.notna(g4) and float(om) > float(g4):
                badges.append("↑ beats GPT-4o")
            else:
                badges.append("")
        except (TypeError, ValueError):
            badges.append("")
    out["Label"] = badges
    return out


def _style_benchmark_df(df: pd.DataFrame) -> Any:
    num_cols = [c for c in ("GPT-4o", "GPT-4o-mini", "Our Model") if c in df.columns]
    s = df.style
    gpt = [c for c in ("GPT-4o", "GPT-4o-mini") if c in df.columns]
    for col in gpt:
        s = s.background_gradient(cmap="RdYlGn", subset=[col], axis=0)
    if "Our Model" in df.columns:
        s = s.background_gradient(
            cmap="Greens", subset=["Our Model"], axis=0, low=0.2, high=0.95
        )
    if num_cols:
        s = s.apply(
            lambda row: [
                "font-weight: 800" if (pd.notna(v) and v == row.max()) else ""
                for v in row
            ],
            axis=1,
            subset=num_cols,
        )
    return s.set_properties(**{"text-align": "center"}, subset=num_cols or None)


# --- Page -------------------------------------------------------------------------
st.set_page_config(
    page_title="CompliancePatchBench",
    page_icon="🛡️",
    layout="wide",
)

st.markdown(_global_dashboard_css(), unsafe_allow_html=True)
st.markdown(
    '<div class="cpb-hero">'
    '<span class="cpb-hero-title">CompliancePatchBench</span>'
    '<p class="cpb-hero-sub">'
    "ML dashboard — metrics and curves load only from the API (no mock numbers)</p></div>",
    unsafe_allow_html=True,
)

# ---- Session defaults ------------------------------------------------------------
if "using_demo" not in st.session_state:
    st.session_state.using_demo = not check_health(ENV_BASE_URL)
if "final_score_demo" not in st.session_state:
    st.session_state.final_score_demo = None
if "last_live_ok" not in st.session_state:
    st.session_state.last_live_ok = False

# Single GET /project (includes ui: public Colab link + live-demo copy for tabs)
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

# ---- Sidebar ---------------------------------------------------------------------
with st.sidebar:
    st.title("🛡️ CompliancePatchBench")
    st.caption("Meta OpenEnv 2026")
    st.divider()
    if check_health(ENV_BASE_URL):
        st.success("API online")
    else:
        st.warning("Set **ENV_BASE_URL** to the FastAPI `*.hf.space` URL for live data")

    st.link_button("GitHub repo", GITHUB_URL, width="stretch")
    st.link_button("Hugging Face — API (FastAPI)", HF_API_SPACE_URL, width="stretch")
    st.link_button("Hugging Face — Streamlit UI", HF_UI_SPACE_URL, width="stretch")
    _colab_url = str(_ui_block.get("public_colab_url", "") or "").strip()
    if _colab_url.startswith("http"):
        st.link_button("Open training notebook (Colab)", _colab_url, width="stretch")

# ---- Tabs ------------------------------------------------------------------------
tab_live, tab_train, tab_bench = st.tabs(
    ["🎯 Live Demo", "📈 Training Progress", "🏆 Benchmark"]
)

# === Tab 1: Live Demo ==============================================================
with tab_live:
    vcode = str(_ld.get("violation_code") or "— (configure in API `ui_data.json`)")
    bcode = str(_ld.get("baseline_code") or "—")
    rcode = str(_ld.get("rl_code") or "—")
    rule = str(_ld.get("rule") or "—")
    sev = str(_ld.get("severity") or "—")
    br = _reward_legend.get("deletion_fail")
    gr = _reward_legend.get("good_patch_example")
    try:
        br_f = float(br) if br is not None else None
    except (TypeError, ValueError):
        br_f = None
    try:
        gr_f = float(gr) if gr is not None else None
    except (TypeError, ValueError):
        gr_f = None

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        st.markdown(
            '<p class="cpb-pill-warn" style="margin:0 0 0.5rem 0;display:inline-block;'
            'padding:0.22rem 0.6rem;border-radius:999px;font-size:0.7rem;">Violation</p>',
            unsafe_allow_html=True,
        )
        st.code(vcode, language="python", line_numbers=False)
        st.caption(f"Rule: {rule} · Severity: {sev}")

    with c2:
        st.markdown("**Baseline**")
        st.code(bcode, language="python", line_numbers=False)
        st.metric("Env reward (from config)", f"{br_f:+.2f}" if br_f is not None else "—")

    with c3:
        st.markdown("**RL / patch example**")
        st.code(rcode, language="python", line_numbers=False)
        st.metric("Reference reward (from config)", f"{gr_f:+.2f}" if gr_f is not None else "—")

    st.divider()
    st.subheader("Best recorded episode (API)")
    if not check_health(ENV_BASE_URL):
        st.warning("Connect **ENV_BASE_URL** to load `/stats/best-episode`.")
    else:
        with st.spinner("Loading best episode…"):
            time.sleep(0.05)
            ep_be = _safe_get(f"{ENV_BASE_URL.rstrip('/')}/stats/best-episode", timeout=10.0)
        if ep_be and isinstance(ep_be.get("steps"), list) and len(ep_be["steps"]) > 0:
            src = str(ep_be.get("source", ""))
            if src == "reference_episode":
                st.caption("Reference episode bundled with API when no `trajectories_rl.jsonl` is present (not a live logged run).")
            cap_bits = [
                f"Task: {ep_be.get('task_id', '—')}",
                f"Difficulty: {ep_be.get('difficulty', '—')}",
                f"Score: {float(ep_be.get('final_score', 0) or 0):+.2f}",
            ]
            if src:
                cap_bits.append(f"Source: {src}")
            st.caption(" · ".join(cap_bits))
            for s in ep_be["steps"]:
                reward = float(s.get("reward", 0) or 0)
                r_str = f"+{reward:.1f}" if reward > 0 else f"{reward:.1f}"
                act = s.get("action", "?")
                sn = s.get("step", "?")
                st.text(f"Step {sn}  {act}  reward {r_str}  — {s.get('note', '')}")
            status = str(ep_be.get("status", "UNKNOWN"))
            score = float(ep_be.get("final_score", 0) or 0)
            del_a = bool(ep_be.get("deletion_attempted"))
            ho = bool(ep_be.get("hidden_oracle_passed"))
            if status == "SUCCESS":
                st.success(
                    f"**{status}** — {score:+.2f} · deletion: {'Yes' if del_a else 'No'} · "
                    f"hidden oracle: {'PASS' if ho else 'FAIL'}"
                )
            else:
                st.warning(f"{status} — {score:+.2f}")
        else:
            st.info("No episode from API (empty or unreachable).")

    st.divider()
    _b0, _b1, _b2 = st.columns([1, 2, 1])
    with _b1:
        if st.button("▶ Run Live Episode", type="primary", width="stretch"):
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
        st.metric("Final score (from env)", f"{fs:+.2f}")
    else:
        st.metric("Final score (from env)", "—")
        st.caption("Run an episode to record `final_score` from the API.")


# === Tab 2: Training Progress ======================================================
with tab_train:
    st.caption("Data source: FastAPI `GET /training-curve` → `project/data/learning_curve.json` (re-read on each request).")

    with st.spinner("Loading /training-curve…"):
        res = _fetch_training_curve(ENV_BASE_URL)

    steps: List[int] = []
    rewards: List[float] = []
    for s in res.get("steps") or []:
        try:
            steps.append(int(s))
        except (TypeError, ValueError):
            pass
    for v in res.get("rewards") or []:
        try:
            rewards.append(float(v))
        except (TypeError, ValueError):
            pass
    n_pts = min(len(steps), len(rewards))
    steps = steps[:n_pts]
    rewards = rewards[:n_pts]
    if len(steps) != len(rewards) and n_pts:
        st.error("API returned mismatched `steps` / `rewards` lengths — not plotting.")

    with st.expander("Debug: steps & rewards (from API, no client-side fill)", expanded=False):
        st.write(f"n={len(steps)}")
        st.json({"steps": steps, "rewards": rewards})

    if not res.get("ok"):
        st.error("Unable to load training data from the API.")
        st.caption(str(res.get("err", "Request failed")))

    elif not steps:
        st.error("⛔ No training points on the server — `learning_curve.json` is empty or missing in the API image.")
        st.info("Export **`project/data/learning_curve.json`** from Colab / `model_training`, commit, **redeploy the API Space**; or `POST /upload-training-log` with `CPB_TRAINING_UPLOAD_TOKEN` + `X-CPB-Token`.")

    elif len(steps) < 3:
        st.info(
            "Training just started — run **≥10** GRPO / trainer steps, export `project/data/learning_curve.json` "
            "and redeploy the **API** (or upload via `POST /upload-training-log`). "
            f"**Current points: {len(steps)}** (need ≥3 to plot without misleading line)."
        )
        if res.get("note"):
            st.caption(str(res["note"]))

    else:
        derived: Optional[Dict[str, Any]] = res.get("derived")
        if not isinstance(derived, dict) or not derived:
            st.error("API returned no `derived` block — cannot show metrics (check server logs).")
        else:
            w = int(derived.get("smoothing_window", 5))
            sml = derived.get("smoothed_rewards")
            if isinstance(sml, list) and len(sml) == len(rewards) and all(
                isinstance(x, (int, float)) for x in sml
            ):
                smoothed: List[float] = [float(x) for x in sml]
            else:
                win = 5 if len(rewards) >= 5 else 3
                smoothed = (
                    pd.Series(rewards)
                    .rolling(window=win, min_periods=1)
                    .mean()
                    .tolist()
                )
            peak_s = derived.get("peak_step")
            peak_r = derived.get("peak_reward")
            try:
                pstep = int(peak_s) if peak_s is not None else None
            except (TypeError, ValueError):
                pstep = None
            try:
                prew = float(peak_r) if peak_r is not None else None
            except (TypeError, ValueError):
                prew = None
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Initial (mean of first 3)", f"{float(derived.get('initial_reward', 0)):+.4f}")
            with c2:
                st.metric("Final reward (last point)", f"{float(derived.get('final_reward', 0)):+.4f}")
            with c3:
                st.metric("Peak reward (max of raw)", f"{float(derived.get('peak_reward', 0)):+.4f}", f"step {derived.get('peak_step', '—')}")
            with c4:
                st.metric("Points in file", str(int(derived.get("total_points", n_pts) or 0)))
            if derived.get("peak_success_rate") is not None and derived.get("initial_success_mean") is not None:
                s1, s2 = st.columns(2)
                with s1:
                    st.metric("Success (initial mean)", f"{float(derived['initial_success_mean']):.0%}")
                with s2:
                    st.metric("Success (last point)", f"{float(derived.get('final_success', 0)):.0%}")
            st.caption(str(derived.get("trend", "")))
            if res.get("note"):
                st.caption(str(res["note"]))

            st.markdown("#### Learning curve (raw + smoothed, from your log file)")
            _train_dark = _streamlit_is_dark()
            sm_name = f"Smoothed (rolling, window={w})"
            fig = _build_reward_plotly(
                steps,
                rewards,
                smoothed,
                dark=_train_dark,
                xaxis_title="Trainer step",
                yaxis_title="Reward",
                smooth_name=sm_name,
                peak_step=pstep,
                peak_reward=prew,
            )
            st.plotly_chart(
                fig,
                width="stretch",
                config={"displayModeBar": True, "displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
            )
            st.caption(
                f"Step range: {steps[0]} – {steps[-1]} (from `learning_curve.json` rows, not array index)"
            )

# === Tab 3: Benchmark ==============================================================
with tab_bench:
    df, ok = fetch_benchmark_table(ENV_BASE_URL)
    if not ok or df.empty:
        st.warning("Backend offline or no benchmark rows — set ENV_BASE_URL to the API Space.")
    else:
        dfx = _add_benchmark_column(df)
        col_order = [c for c in ("Task", "Difficulty", "GPT-4o", "GPT-4o-mini", "Our Model", "Label") if c in dfx.columns]
        dfx = dfx[col_order]
        try:
            st.dataframe(
                _style_benchmark_df(dfx),
                width="stretch",
                hide_index=True,
                column_config={
                    "Label": st.column_config.TextColumn(" "),
                },
            )
        except Exception:
            st.dataframe(
                dfx,
                width="stretch",
                hide_index=True,
            )

    bm = _rl_cfg.get("base_model") or _ui_block.get("base_model")
    if bm:
        st.caption(f"Base model (from project API / training config): `{bm}`")


# ---- Footer ----------------------------------------------------------------------
st.divider()
st.caption(
    f"[GitHub]({GITHUB_URL}) · [HF API Space]({HF_API_SPACE_URL}) · [HF UI Space]({HF_UI_SPACE_URL}) · Meta OpenEnv Hackathon 2026"
)
