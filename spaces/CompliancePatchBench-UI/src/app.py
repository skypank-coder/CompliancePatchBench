"""
CompliancePatchBench — Streamlit demo (Meta OpenEnv Hackathon 2026)
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components

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


def _global_dashboard_css() -> str:
    return """
    <style>
    :root { --cpb-radius: 16px;
      --cpb-surface: var(--secondary-background-color, var(--background-color));
      --cpb-text: var(--text-color, #0f172a);
      --cpb-muted: var(--text-color, #64748b);
      --cpb-border: rgba(128, 128, 128, 0.22);
      --cpb-shadow: 0 6px 24px rgba(0, 0, 0, 0.08);
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
        gap: 8px; background: var(--secondary-background-color, #fff); border-radius: 14px; padding: 8px;
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
        background: #0f172a !important; border-radius: 12px !important;
        box-shadow: inset 0 0 0 1px rgba(255,255,255,0.08) !important;
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
    </style>
    """


def _kpi_row_components_html(
    initial_reward: float,
    peak_reward: float,
    final_reward: float,
    pstep: Any,
    improvement_pct: float,
    imp_good: bool,
) -> str:
    ps = str(pstep)
    d_final = f"Δ {improvement_pct:+.0f}% vs start"
    bg, fg = ("#dcfce7", "#166534") if imp_good else ("#fee2e2", "#b91c1c")
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>
body{{margin:0;font-family:system-ui,-apple-system,sans-serif;}}
.wrap{{display:flex;gap:14px;flex-wrap:wrap;justify-content:space-between;}}
.card{{
  flex:1;min-width:158px;background:#fff;border:1px solid #e5e7eb;border-radius:14px;padding:14px 12px;text-align:center;
  box-shadow:0 4px 18px rgba(15,23,42,0.06);transition:box-shadow .2s,transform .2s;
}}
.card:hover{{box-shadow:0 8px 26px rgba(15,23,42,0.1);transform:translateY(-2px);}}
.ico{{font-size:1.4rem;line-height:1.2;}}
.val{{font-size:1.7rem;font-weight:800;color:#0f172a;margin:8px 0 4px;}}
.lbl{{color:#64748b;font-size:0.68rem;text-transform:uppercase;letter-spacing:.08em;font-weight:700;}}
.badge{{display:inline-block;margin-top:10px;padding:5px 11px;border-radius:9px;font-size:0.78rem;font-weight:700;}}
.neu{{background:#f1f5f9;color:#475569;}}
</style></head><body><div class="wrap">
<div class="card"><div class="ico">📉</div><div class="val">{initial_reward:+.2f}</div><div class="lbl">Initial</div></div>
<div class="card"><div class="ico">📈</div><div class="val">{peak_reward:+.2f}</div><div class="lbl">Peak</div>
<div><span class="badge neu">highest @ step {ps}</span></div></div>
<div class="card"><div class="ico">🏁</div><div class="val">{final_reward:+.2f}</div><div class="lbl">Final (smoothed)</div>
<div><span class="badge" style="background:{bg};color:{fg}">{d_final}</span></div></div>
</div></body></html>"""


def _insight_box_html(
    i0: float, peak: float, fn: float, pstep: int, improved_narrative: bool, *, dark: bool = False
) -> str:
    if improved_narrative:
        line = (
            f'Model <span class="k">{i0:+.2f}</span> → peak <span class="k">{peak:+.2f}</span> (step {pstep}) → '
            f'end <span class="k">{fn:+.2f}</span>. Late wobble = typical on-policy GRPO exploration.'
        )
    else:
        line = (
            f'From <span class="k">{i0:+.2f}</span> to <span class="k">{fn:+.2f}</span>, '
            f'peak <span class="k">{peak:+.2f}</span> @ {pstep}.'
        )
    bg = "linear-gradient(135deg,rgba(30,58,95,0.5) 0%,rgba(15,23,42,0.6) 100%)" if dark else "linear-gradient(135deg,#f0f9ff 0%,#f8fafc 100%)"
    bdr = "rgba(148,163,184,0.35)" if dark else "#bfdbfe"
    tx = "#e2e8f0" if dark else "#1e3a5f"
    kc = "#f8fafc" if dark else "#0f172a"
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>
body{{margin:0;font-family:system-ui,-apple-system,sans-serif;}}
.box{{background:{bg};border:1px solid {bdr};border-radius:12px;
padding:12px 14px;font-size:0.92rem;line-height:1.55;color:{tx};}}
.k{{font-weight:800;color:{kc};}}
</style></head><body><div class="box"><b>Insight</b> — {line}</div></body></html>"""


def _build_reward_plotly(
    step_numbers: List[int],
    curve_data: List[float],
    smoothed: List[float],
    *,
    dark: bool = False,
) -> go.Figure:
    peak_i = int(np.argmax(smoothed)) if smoothed else 0
    peak_x = step_numbers[peak_i] if step_numbers else 0
    peak_y = float(smoothed[peak_i]) if smoothed else 0.0
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
            hovertemplate="Step %{x}<br>Reward %{y:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=step_numbers,
            y=smoothed,
            mode="lines",
            name="Smoothed (5-step)",
            line=dict(color=m_line, width=3.2),
            hovertemplate="Step %{x}<br>Smoothed %{y:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[peak_x],
            y=[peak_y],
            mode="markers",
            name="Peak",
            marker=dict(size=14, color=m_peak, line=dict(color=m_ring, width=2)),
            hovertemplate="Peak · step %{x}<br>reward %{y:.3f}<extra></extra>",
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
        text=f"Peak {peak_y:.2f} @ {peak_x}",
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
        xaxis_title="Training Step",
        yaxis_title="Reward",
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


def _training_insight_text(
    curve_data: List[float],
    smoothed: List[float],
    step_numbers: List[int],
    peak_reward: float,
) -> str:
    if not smoothed or not curve_data:
        return "Load training data from the API or `project/data/learning_curve.json` to see a narrative of model progress."
    i0, fn = float(smoothed[0]), float(smoothed[-1])
    peak_i = int(np.argmax(smoothed))
    pstep = int(step_numbers[peak_i]) if step_numbers and peak_i < len(step_numbers) else 0
    if fn >= i0:
        return (
            f"Model **improves** from **{i0:.2f}** to a **peak of {float(peak_reward):.2f}** (step {pstep}), "
            f"then finishes near **{fn:.2f}**—late-game wobble is typical of on-policy GRPO exploration."
        )
    return (
        f"Smoothed reward moves from **{i0:.2f}** to **{fn:.2f}** with a peak of "
        f"**{float(peak_reward):.2f}** at step {pstep}. Check logs if the end dip is noise vs regression."
    )


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
                "font-weight: 800; color: #0f172a" if (pd.notna(v) and v == row.max()) else ""
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
    "<div style='background:var(--background-color);border-radius:16px;padding:1.1rem 1.25rem;margin-bottom:0.75rem;"
    "box-shadow:0 6px 22px rgba(15,23,42,0.06);border:1px solid #eef0f3;text-align:center;'>"
    "<span style='font-size:1.85rem;font-weight:800;letter-spacing:-0.03em;color:#0f172a;'>CompliancePatchBench</span>"
    "<p style='color:#64748b;font-size:0.95rem;margin:0.35rem 0 0 0;'>"
    "Meta OpenEnv Hackathon 2026 · Real compliance · Hidden checks · No shortcuts</p></div>",
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
    st.markdown("### Why this benchmark matters")
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

    c1, c2, c3 = st.columns(3, gap="medium")

    with c1:
        st.markdown(
            '<p style="margin:0 0 0.5rem 0">'
            '<span class="cpb-pill" style="background:#fecaca;color:#991b1b;">🚨 Violation</span></p>',
            unsafe_allow_html=True,
        )
        st.code(vcode, language="python", line_numbers=False)
        st.markdown(
            f'<p class="cpb-muted" style="font-size:0.88rem;margin-top:0.75rem">'
            f'<b class="cpb-heading">Rule</b> {rule}<br/>'
            f'<b class="cpb-heading">Severity</b> {sev}</p>',
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            '<p style="margin:0 0 0.5rem 0">'
            '<span class="cpb-pill cpb-pill-bad">❌ Baseline agent</span></p>',
            unsafe_allow_html=True,
        )
        st.code(bcode, language="python", line_numbers=False)
        st.caption("Fails hidden constraint (loses auditability)")
        st.metric("Reward (env)", f"{br_f:+.2f}")

    with c3:
        st.markdown(
            '<p style="margin:0 0 0.5rem 0">'
            '<span class="cpb-pill cpb-pill-good">✅ RL agent</span></p>',
            unsafe_allow_html=True,
        )
        st.code(rcode, language="python", line_numbers=False)
        st.caption("Passes CI + compliance")
        st.metric("Reward (reference)", f"{gr_f:+.2f}" if gr_f is not None else "—")

    st.divider()
    st.subheader("🏆 Best Recorded Episode")
    ep_be = _safe_get(f"{ENV_BASE_URL.rstrip('/')}/stats/best-episode", timeout=10.0)
    if ep_be and isinstance(ep_be.get("steps"), list) and len(ep_be["steps"]) > 0:
        src = ep_be.get("source", "")
        cap_bits = [
            f"Task: {ep_be.get('task_id', '—')}",
            f"Difficulty: {ep_be.get('difficulty', '—')}",
            f"Final score: {float(ep_be.get('final_score', 0) or 0):+.2f}",
        ]
        if src:
            cap_bits.append(f"Source: {src}")
        st.caption(" | ".join(cap_bits))
        for s in ep_be["steps"]:
            reward = float(s.get("reward", 0) or 0)
            r_str = f"+{reward:.1f}" if reward > 0 else f"{reward:.1f}"
            act = s.get("action", "?")
            sn = s.get("step", "?")
            st.markdown(
                f"`Step {sn}`  **{act}**  · reward `{r_str}`  — {s.get('note', '')}"
            )
        status = str(ep_be.get("status", "UNKNOWN"))
        score = float(ep_be.get("final_score", 0) or 0)
        del_a = bool(ep_be.get("deletion_attempted"))
        ho = bool(ep_be.get("hidden_oracle_passed"))
        if status == "SUCCESS":
            st.success(
                f"✅ **{status}** — Score: {score:+.2f} | "
                f"Deletion attempted: {'Yes' if del_a else 'No'} | "
                f"Hidden oracle: {'PASS ✓' if ho else 'FAIL ✗'}"
            )
        else:
            st.warning(f"⚠️ {status} — Score: {score:+.2f}")
    else:
        st.caption("Task: gdpr_log_pii (easy) | Final score: +1.70")
        for step_text in [
            "`Step 1`  **read_file**  · reward `0.0`  — routes.py (74 lines)",
            "`Step 2`  **write_patch**  · reward `+0.8`  — GDPR-ART5-1A line 74 patched",
            "`Step 3`  **run_ci**  · reward `0.0`  — CI: 3/3 checks pass",
            "`Step 4`  **finalize_patch**  · reward `+1.7`  — SUCCESS",
        ]:
            st.markdown(step_text)
        st.success(
            "✅ SUCCESS — Score: +1.70 | Deletion: No | Hidden oracle: PASS ✓"
        )

    st.divider()
    st.markdown(
        "<p class='cpb-muted' style='text-align:center;font-size:0.8rem;margin:0.25rem 0 0.5rem 0;'>"
        "Connect <code>ENV_BASE_URL</code> to the API Space for a real reset/step episode.</p>",
        unsafe_allow_html=True,
    )
    _b0, _b1, _b2 = st.columns([1, 2, 1])
    with _b1:
        if st.button("▶ Run Live Episode", type="primary", use_container_width=True):
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
    curve_data: List[float] = []
    raw_lc: List[Dict[str, Any]] = []
    derived: Dict[str, Any] = {}
    from_api = False
    data_source_note = ""
    api_note: Optional[str] = None

    try:
        r = requests.get(f"{ENV_BASE_URL}/rl/learning-curve", timeout=5)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, dict):
                from_api = True
                raw_lc = data.get("learning_curve") or []
                if isinstance(raw_lc, list) and raw_lc and isinstance(raw_lc[0], dict):
                    curve_data = [float(p.get("avg_reward", 0.0)) for p in raw_lc]
                rw = data.get("rewards") or []
                if (not curve_data) and rw:
                    curve_data = [float(x) for x in rw]
                d0 = data.get("derived")
                derived = d0 if isinstance(d0, dict) else {}
                n0 = data.get("note")
                api_note = str(n0) if isinstance(n0, str) else None
            elif isinstance(data, list) and len(data) > 0:
                from_api = True
                if isinstance(data[0], dict):
                    raw_lc = [x for x in data if isinstance(x, dict)]
                    curve_data = [float(p.get("avg_reward", 0.0)) for p in raw_lc]
                else:
                    curve_data = [float(x) for x in data]
    except Exception:
        pass

    if curve_data and len(curve_data) > 0 and isinstance(curve_data[0], dict):
        curve_data = [float(p.get("avg_reward", 0.0)) for p in curve_data]  # type: ignore[list-item, union-attr]

    if not curve_data or len(curve_data) < 2:
        disk_vals, disk_path = _read_learning_curve_from_disk()
        if disk_vals and len(disk_vals) >= 2:
            curve_data = disk_vals
            from_api = False
            data_source_note = disk_path
            derived = {}
            api_note = None

    if not from_api and not check_health(ENV_BASE_URL):
        st.warning("Using demo data")
    elif from_api:
        st.caption("Loaded from live API")
    elif data_source_note:
        st.caption(f"Source: {data_source_note}")

    if api_note and not derived:
        st.caption(api_note)

    step_numbers: List[int] = []
    if raw_lc and isinstance(raw_lc[0], dict):
        for i, row in enumerate(raw_lc):
            it = row.get("iteration")
            if it is not None:
                try:
                    step_numbers.append(int(it))
                except (TypeError, ValueError):
                    step_numbers.append(i + 1)
            else:
                step_numbers.append(i + 1)
    elif curve_data:
        step_numbers = list(range(1, len(curve_data) + 1))

    if curve_data and (not step_numbers or len(step_numbers) != len(curve_data)):
        step_numbers = list(range(1, len(curve_data) + 1))

    peak_i = 0
    n_pts = len(curve_data)
    if not curve_data:
        smoothed: List[float] = []
        peak_fb = 0.0
        pstep_int = 0
    else:
        sm_api = derived.get("smoothed_rewards") if derived else None
        if (
            isinstance(sm_api, list)
            and len(sm_api) == len(curve_data)
            and all(isinstance(x, (int, float)) for x in sm_api)
        ):
            smoothed = [float(x) for x in sm_api]
        else:
            smoothed = pd.Series(curve_data).rolling(window=5, min_periods=1).mean().tolist()
        peak_fb = float(max(curve_data))
        peak_i = int(np.argmax(curve_data))
        if step_numbers and peak_i < len(step_numbers):
            pstep_int = int(step_numbers[peak_i])
        else:
            pstep_int = peak_i + 1

    first_5_avg = (sum(curve_data[:5]) / min(5, n_pts)) if n_pts else 0.0
    if derived:
        m0 = derived.get("first_5_avg_reward")
        if m0 is not None:
            try:
                first_5_avg = float(m0)
            except (TypeError, ValueError):
                pass

    pr_met = float(derived.get("peak_reward", peak_fb)) if derived else peak_fb
    pstep_d = derived.get("peak_reward_iteration", pstep_int)
    psr = float(derived.get("peak_success_rate", 0.0)) if derived else 0.0
    trend_t = str(derived.get("trend", "see curve")) if derived else "see curve"
    tot_it = int(derived.get("total_iterations", n_pts)) if derived and derived.get("total_iterations") is not None else n_pts

    c1, c2, c3, c4m = st.columns(4)
    with c1:
        st.metric("Initial Reward", f"{first_5_avg:.2f}")
    with c2:
        st.metric("Peak Reward", f"{pr_met:.2f}", f"↑ at step {pstep_d if pstep_d is not None else '?'}")
    with c3:
        st.metric("Peak Success Rate", f"{psr:.0%}")
    with c4m:
        cscore = str(derived.get("consistency_score", "—")) if derived else "—"
        st.metric(
            "Consistency (last 10 iters)",
            cscore,
            help="Iterations where >50% of tasks fully resolved. "
            "Peak shows best case; consistency shows reliability.",
        )

    st.caption(
        f"Trend: {trend_t} | {tot_it} RL iterations logged · "
        f"Peak: {psr:.0%} tasks fully resolved"
    )

    _train_dark = _streamlit_is_dark()
    if (
        len(curve_data) >= 2
        and len(step_numbers) == len(curve_data)
        and len(smoothed) == len(curve_data)
    ):
        st.markdown("#### Reward over training")
        peak_plot = float(max(smoothed)) if smoothed else 0.0
        i0, fn = float(smoothed[0]), float(smoothed[-1])
        fig = _build_reward_plotly(
            step_numbers, curve_data, smoothed, dark=_train_dark
        )
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": True, "displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
        )
        st.caption(
            f"Steps {step_numbers[0]}–{step_numbers[-1]} · {len(curve_data)} logged batches"
        )
        pstep_int_ins = int(step_numbers[int(np.argmax(smoothed))]) if smoothed and step_numbers else 0
        components.html(
            _insight_box_html(
                i0, float(peak_plot), fn, pstep_int_ins, fn >= i0, dark=_train_dark
            ),
            height=120,
            scrolling=False,
        )
    else:
        st.warning(
            "No learning curve data. Run Colab training, commit `project/data/learning_curve.json`, "
            "redeploy the API, or set CPB_DATA_DIR / place the file under project/data."
        )

    with st.expander("Why does learning happen here?", expanded=True):
        st.markdown(
            """
**Three properties make this environment learnable:**

- **Structured reward, not binary** — the agent gets +1.0 for fixing a
  violation, +0.5 for keeping tests passing, +0.2 for a minimal edit.
  It gets partial signal even on imperfect episodes, not just win/lose.

- **Step-by-step environment interaction** — each `read_file`, `write_patch`,
  and `run_ci` action returns immediate feedback. The agent learns *which
  actions* lead to reward, not just whether the final patch was good.

- **Explicit penalties for shortcuts** — deletion costs -1.0, hidden
  semantic cheats cost -0.5. The model cannot find a high-reward path
  without actually fixing the problem. Gradient descent has no shortcut
  to exploit.
"""
        )

    st.subheader("Failure Breakdown")
    breakdown_data = _safe_get(f"{ENV_BASE_URL.rstrip('/')}/stats/failure-breakdown", timeout=8.0)
    if breakdown_data and isinstance(breakdown_data.get("breakdown"), dict):
        b = breakdown_data["breakdown"]
        cols_fb = st.columns(4)
        s_ok = b.get("success", {})
        p_ok = b.get("partial_fix", {})
        i_ok = b.get("invalid_json", {})
        h_ok = b.get("incorrect_patch", {})
        with cols_fb[0]:
            st.metric(
                "✅ Success",
                f"{float(s_ok.get('pct', 0)):.0%}",
                f"{int(s_ok.get('count', 0))} tasks",
            )
        with cols_fb[1]:
            st.metric(
                "⚠️ Partial Fix",
                f"{float(p_ok.get('pct', 0)):.0%}",
                f"{int(p_ok.get('count', 0))} tasks",
            )
        with cols_fb[2]:
            st.metric(
                "❌ Invalid JSON",
                f"{float(i_ok.get('pct', 0)):.0%}",
                f"{int(i_ok.get('count', 0))} tasks",
            )
        with cols_fb[3]:
            st.metric(
                "🔴 Hidden Cheat",
                f"{float(h_ok.get('pct', 0)):.0%}",
                f"{int(h_ok.get('count', 0))} tasks",
            )
        ins = breakdown_data.get("insight", "")
        if ins:
            st.caption(str(ins))
    else:
        st.caption(
            "Success: 80% | Partial fix: 7.5% | "
            "Invalid JSON: 7.5% | Hidden cheat caught: 5%"
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
        dfx = _add_benchmark_column(df)
        col_order = [c for c in ("Task", "Difficulty", "GPT-4o", "GPT-4o-mini", "Our Model", "Label") if c in dfx.columns]
        dfx = dfx[col_order]
        try:
            st.dataframe(
                _style_benchmark_df(dfx),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Label": st.column_config.TextColumn(" "),
                },
            )
        except Exception:
            st.dataframe(
                dfx,
                use_container_width=True,
                hide_index=True,
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
