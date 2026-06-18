from __future__ import annotations

import json
import os
import time
import uuid

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE = os.getenv("API_BASE_URL", "http://api:8000")
STREAM_ENDPOINT = f"{API_BASE}/api/v1/research/stream"

NODE_ICONS: dict[str, str] = {
    "planner": "🗺️",
    "retriever": "🔍",
    "summariser": "✍️",
    "critic": "🧐",
}

NODE_LABELS: dict[str, str] = {
    "planner": "Planner",
    "retriever": "Retriever",
    "summariser": "Summariser",
    "critic": "Critic",
}

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Research Copilot",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Custom CSS — dark-terminal aesthetic with amber accents
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    /* ── Global resets ──────────────────────────────── */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0d0f14;
        color: #e2e8f0;
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }
    [data-testid="stHeader"] { background: transparent; }

    /* ── Hero banner ─────────────────────────────────── */
    .hero {
        padding: 2.5rem 0 1.5rem 0;
        text-align: center;
    }
    .hero h1 {
        font-size: 2.6rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        background: linear-gradient(90deg, #f6a623, #f97316);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .hero p {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-top: 0;
    }

    /* ── Input card ──────────────────────────────────── */
    .query-card {
        background: #161b27;
        border: 1px solid #1e2535;
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.8rem;
    }

    /* ── Agent pipeline track ────────────────────────── */
    .pipeline-track {
        display: flex;
        gap: 0.7rem;
        margin-bottom: 1.4rem;
        flex-wrap: wrap;
    }
    .agent-pill {
        display: flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.35rem 0.85rem;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 500;
        border: 1px solid #2a3347;
        background: #161b27;
        color: #64748b;
        transition: all 0.25s;
    }
    .agent-pill.active {
        background: rgba(246, 166, 35, 0.12);
        border-color: #f6a623;
        color: #f6a623;
    }
    .agent-pill.done {
        background: rgba(34, 197, 94, 0.1);
        border-color: #22c55e;
        color: #22c55e;
    }

    /* ── Step feed ───────────────────────────────────── */
    .step-feed {
        background: #0a0c11;
        border: 1px solid #1e2535;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        max-height: 240px;
        overflow-y: auto;
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
        font-size: 0.8rem;
        color: #94a3b8;
        margin-bottom: 1.6rem;
    }
    .step-line { margin: 0.15rem 0; }
    .step-line.fresh { color: #f6a623; }

    /* ── Result card ─────────────────────────────────── */
    .result-card {
        background: #161b27;
        border: 1px solid #1e2535;
        border-radius: 12px;
        padding: 1.6rem 2rem;
    }
    .result-label {
        font-size: 0.72rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #475569;
        margin-bottom: 0.6rem;
    }

    /* ── Scrollbar ───────────────────────────────────── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0d0f14; }
    ::-webkit-scrollbar-thumb { background: #2a3347; border-radius: 3px; }

    /* ── Streamlit widget overrides ──────────────────── */
    div[data-testid="stTextInput"] input {
        background: #0d0f14 !important;
        border: 1px solid #2a3347 !important;
        color: #e2e8f0 !important;
        border-radius: 8px !important;
    }
    div[data-testid="stButton"] button {
        background: linear-gradient(90deg, #f6a623, #f97316) !important;
        color: #0d0f14 !important;
        border: none !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        padding: 0.5rem 1.4rem !important;
    }
    div[data-testid="stButton"] button:hover {
        filter: brightness(1.1) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <h1>🔬 Research Copilot</h1>
        <p>Multi-agent AI pipeline · Plan · Retrieve · Synthesise · Critique</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "history" not in st.session_state:
    st.session_state.history = []  # list of {query, steps, summary}

# ---------------------------------------------------------------------------
# Query input
# ---------------------------------------------------------------------------

st.markdown('<div class="query-card">', unsafe_allow_html=True)
query = st.text_input(
    label="Research question",
    placeholder="e.g. How does Retrieval-Augmented Generation improve LLM factuality?",
    label_visibility="collapsed",
)
run_btn = st.button("▶  Run Research", use_container_width=False)
st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Pipeline visualisation placeholders
# ---------------------------------------------------------------------------

pipeline_ph = st.empty()
step_feed_ph = st.empty()
result_ph = st.empty()
status_ph = st.empty()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _render_pipeline(active_node: str, done_nodes: list[str]) -> None:
    pills_html = '<div class="pipeline-track">'
    for key, label in NODE_LABELS.items():
        icon = NODE_ICONS[key]
        css_class = "agent-pill"
        if key == active_node:
            css_class += " active"
        elif key in done_nodes:
            css_class += " done"
        pills_html += f'<div class="{css_class}">{icon} {label}</div>'
    pills_html += "</div>"
    pipeline_ph.markdown(pills_html, unsafe_allow_html=True)


def _render_steps(steps: list[str], fresh_index: int = -1) -> None:
    lines_html = '<div class="step-feed">'
    for i, step in enumerate(steps):
        cls = "step-line fresh" if i == fresh_index else "step-line"
        escaped = step.replace("<", "&lt;").replace(">", "&gt;")
        lines_html += f'<div class="{cls}">› {escaped}</div>'
    lines_html += "</div>"
    step_feed_ph.markdown(lines_html, unsafe_allow_html=True)


def _render_result(summary: str) -> None:
    result_ph.markdown(
        f'<div class="result-card">'
        f'<div class="result-label">📄 Research Report</div>',
        unsafe_allow_html=True,
    )
    result_ph.markdown(summary)


# ---------------------------------------------------------------------------
# SSE stream consumer
# ---------------------------------------------------------------------------


def _run_stream(research_query: str) -> None:
    """
    Opens the SSE endpoint, parses events incrementally, and updates Streamlit
    widgets live as each agent node completes.
    """
    payload = {
        "query": research_query,
        "session_id": st.session_state.session_id,
    }

    active_node: str = ""
    done_nodes: list[str] = []
    accumulated_steps: list[str] = []
    final_summary: str = ""
    error_message: str = ""

    _render_pipeline(active_node, done_nodes)
    _render_steps([])

    try:
        with requests.post(
            STREAM_ENDPOINT,
            json=payload,
            stream=True,
            timeout=300,
            headers={"Accept": "text/event-stream"},
        ) as resp:
            resp.raise_for_status()

            event_type = ""
            data_lines: list[str] = []

            for raw_line in resp.iter_lines(decode_unicode=True):
                if raw_line.startswith("event:"):
                    event_type = raw_line[len("event:"):].strip()
                    data_lines = []
                elif raw_line.startswith("data:"):
                    data_lines.append(raw_line[len("data:"):].strip())
                elif raw_line == "":
                    # End of an SSE frame — process it
                    if not event_type:
                        event_type = ""
                        data_lines = []
                        continue

                    raw_data = " ".join(data_lines)
                    try:
                        payload_data: dict = json.loads(raw_data)
                    except json.JSONDecodeError:
                        payload_data = {}

                    if event_type == "node_start":
                        active_node = payload_data.get("node", "")
                        _render_pipeline(active_node, done_nodes)

                    elif event_type == "step":
                        node = payload_data.get("node", "")
                        message = payload_data.get("message", "")
                        if message:
                            accumulated_steps.append(message)
                        _render_steps(accumulated_steps, fresh_index=len(accumulated_steps) - 1)
                        if node and node not in done_nodes:
                            done_nodes.append(node)
                        _render_pipeline(active_node, done_nodes)

                    elif event_type == "result":
                        final_summary = payload_data.get("summary", "")
                        extra_steps: list[str] = payload_data.get("steps", [])
                        # Merge any steps we may have missed
                        for s in extra_steps:
                            if s not in accumulated_steps:
                                accumulated_steps.append(s)
                        _render_steps(accumulated_steps)
                        _render_result(final_summary)

                    elif event_type == "error":
                        error_message = payload_data.get("detail", "Unknown error.")
                        status_ph.error(f"⚠️ Stream error: {error_message}")

                    elif event_type == "done":
                        _render_pipeline("", done_nodes)
                        if not error_message:
                            status_ph.success("✅ Research complete.")
                        break

                    # Reset for next frame
                    event_type = ""
                    data_lines = []

    except requests.exceptions.ConnectionError:
        status_ph.error(
            "🔌 Cannot reach the API service. "
            "Make sure the backend container is running on `http://api:8000`."
        )
        return
    except requests.exceptions.Timeout:
        status_ph.error("⏱️ Request timed out. The model may be loading — please retry.")
        return
    except Exception as exc:  # noqa: BLE001
        status_ph.error(f"Unexpected error: {exc}")
        return

    # Persist to session history
    if final_summary:
        st.session_state.history.append(
            {
                "query": research_query,
                "steps": accumulated_steps,
                "summary": final_summary,
            }
        )


# ---------------------------------------------------------------------------
# Trigger
# ---------------------------------------------------------------------------

if run_btn and query.strip():
    status_ph.info("⏳ Connecting to the Research Copilot pipeline…")
    time.sleep(0.3)
    status_ph.empty()
    _run_stream(query.strip())
elif run_btn:
    status_ph.warning("Please enter a research question first.")

# ---------------------------------------------------------------------------
# History accordion
# ---------------------------------------------------------------------------

if st.session_state.history:
    st.markdown("---")
    st.markdown("#### 📚 Previous Sessions")
    for i, entry in enumerate(reversed(st.session_state.history)):
        with st.expander(f"🔎 {entry['query'][:80]}…" if len(entry["query"]) > 80 else f"🔎 {entry['query']}"):
            st.markdown("**Steps taken:**")
            for step in entry["steps"]:
                st.caption(f"› {step}")
            st.markdown("---")
            st.markdown(entry["summary"])