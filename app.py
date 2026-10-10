"""
app.py
------
Streamlit frontend for the AI Multi-Agent Research and Analysis System.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import time
import logging
from typing import Any

import streamlit as st

# ── Page config (must be the very first Streamlit call) ───────────────────────
st.set_page_config(
    page_title="AI Multi-Agent Research & Analysis",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

from config import APP_TITLE, GOOGLE_API_KEY                  # noqa: E402
from memory import append_turn, clear_history                  # noqa: E402
from workflow import stream_workflow_events                     # noqa: E402

logger = logging.getLogger(__name__)

# ── Sample tasks ──────────────────────────────────────────────────────────────
SAMPLE_TASKS = [
    "What is LangGraph and how does it differ from LangChain?",
    "What are the latest developments in quantum computing in 2025?",
    "Compare Python and Rust for systems programming with concrete examples.",
    "Explain the CAP theorem with a real-world distributed systems example.",
    "What is the current stock price and recent news about NVIDIA?",
    "Summarise the key ideas behind transformer neural network architecture.",
    "What are the best practices for securing a REST API?",
    "Compare the top large language models available today.",
]

# ── CSS ───────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
/* ── Global ── */
body { font-family: 'Inter', sans-serif; }

/* ── Sidebar ── */
[data-testid="stSidebar"] { background: #0f172a; }
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #7c3aed !important; }

/* ── Agent status cards ── */
.agent-card {
    border-radius: 10px;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 14px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.agent-pending  { background: #1e293b; border-left: 3px solid #475569; color: #94a3b8; }
.agent-running  { background: #1e3a5f; border-left: 3px solid #3b82f6; color: #93c5fd;
                  animation: pulse 1.5s ease-in-out infinite; }
.agent-done     { background: #14532d; border-left: 3px solid #22c55e; color: #86efac; }
.agent-skipped  { background: #1c1917; border-left: 3px solid #78716c; color: #78716c; }
.agent-error    { background: #450a0a; border-left: 3px solid #ef4444; color: #fca5a5; }

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.65; }
}

/* ── Metric strip ── */
.metric-strip {
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    margin: 12px 0;
}
.metric-chip {
    background: #1e293b;
    border-radius: 8px;
    padding: 6px 14px;
    font-size: 13px;
    color: #cbd5e1;
}
.metric-chip strong { color: #a78bfa; }

/* ── Final response card ── */
.response-card {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 24px 28px;
    margin-top: 12px;
}
</style>
"""

# ── Session state initialisation ──────────────────────────────────────────────

def _init_session() -> None:
    defaults: dict[str, Any] = {
        "conversation_history": [],
        "last_result": None,
        "is_running": False,
        "agent_statuses": {},       # node_name → "pending" | "running" | "done" | "skipped" | "error"
        "activity_log": [],         # list of str activity messages
        "run_count": 0,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ── Agent display helpers ─────────────────────────────────────────────────────

AGENT_META: dict[str, dict[str, str]] = {
    "coordinator_agent": {"icon": "🧭", "label": "Coordinator Agent", "desc": "Classifying task & deciding route"},
    "research_agent":    {"icon": "🔍", "label": "Research Agent",    "desc": "Searching the web for evidence"},
    "analysis_agent":    {"icon": "🧠", "label": "Analysis Agent",    "desc": "Extracting insights & patterns"},
    "writer_agent":      {"icon": "✍️",  "label": "Writer Agent",      "desc": "Composing the final answer"},
}

STATUS_LABELS: dict[str, str] = {
    "pending": "⏳ Pending",
    "running": "⚡ Running",
    "done":    "✅ Done",
    "skipped": "⏭️ Skipped",
    "error":   "❌ Error",
}


def _render_agent_card(node: str, status: str) -> str:
    meta = AGENT_META.get(node, {"icon": "🤖", "label": node, "desc": ""})
    css_class = f"agent-{status}"
    label = STATUS_LABELS.get(status, status)
    return (
        f'<div class="agent-card {css_class}">'
        f'<span style="font-size:20px">{meta["icon"]}</span>'
        f'<div><strong>{meta["label"]}</strong>&nbsp;&nbsp;{label}<br>'
        f'<small>{meta["desc"]}</small></div>'
        f'</div>'
    )


# ── Main app ──────────────────────────────────────────────────────────────────

def main() -> None:
    _init_session()
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <h1 style="color:#7c3aed;margin-bottom:4px;">🤖 {APP_TITLE}</h1>
        <p style="color:#94a3b8;font-size:15px;margin-top:0">
        A production-style multi-agent AI system powered by Google Gemini, LangGraph,
        and DuckDuckGo Search. The coordinator routes your task intelligently —
        only invoking web research when genuinely required.
        </p>
        """,
        unsafe_allow_html=True,
    )

    # ── API key guard ─────────────────────────────────────────────────────────
    if not GOOGLE_API_KEY:
        st.error(
            "**🔑 GOOGLE_API_KEY not found.**\n\n"
            "1. Copy `.env.example` → `.env`\n"
            "2. Paste your Gemini API key (get one free at https://aistudio.google.com/app/apikey)\n"
            "3. Restart the app with `streamlit run app.py`"
        )
        st.stop()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🤖 Agent Status")
        status_container = st.container()

        st.markdown("---")
        st.markdown("## 📋 Sample Tasks")
        for sample in SAMPLE_TASKS:
            if st.button(sample[:60] + ("…" if len(sample) > 60 else ""), key=f"sample_{sample[:20]}"):
                st.session_state["prefill_task"] = sample
                st.rerun()

        st.markdown("---")
        st.markdown("## 🗑️ Conversation")
        if st.button("Clear Conversation", use_container_width=True):
            st.session_state["conversation_history"] = clear_history()
            st.session_state["last_result"] = None
            st.session_state["activity_log"] = []
            st.session_state["agent_statuses"] = {}
            st.rerun()

        st.markdown("---")
        st.markdown(
            f"**Turns:** {len(st.session_state['conversation_history']) // 2}  \n"
            f"**Runs:** {st.session_state['run_count']}"
        )

    # ── Render agent status in sidebar ───────────────────────────────────────
    with status_container:
        agent_statuses = st.session_state.get("agent_statuses", {})
        for node in AGENT_META:
            status = agent_statuses.get(node, "pending")
            st.markdown(_render_agent_card(node, status), unsafe_allow_html=True)

    # ── Main content ──────────────────────────────────────────────────────────
    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        # Task input
        prefill = st.session_state.pop("prefill_task", "")
        user_task = st.text_area(
            "📝 Enter your task or question",
            value=prefill,
            height=120,
            placeholder="e.g. What are the latest advances in fusion energy research?",
            help="Describe your task clearly. The system will decide whether web research is needed.",
        )

        run_clicked = st.button(
            "🚀 Run Analysis",
            use_container_width=True,
            disabled=st.session_state["is_running"],
            type="primary",
        )

        # ── Activity log ──────────────────────────────────────────────────────
        if st.session_state["activity_log"]:
            with st.expander("📡 Workflow Activity Log", expanded=True):
                for msg in st.session_state["activity_log"]:
                    st.markdown(msg)

    with col_right:
        # Conversation history
        history = st.session_state.get("conversation_history", [])
        if history:
            with st.expander(f"💬 Conversation History ({len(history) // 2} turns)", expanded=False):
                for turn in history:
                    role = turn["role"].capitalize()
                    icon = "🧑" if turn["role"] == "user" else "🤖"
                    content = turn["content"]
                    if len(content) > 300:
                        content = content[:300] + "…"
                    st.markdown(f"**{icon} {role}:** {content}")
                    st.markdown("---")

    # ── Workflow execution ────────────────────────────────────────────────────
    if run_clicked:
        task = user_task.strip()
        if not task:
            st.warning("⚠️ Please enter a task before clicking Run Analysis.")
        else:
            st.session_state["is_running"] = True
            st.session_state["activity_log"] = []
            st.session_state["agent_statuses"] = {n: "pending" for n in AGENT_META}
            st.session_state["last_result"] = None
            st.session_state["run_count"] += 1

            # Accumulated state across streamed updates
            accumulated: dict[str, Any] = {
                "coordinator_decision": "",
                "needs_research": False,
                "research_findings": "",
                "source_urls": [],
                "analysis": "",
                "final_response": "",
                "error": None,
                "warnings": [],
                "conversation_history": st.session_state["conversation_history"],
            }

            # Placeholders for live updates
            status_placeholder = st.empty()
            progress_bar = st.progress(0)

            node_order = list(AGENT_META.keys())
            node_progress = {n: i / len(node_order) for i, n in enumerate(node_order)}
            node_progress["writer_agent"] = 1.0

            try:
                for node_name, partial_state in stream_workflow_events(
                    task,
                    conversation_history=st.session_state["conversation_history"],
                ):
                    # Merge partial update into accumulated state
                    accumulated.update(partial_state)

                    # Update agent status
                    statuses = st.session_state["agent_statuses"]
                    # Mark previous nodes done
                    seen = False
                    for n in node_order:
                        if n == node_name:
                            seen = True
                            statuses[n] = "error" if accumulated.get("error") else "done"
                        elif not seen:
                            if statuses.get(n) == "pending":
                                statuses[n] = "skipped"

                    # Check if research was skipped
                    if node_name == "analysis_agent" and not accumulated.get("needs_research"):
                        statuses["research_agent"] = "skipped"

                    st.session_state["agent_statuses"] = statuses

                    # Activity log entry
                    meta = AGENT_META.get(node_name, {})
                    icon = meta.get("icon", "🤖")
                    label = meta.get("label", node_name)

                    if node_name == "coordinator_agent":
                        decision = accumulated.get("coordinator_decision", "?")
                        needs = accumulated.get("needs_research", False)
                        st.session_state["activity_log"].append(
                            f"{icon} **{label}** — Decision: `{decision}` | Research needed: `{needs}`"
                        )
                    elif node_name == "research_agent":
                        n_sources = len(accumulated.get("source_urls", []))
                        st.session_state["activity_log"].append(
                            f"{icon} **{label}** — Found {n_sources} source(s)"
                        )
                    elif node_name == "analysis_agent":
                        chars = len(accumulated.get("analysis", ""))
                        st.session_state["activity_log"].append(
                            f"{icon} **{label}** — Analysis ready ({chars} chars)"
                        )
                    elif node_name == "writer_agent":
                        chars = len(accumulated.get("final_response", ""))
                        st.session_state["activity_log"].append(
                            f"{icon} **{label}** — Final response ready ({chars} chars)"
                        )

                    # Update progress bar
                    progress = node_progress.get(node_name, 0)
                    progress_bar.progress(min(int(progress * 100), 100))

                    # Warnings
                    for w in accumulated.get("warnings", []):
                        st.session_state["activity_log"].append(f"⚠️ {w}")

                    # Force sidebar re-render (Streamlit reruns on button click,
                    # but we need intra-loop updates via the placeholder)
                    with status_placeholder.container():
                        for n in node_order:
                            s = st.session_state["agent_statuses"].get(n, "pending")
                            st.markdown(_render_agent_card(n, s), unsafe_allow_html=True)

                # Workflow complete
                progress_bar.progress(100)
                st.session_state["last_result"] = accumulated
                # Persist conversation history
                st.session_state["conversation_history"] = accumulated.get(
                    "conversation_history",
                    st.session_state["conversation_history"],
                )

                # Mark all remaining nodes done/skipped
                final_statuses = st.session_state["agent_statuses"]
                for n in node_order:
                    if final_statuses.get(n) == "pending":
                        final_statuses[n] = "skipped"
                st.session_state["agent_statuses"] = final_statuses

            except Exception as exc:
                logger.exception("Workflow execution failed: %s", exc)
                st.session_state["activity_log"].append(f"❌ **Workflow error:** {exc}")
                st.session_state["last_result"] = {
                    "error": str(exc),
                    "final_response": f"⚠️ **Workflow failed:** {exc}",
                }
            finally:
                st.session_state["is_running"] = False
                time.sleep(0.1)
                st.rerun()

    # ── Display last result ───────────────────────────────────────────────────
    result = st.session_state.get("last_result")
    if result:
        st.markdown("---")

        # Error banner
        if result.get("error"):
            st.error(f"**Error:** {result['error']}")

        # Metric strip
        decision = result.get("coordinator_decision", "—")
        researched = "Yes" if result.get("source_urls") else "No"
        n_sources = len(result.get("source_urls", []))
        resp_len = len(result.get("final_response", ""))

        st.markdown(
            f"""
            <div class="metric-strip">
                <div class="metric-chip">Route: <strong>{decision}</strong></div>
                <div class="metric-chip">Web Research: <strong>{researched}</strong></div>
                <div class="metric-chip">Sources: <strong>{n_sources}</strong></div>
                <div class="metric-chip">Response: <strong>{resp_len} chars</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Final response
        if result.get("final_response"):
            st.markdown("### 📄 Final Response")
            st.markdown(
                '<div class="response-card">',
                unsafe_allow_html=True,
            )
            st.markdown(result["final_response"])
            st.markdown("</div>", unsafe_allow_html=True)

        # Research expandable
        if result.get("research_findings") and result.get("source_urls"):
            with st.expander("🔍 Research Findings", expanded=False):
                st.markdown(result["research_findings"])

        if result.get("source_urls"):
            with st.expander(f"🔗 Sources ({len(result['source_urls'])})", expanded=False):
                for i, url in enumerate(result["source_urls"], 1):
                    st.markdown(f"{i}. [{url}]({url})")

        # Analysis expandable
        if result.get("analysis"):
            with st.expander("🧠 Analysis Details", expanded=False):
                st.markdown(result["analysis"])

        # Warnings
        if result.get("warnings"):
            with st.expander("⚠️ Warnings", expanded=False):
                for w in result["warnings"]:
                    st.warning(w)


if __name__ == "__main__":
    main()
