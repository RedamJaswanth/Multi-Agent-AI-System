"""
workflow.py
-----------
LangGraph workflow: wires the four agents into a conditional graph.

Routing logic
─────────────
coordinator → (needs_research?) ──yes──► research → analysis → writer → END
                                 └──no──► analysis → writer → END

Every path terminates at the writer node which then transitions to END.
The recursion limit is set to prevent infinite loops.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from agents import (
    analysis_agent,
    coordinator_agent,
    research_agent,
    writer_agent,
)
from config import MAX_RECURSION_LIMIT
from state import AgentState

logger = logging.getLogger(__name__)


# ── Conditional edge: after coordinator, route to research or analysis ─────────

def route_after_coordinator(
    state: AgentState,
) -> Literal["research_agent", "analysis_agent"]:
    """
    If the coordinator flagged needs_research=True, go to research.
    Otherwise skip straight to analysis.
    Also short-circuit to writer if a fatal error was raised.
    """
    if state.get("error"):
        # Fatal error: skip all remaining agents, go directly to writer
        return "analysis_agent"   # analysis will pass-through; writer catches error

    if state.get("needs_research", False):
        logger.info("[Router] → research_agent")
        return "research_agent"

    logger.info("[Router] → analysis_agent (no research needed)")
    return "analysis_agent"


# ── Build and compile the graph ────────────────────────────────────────────────

def build_graph() -> Any:
    """
    Construct, compile, and return the LangGraph StateGraph.

    Returns
    -------
    CompiledStateGraph
        A compiled graph ready to be invoked or streamed.
    """
    builder = StateGraph(AgentState)

    # Register all nodes
    builder.add_node("coordinator_agent", coordinator_agent)
    builder.add_node("research_agent", research_agent)
    builder.add_node("analysis_agent", analysis_agent)
    builder.add_node("writer_agent", writer_agent)

    # Entry point
    builder.add_edge(START, "coordinator_agent")

    # Conditional routing after coordinator
    builder.add_conditional_edges(
        "coordinator_agent",
        route_after_coordinator,
        {
            "research_agent": "research_agent",
            "analysis_agent": "analysis_agent",
        },
    )

    # After research → analysis
    builder.add_edge("research_agent", "analysis_agent")

    # After analysis → writer
    builder.add_edge("analysis_agent", "writer_agent")

    # Writer is always the terminal node
    builder.add_edge("writer_agent", END)

    graph = builder.compile()
    logger.info("LangGraph workflow compiled successfully.")
    return graph


# ── Public run helpers ─────────────────────────────────────────────────────────

def run_workflow(
    user_task: str,
    conversation_history: list[dict[str, Any]] | None = None,
) -> AgentState:
    """
    Execute the full workflow synchronously and return the final state.

    Parameters
    ----------
    user_task : str
        The task entered by the user.
    conversation_history : list[dict], optional
        Prior conversation turns for context.

    Returns
    -------
    AgentState
        The final merged state after all agents have run.
    """
    graph = build_graph()
    initial_state: AgentState = {
        "user_task": user_task.strip(),
        "conversation_history": conversation_history or [],
        "coordinator_decision": "",
        "needs_research": False,
        "research_findings": "",
        "source_urls": [],
        "analysis": "",
        "final_response": "",
        "error": None,
        "warnings": [],
        "current_agent": "starting",
    }
    config = {"recursion_limit": MAX_RECURSION_LIMIT}
    final_state: AgentState = graph.invoke(initial_state, config=config)
    return final_state


def stream_workflow_events(
    user_task: str,
    conversation_history: list[dict[str, Any]] | None = None,
):
    """
    Stream workflow node outputs as they complete.

    Yields
    ------
    tuple[str, AgentState]
        (node_name, partial_state) for each completed node.
    """
    graph = build_graph()
    initial_state: AgentState = {
        "user_task": user_task.strip(),
        "conversation_history": conversation_history or [],
        "coordinator_decision": "",
        "needs_research": False,
        "research_findings": "",
        "source_urls": [],
        "analysis": "",
        "final_response": "",
        "error": None,
        "warnings": [],
        "current_agent": "starting",
    }
    config = {"recursion_limit": MAX_RECURSION_LIMIT}

    for event in graph.stream(initial_state, config=config, stream_mode="updates"):
        for node_name, partial_state in event.items():
            logger.debug("[Stream] Node '%s' emitted update", node_name)
            yield node_name, partial_state
