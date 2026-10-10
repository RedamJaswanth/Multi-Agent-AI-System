"""
state.py
--------
Shared TypedDict state schema for the LangGraph multi-agent workflow.
All agents read from and write to this shared structure.
"""

from __future__ import annotations

from typing import Any, Optional
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    """
    Shared state passed between every node in the LangGraph workflow.

    Fields
    ------
    user_task : str
        The original task submitted by the user.
    coordinator_decision : str
        One of: "RESEARCH", "ANALYSIS", "RESEARCH_AND_ANALYSIS".
    needs_research : bool
        True when the coordinator decides web research is required.
    research_findings : str
        Textual summary of the web research results.
    source_urls : list[str]
        Deduplicated list of source URLs gathered during research.
    analysis : str
        In-depth analysis produced by the Analysis Agent.
    final_response : str
        Polished Markdown answer produced by the Writer Agent.
    error : str | None
        Human-readable error message, or None when everything is fine.
    warnings : list[str]
        Non-fatal warnings (e.g. partial search failures).
    conversation_history : list[dict[str, str]]
        Ordered list of {"role": …, "content": …} conversation turns.
    current_agent : str
        Name of the agent node currently executing (used for UI status).
    """

    user_task: str
    coordinator_decision: str
    needs_research: bool
    research_findings: str
    source_urls: list[str]
    analysis: str
    final_response: str
    error: Optional[str]
    warnings: list[str]
    conversation_history: list[dict[str, Any]]
    current_agent: str
