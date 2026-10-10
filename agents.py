"""
agents.py
---------
Four distinct agent node functions for the LangGraph workflow:

  1. coordinator_agent  – classifies the task and decides routing
  2. research_agent     – gathers web evidence via DuckDuckGo
  3. analysis_agent     – extracts insights from the task + research
  4. writer_agent       – produces the polished final Markdown answer

Each function accepts an AgentState dict and returns a *partial* AgentState
dict (LangGraph merges these into the shared state automatically).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from config import (
    GOOGLE_API_KEY,
    GEMINI_MODEL,
    MODEL_TEMPERATURE,
    MODEL_MAX_TOKENS,
)
from memory import format_history_for_prompt
from state import AgentState
from tools import (
    extract_sources,
    filter_results,
    format_search_results,
    web_search,
)

logger = logging.getLogger(__name__)

# ── Shared model factory ──────────────────────────────────────────────────────

def _get_llm() -> ChatGoogleGenerativeAI:
    """Instantiate the Gemini model (raises if key is missing)."""
    if not GOOGLE_API_KEY:
        raise EnvironmentError(
            "GOOGLE_API_KEY is not set. "
            "Please create a .env file following .env.example."
        )
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=MODEL_TEMPERATURE,
        max_output_tokens=MODEL_MAX_TOKENS,
    )


def _call_llm(llm: ChatGoogleGenerativeAI, system: str, human: str) -> str:
    """Send a system + human prompt to the model and return the text reply.

    langchain-google-genai 4.x may return response.content as a list of
    content blocks instead of a plain string.  We normalise both cases here.
    """
    messages = [SystemMessage(content=system), HumanMessage(content=human)]
    response = llm.invoke(messages)
    content = response.content

    # Newer versions return a list of dicts/strings; flatten to plain text
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(block.get("text", ""))
        content = "".join(parts)

    return str(content).strip()


# ── Agent 1: Coordinator / Router ─────────────────────────────────────────────

_COORDINATOR_SYSTEM = """You are the Coordinator Agent for an AI research and analysis system.

Your job is to classify a user task and decide the correct processing strategy.

Respond with ONLY a valid JSON object — no markdown fences, no extra text — in this exact format:
{
  "decision": "<RESEARCH | ANALYSIS | RESEARCH_AND_ANALYSIS>",
  "needs_research": <true | false>,
  "reasoning": "<one sentence>"
}

Decision rules:
- ANALYSIS        → The task can be answered from general knowledge alone
                    (definitions, explanations, coding help, math, opinions).
- RESEARCH        → The task explicitly requires up-to-date facts, news,
                    prices, recent events, or real-time information.
- RESEARCH_AND_ANALYSIS → The task requires both web research AND deeper
                    analytical thinking (compare options, evaluate evidence,
                    produce a recommendation based on current data).

Be conservative: only choose RESEARCH or RESEARCH_AND_ANALYSIS when web
information is genuinely necessary. Most factual or conceptual questions
do not need web search.
"""


def coordinator_agent(state: AgentState) -> dict[str, Any]:
    """Classify the user task and decide the routing strategy."""
    logger.info("[Coordinator] Starting")
    user_task = state.get("user_task", "").strip()

    if not user_task:
        return {
            "current_agent": "coordinator",
            "coordinator_decision": "ANALYSIS",
            "needs_research": False,
            "error": "No task provided. Please enter a task.",
        }

    history_ctx = format_history_for_prompt(state.get("conversation_history", []))
    human_msg = f"{history_ctx}\n\nUser Task:\n{user_task}" if history_ctx else f"User Task:\n{user_task}"

    try:
        llm = _get_llm()
        raw = _call_llm(llm, _COORDINATOR_SYSTEM, human_msg)
        logger.debug("[Coordinator] Raw response: %s", raw)

        # Strip accidental markdown fences
        cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
        parsed = json.loads(cleaned)

        decision = parsed.get("decision", "ANALYSIS").upper()
        if decision not in {"RESEARCH", "ANALYSIS", "RESEARCH_AND_ANALYSIS"}:
            logger.warning("[Coordinator] Unknown decision '%s', defaulting to ANALYSIS", decision)
            decision = "ANALYSIS"

        needs_research = bool(parsed.get("needs_research", False))

        logger.info("[Coordinator] Decision=%s needs_research=%s", decision, needs_research)
        return {
            "current_agent": "coordinator",
            "coordinator_decision": decision,
            "needs_research": needs_research,
            "error": None,
            "warnings": [],
        }

    except EnvironmentError as exc:
        return {"current_agent": "coordinator", "error": str(exc)}
    except (json.JSONDecodeError, KeyError) as exc:
        logger.warning("[Coordinator] Could not parse JSON response: %s", exc)
        return {
            "current_agent": "coordinator",
            "coordinator_decision": "ANALYSIS",
            "needs_research": False,
            "warnings": ["Coordinator could not parse model output; defaulting to ANALYSIS."],
            "error": None,
        }
    except Exception as exc:
        logger.exception("[Coordinator] Unexpected error: %s", exc)
        return {"current_agent": "coordinator", "error": f"Coordinator error: {exc}"}


# ── Agent 2: Research Agent ───────────────────────────────────────────────────

_RESEARCH_SYSTEM = """You are the Research Agent in an AI research system.

You have been given a set of web search results related to the user's task.
Your job is to:
1. Review the search results critically.
2. Remove results that are clearly off-topic or low-quality.
3. Write a comprehensive, factual summary of the relevant evidence.
4. Preserve ALL source URLs — never invent URLs or claim a search succeeded
   when no results were found.

Format your summary in clear paragraphs. Do not use markdown headers.
Begin with a one-sentence overview, then expand with key findings.
"""


def research_agent(state: AgentState) -> dict[str, Any]:
    """Search the web and summarise findings for the user's task."""
    logger.info("[Research] Starting")
    user_task = state.get("user_task", "").strip()

    if not user_task:
        return {
            "current_agent": "research",
            "research_findings": "No task provided — skipping research.",
            "source_urls": [],
            "warnings": ["Research skipped: empty task."],
        }

    # Run DuckDuckGo search
    raw_results = web_search(user_task)
    filtered = filter_results(raw_results)
    sources = extract_sources(filtered)
    formatted = format_search_results(filtered)

    warnings: list[str] = []

    if not filtered:
        if raw_results:
            warnings.append("Web search returned results but all were filtered as irrelevant.")
        else:
            warnings.append("Web search returned no results. The analysis will proceed without web data.")
        return {
            "current_agent": "research",
            "research_findings": "No relevant web results were found for this task.",
            "source_urls": [],
            "warnings": warnings,
        }

    # Ask the model to summarise
    human_msg = (
        f"User Task: {user_task}\n\n"
        f"Search Results:\n{formatted}\n\n"
        "Please summarise the relevant research findings."
    )

    try:
        llm = _get_llm()
        summary = _call_llm(llm, _RESEARCH_SYSTEM, human_msg)
        logger.info("[Research] Summary produced (%d chars), %d sources", len(summary), len(sources))
        return {
            "current_agent": "research",
            "research_findings": summary,
            "source_urls": sources,
            "warnings": warnings,
        }
    except EnvironmentError as exc:
        return {"current_agent": "research", "error": str(exc)}
    except Exception as exc:
        logger.exception("[Research] Error during summarisation: %s", exc)
        return {
            "current_agent": "research",
            "research_findings": formatted,   # fall back to raw formatted results
            "source_urls": sources,
            "warnings": warnings + [f"Research summarisation failed: {exc}"],
        }


# ── Agent 3: Analysis Agent ───────────────────────────────────────────────────

_ANALYSIS_SYSTEM = """You are the Analysis Agent in an AI research system.

Your job is to analyse the user's task and any available research, then extract:
- Key facts and data points (clearly marked as sourced when from research)
- Patterns, trends, and relationships
- Important insights and observations
- Logical conclusions or answers
- Assumptions made when information is unavailable

Be intellectually rigorous. Clearly distinguish what is known from research,
what is general knowledge, and what is inferred or assumed.

Write in well-structured paragraphs. Avoid bullet-point lists only — mix prose
with structured points where appropriate. Aim for analytical depth.
"""


def analysis_agent(state: AgentState) -> dict[str, Any]:
    """Analyse the task and research findings to extract structured insights."""
    logger.info("[Analysis] Starting")
    user_task = state.get("user_task", "").strip()
    research = state.get("research_findings", "")
    sources = state.get("source_urls", [])

    research_section = ""
    if research and research != "No relevant web results were found for this task.":
        research_section = f"\nResearch Findings:\n{research}"
        if sources:
            research_section += f"\n\nSources: {', '.join(sources[:5])}"

    human_msg = (
        f"User Task: {user_task}"
        f"{research_section}\n\n"
        "Please analyse the task and produce a detailed analysis."
    )

    try:
        llm = _get_llm()
        analysis = _call_llm(llm, _ANALYSIS_SYSTEM, human_msg)
        logger.info("[Analysis] Analysis produced (%d chars)", len(analysis))
        return {"current_agent": "analysis", "analysis": analysis}
    except EnvironmentError as exc:
        return {"current_agent": "analysis", "error": str(exc)}
    except Exception as exc:
        logger.exception("[Analysis] Unexpected error: %s", exc)
        return {
            "current_agent": "analysis",
            "analysis": f"Analysis could not be completed: {exc}",
            "warnings": (state.get("warnings") or []) + [f"Analysis error: {exc}"],
        }


# ── Agent 4: Writer Agent ─────────────────────────────────────────────────────

_WRITER_SYSTEM = """You are the Writer Agent in an AI research and analysis system.

Your job is to produce the final answer for the user using:
- The original user task
- The analysis produced by the Analysis Agent
- The research findings (when available)

Requirements:
- Write in clear, professional Markdown (headings, bullet points, bold text).
- Lead with a concise direct answer or executive summary.
- Expand with well-organised supporting detail.
- Cite sources inline when making research-backed claims, e.g. ([Source](url)).
- Do NOT invent sources. Only cite URLs actually provided.
- Do NOT repeat the same information multiple times.
- End with a brief conclusion or next-steps section when appropriate.
- Avoid jargon. Write for a technically literate but general audience.
"""


def writer_agent(state: AgentState) -> dict[str, Any]:
    """Generate the final polished Markdown answer."""
    logger.info("[Writer] Starting")
    user_task = state.get("user_task", "").strip()
    analysis = state.get("analysis", "")
    research = state.get("research_findings", "")
    sources = state.get("source_urls", [])

    # Check for upstream errors
    if state.get("error"):
        return {
            "current_agent": "writer",
            "final_response": (
                f"⚠️ **An error occurred before writing could begin:**\n\n"
                f"{state['error']}"
            ),
        }

    research_section = ""
    if research and research != "No relevant web results were found for this task.":
        research_section = f"\nResearch Summary:\n{research}"
        if sources:
            source_list = "\n".join(f"- {u}" for u in sources[:10])
            research_section += f"\n\nAvailable Sources:\n{source_list}"

    human_msg = (
        f"User Task: {user_task}\n\n"
        f"Analysis:\n{analysis}"
        f"{research_section}\n\n"
        "Please write the final comprehensive answer in Markdown."
    )

    try:
        llm = _get_llm()
        response = _call_llm(llm, _WRITER_SYSTEM, human_msg)
        logger.info("[Writer] Final response produced (%d chars)", len(response))

        # Update conversation history
        history = state.get("conversation_history", [])
        history = history + [
            {"role": "user", "content": user_task},
            {"role": "assistant", "content": response},
        ]

        return {
            "current_agent": "writer",
            "final_response": response,
            "conversation_history": history,
        }
    except EnvironmentError as exc:
        return {"current_agent": "writer", "error": str(exc)}
    except Exception as exc:
        logger.exception("[Writer] Unexpected error: %s", exc)
        return {
            "current_agent": "writer",
            "final_response": (
                f"⚠️ **Writing failed due to an unexpected error:**\n\n`{exc}`\n\n"
                "Please try again or simplify your task."
            ),
        }
