"""
tests/test_agents.py
--------------------
Unit tests for all four agent functions.

All LLM calls and DuckDuckGo searches are mocked so tests run without
consuming API quota and without network access.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_llm_response(text: str) -> MagicMock:
    """Return a MagicMock that looks like a LangChain ChatMessage."""
    mock = MagicMock()
    mock.content = text
    return mock


COORDINATOR_RESEARCH_RESPONSE = json.dumps({
    "decision": "RESEARCH",
    "needs_research": True,
    "reasoning": "Task requires current web data.",
})

COORDINATOR_ANALYSIS_RESPONSE = json.dumps({
    "decision": "ANALYSIS",
    "needs_research": False,
    "reasoning": "Task can be answered from general knowledge.",
})

COORDINATOR_BOTH_RESPONSE = json.dumps({
    "decision": "RESEARCH_AND_ANALYSIS",
    "needs_research": True,
    "reasoning": "Needs both research and deep analysis.",
})


# ── Coordinator Agent Tests ───────────────────────────────────────────────────

class TestCoordinatorAgent:
    @patch("agents._get_llm")
    def test_analysis_decision(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _make_llm_response(COORDINATOR_ANALYSIS_RESPONSE)
        mock_get_llm.return_value = mock_llm

        from agents import coordinator_agent
        result = coordinator_agent({"user_task": "Explain what a binary tree is."})

        assert result["coordinator_decision"] == "ANALYSIS"
        assert result["needs_research"] is False
        assert result.get("error") is None

    @patch("agents._get_llm")
    def test_research_decision(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _make_llm_response(COORDINATOR_RESEARCH_RESPONSE)
        mock_get_llm.return_value = mock_llm

        from agents import coordinator_agent
        result = coordinator_agent({"user_task": "What is the latest news about AI?"})

        assert result["coordinator_decision"] == "RESEARCH"
        assert result["needs_research"] is True

    @patch("agents._get_llm")
    def test_research_and_analysis_decision(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _make_llm_response(COORDINATOR_BOTH_RESPONSE)
        mock_get_llm.return_value = mock_llm

        from agents import coordinator_agent
        result = coordinator_agent({"user_task": "Compare the best LLMs available today."})

        assert result["coordinator_decision"] == "RESEARCH_AND_ANALYSIS"
        assert result["needs_research"] is True

    def test_empty_task(self):
        from agents import coordinator_agent
        result = coordinator_agent({"user_task": ""})
        assert result.get("error") is not None
        assert "No task" in result["error"]

    @patch("agents._get_llm")
    def test_invalid_json_falls_back_to_analysis(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _make_llm_response("not valid json at all")
        mock_get_llm.return_value = mock_llm

        from agents import coordinator_agent
        result = coordinator_agent({"user_task": "Some task."})

        # Should fall back gracefully
        assert result["coordinator_decision"] == "ANALYSIS"
        assert result.get("error") is None
        assert len(result.get("warnings", [])) > 0

    @patch("agents._get_llm")
    def test_unknown_decision_defaults_to_analysis(self, mock_get_llm):
        bad_decision = json.dumps({"decision": "UNKNOWN_MODE", "needs_research": False, "reasoning": "?"})
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _make_llm_response(bad_decision)
        mock_get_llm.return_value = mock_llm

        from agents import coordinator_agent
        result = coordinator_agent({"user_task": "Do something."})
        assert result["coordinator_decision"] == "ANALYSIS"

    def test_missing_api_key(self):
        from agents import coordinator_agent
        with patch("agents.GOOGLE_API_KEY", None):
            with patch("agents._get_llm", side_effect=EnvironmentError("GOOGLE_API_KEY is not set.")):
                result = coordinator_agent({"user_task": "What is AI?"})
                assert result.get("error") is not None
                assert "GOOGLE_API_KEY" in result["error"]


# ── Research Agent Tests ──────────────────────────────────────────────────────

class TestResearchAgent:
    @patch("agents.web_search")
    @patch("agents._get_llm")
    def test_successful_research(self, mock_get_llm, mock_search):
        mock_search.return_value = [
            {"title": "AI News", "href": "https://example.com/ai", "body": "AI is advancing rapidly."},
            {"title": "ML Update", "href": "https://example.com/ml", "body": "Machine learning progress."},
        ]
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _make_llm_response("AI is advancing with new models.")
        mock_get_llm.return_value = mock_llm

        from agents import research_agent
        result = research_agent({"user_task": "Latest AI news"})

        assert result["research_findings"] == "AI is advancing with new models."
        assert len(result["source_urls"]) == 2
        assert "https://example.com/ai" in result["source_urls"]

    @patch("agents.web_search")
    def test_empty_search_results(self, mock_search):
        mock_search.return_value = []

        from agents import research_agent
        result = research_agent({"user_task": "Very obscure topic"})

        assert "No relevant web results" in result["research_findings"]
        assert result["source_urls"] == []
        assert len(result.get("warnings", [])) > 0

    @patch("agents.web_search")
    def test_filtered_all_irrelevant(self, mock_search):
        # Results with no body — will all be filtered out
        mock_search.return_value = [
            {"title": "Page", "href": "https://example.com", "body": ""},
            {"title": "Page2", "href": "", "body": "Some text"},
        ]

        from agents import research_agent
        result = research_agent({"user_task": "Topic"})

        assert result["source_urls"] == []
        assert len(result.get("warnings", [])) > 0

    def test_empty_task(self):
        from agents import research_agent
        result = research_agent({"user_task": ""})
        assert "No task" in result["research_findings"]
        assert result["source_urls"] == []


# ── Analysis Agent Tests ──────────────────────────────────────────────────────

class TestAnalysisAgent:
    @patch("agents._get_llm")
    def test_analysis_with_research(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _make_llm_response("Key insight: X causes Y.")
        mock_get_llm.return_value = mock_llm

        from agents import analysis_agent
        result = analysis_agent({
            "user_task": "Explain AI trends.",
            "research_findings": "AI is growing fast.",
            "source_urls": ["https://example.com"],
        })

        assert result["analysis"] == "Key insight: X causes Y."
        assert result["current_agent"] == "analysis"

    @patch("agents._get_llm")
    def test_analysis_without_research(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _make_llm_response("Binary trees are hierarchical structures.")
        mock_get_llm.return_value = mock_llm

        from agents import analysis_agent
        result = analysis_agent({
            "user_task": "Explain binary trees.",
            "research_findings": "",
            "source_urls": [],
        })

        assert "Binary trees" in result["analysis"]

    @patch("agents._get_llm")
    def test_analysis_error_handling(self, mock_get_llm):
        mock_get_llm.side_effect = Exception("Model timeout")

        from agents import analysis_agent
        result = analysis_agent({
            "user_task": "Explain something.",
            "research_findings": "",
            "source_urls": [],
        })

        assert "could not be completed" in result["analysis"]


# ── Writer Agent Tests ────────────────────────────────────────────────────────

class TestWriterAgent:
    @patch("agents._get_llm")
    def test_successful_write(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _make_llm_response(
            "# Answer\n\nThis is the final answer."
        )
        mock_get_llm.return_value = mock_llm

        from agents import writer_agent
        result = writer_agent({
            "user_task": "Explain X.",
            "analysis": "X is important because of Y.",
            "research_findings": "Researchers found X.",
            "source_urls": ["https://research.com"],
            "conversation_history": [],
            "error": None,
        })

        assert "# Answer" in result["final_response"]
        assert result["current_agent"] == "writer"
        # Conversation history should be updated
        assert len(result["conversation_history"]) == 2
        assert result["conversation_history"][0]["role"] == "user"
        assert result["conversation_history"][1]["role"] == "assistant"

    def test_passes_through_upstream_error(self):
        from agents import writer_agent
        result = writer_agent({
            "user_task": "Any task",
            "analysis": "",
            "research_findings": "",
            "source_urls": [],
            "conversation_history": [],
            "error": "Coordinator crashed",
        })

        assert "error occurred" in result["final_response"].lower() or "error" in result["final_response"].lower()

    @patch("agents._get_llm")
    def test_writer_error_handling(self, mock_get_llm):
        mock_get_llm.side_effect = Exception("API rate limit")

        from agents import writer_agent
        result = writer_agent({
            "user_task": "Write something.",
            "analysis": "Some analysis.",
            "research_findings": "",
            "source_urls": [],
            "conversation_history": [],
            "error": None,
        })

        assert "failed" in result["final_response"].lower() or "error" in result["final_response"].lower()

    @patch("agents._get_llm")
    def test_conversation_history_accumulation(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _make_llm_response("Answer to second question.")
        mock_get_llm.return_value = mock_llm

        from agents import writer_agent
        existing_history = [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
        ]
        result = writer_agent({
            "user_task": "Second question",
            "analysis": "Analysis.",
            "research_findings": "",
            "source_urls": [],
            "conversation_history": existing_history,
            "error": None,
        })

        assert len(result["conversation_history"]) == 4
