"""
tests/test_workflow.py
-----------------------
Integration-level tests for the LangGraph workflow routing logic.

All LLM calls and web searches are mocked.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


def _mock_llm_factory(response_text: str) -> MagicMock:
    mock_msg = MagicMock()
    mock_msg.content = response_text
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_msg
    return mock_llm


ANALYSIS_DECISION = json.dumps({
    "decision": "ANALYSIS",
    "needs_research": False,
    "reasoning": "General knowledge task.",
})

RESEARCH_DECISION = json.dumps({
    "decision": "RESEARCH",
    "needs_research": True,
    "reasoning": "Needs current data.",
})


# ── Routing tests ─────────────────────────────────────────────────────────────

class TestWorkflowRouting:
    @patch("agents.web_search", return_value=[])
    @patch("agents._get_llm")
    def test_analysis_path_skips_research(self, mock_get_llm, mock_search):
        """When coordinator decides ANALYSIS, research_agent must not make web searches."""
        # Return a valid mock LLM for every agent call
        def make_llm():
            mock = MagicMock()
            def invoke(messages):
                system_content = messages[0].content if messages else ""
                if "Coordinator Agent" in system_content:
                    r = MagicMock(); r.content = ANALYSIS_DECISION; return r
                elif "Analysis Agent" in system_content:
                    r = MagicMock(); r.content = "Analysis text."; return r
                elif "Writer Agent" in system_content:
                    r = MagicMock(); r.content = "# Final Answer\n\nAnswer."; return r
                r = MagicMock(); r.content = "Generic response."; return r
            mock.invoke = invoke
            return mock

        mock_get_llm.side_effect = make_llm

        from workflow import run_workflow
        result = run_workflow("Explain recursion in programming.")

        # Coordinator said no research needed → web_search should never be called
        mock_search.assert_not_called()
        assert result.get("needs_research") is False
        assert result.get("source_urls", []) == []
        assert result.get("final_response", "") != ""
        assert result.get("error") is None

    @patch("agents.web_search")
    @patch("agents._get_llm")
    def test_research_path_calls_research(self, mock_get_llm, mock_search):
        """When coordinator decides RESEARCH, research_agent must run."""
        mock_search.return_value = [
            {"title": "AI News", "href": "https://ainews.com", "body": "AI is growing."},
        ]
        call_sequence: list[str] = []

        def side_effect_llm():
            mock = MagicMock()
            def invoke(messages):
                system_content = messages[0].content if messages else ""
                if "Coordinator Agent" in system_content:
                    call_sequence.append("coordinator")
                    r = MagicMock(); r.content = RESEARCH_DECISION; return r
                elif "Research Agent" in system_content:
                    call_sequence.append("research")
                    r = MagicMock(); r.content = "AI research summary."; return r
                elif "Analysis Agent" in system_content:
                    call_sequence.append("analysis")
                    r = MagicMock(); r.content = "Trend analysis."; return r
                elif "Writer Agent" in system_content:
                    call_sequence.append("writer")
                    r = MagicMock(); r.content = "# AI Trends\n\nAI is evolving."; return r
                r = MagicMock(); r.content = "Generic."; return r
            mock.invoke = invoke
            return mock

        mock_get_llm.side_effect = side_effect_llm

        from workflow import run_workflow
        result = run_workflow("What are the latest AI trends in 2025?")

        assert "research" in call_sequence, (
            f"Research was NOT called despite RESEARCH decision. Call sequence: {call_sequence}"
        )
        assert result.get("error") is None


class TestWorkflowEdgeCases:
    def test_empty_task_returns_error(self):
        """Submitting an empty task should produce a graceful error, not crash."""
        from workflow import run_workflow
        # Coordinator will catch the empty task without calling LLM
        result = run_workflow("")
        # Should not raise; should contain an error message
        assert result is not None

    @patch("agents.web_search", return_value=[])
    @patch("agents._get_llm")
    def test_workflow_completes_with_no_search_results(self, mock_get_llm, mock_search):
        """Empty search results should not crash the workflow."""
        call_count = [0]

        def side_effect_llm():
            mock = MagicMock()
            def invoke(messages):
                call_count[0] += 1
                system_content = messages[0].content if messages else ""
                if "Coordinator Agent" in system_content:
                    r = MagicMock(); r.content = RESEARCH_DECISION; return r
                r = MagicMock(); r.content = "Generic response."; return r
            mock.invoke = invoke
            return mock

        mock_get_llm.side_effect = side_effect_llm

        from workflow import run_workflow
        result = run_workflow("Latest Mars mission news?")

        assert result is not None
        assert result.get("final_response") != "" or result.get("error") is not None

    @patch("agents.web_search", return_value=[])
    @patch("agents._get_llm")
    def test_workflow_streaming_yields_events(self, mock_get_llm, mock_search):
        """stream_workflow_events should yield (node_name, partial_state) tuples."""
        def side_effect_llm():
            mock = MagicMock()
            def invoke(messages):
                system_content = messages[0].content if messages else ""
                if "Coordinator Agent" in system_content:
                    r = MagicMock(); r.content = ANALYSIS_DECISION; return r
                r = MagicMock(); r.content = "Some response."; return r
            mock.invoke = invoke
            return mock

        mock_get_llm.side_effect = side_effect_llm

        from workflow import stream_workflow_events
        events = list(stream_workflow_events("Explain Python decorators."))

        assert len(events) > 0, "No events were streamed"
        for node_name, partial_state in events:
            assert isinstance(node_name, str)
            assert isinstance(partial_state, dict)


class TestMemoryModule:
    def test_append_turn(self):
        from memory import append_turn
        history = []
        history = append_turn(history, "user", "Hello")
        history = append_turn(history, "assistant", "Hi!")
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["content"] == "Hi!"

    def test_format_history_empty(self):
        from memory import format_history_for_prompt
        result = format_history_for_prompt([])
        assert result == ""

    def test_format_history_non_empty(self):
        from memory import format_history_for_prompt
        history = [
            {"role": "user", "content": "Question?"},
            {"role": "assistant", "content": "Answer."},
        ]
        result = format_history_for_prompt(history)
        assert "Question?" in result
        assert "Answer." in result
        assert "History" in result

    def test_clear_history(self):
        from memory import clear_history
        result = clear_history()
        assert result == []


class TestToolsModule:
    def test_filter_results_removes_empty_bodies(self):
        from tools import filter_results
        results = [
            {"title": "Good", "href": "https://a.com", "body": "Relevant content."},
            {"title": "Bad",  "href": "https://b.com", "body": ""},
            {"title": "Bad2", "href": "",               "body": "Something."},
        ]
        filtered = filter_results(results)
        assert len(filtered) == 1
        assert filtered[0]["href"] == "https://a.com"

    def test_extract_sources_deduplicates(self):
        from tools import extract_sources
        results = [
            {"href": "https://a.com", "body": "x"},
            {"href": "https://a.com", "body": "y"},   # duplicate
            {"href": "https://b.com", "body": "z"},
        ]
        sources = extract_sources(results)
        assert len(sources) == 2
        assert sources[0] == "https://a.com"
        assert sources[1] == "https://b.com"

    def test_format_search_results_empty(self):
        from tools import format_search_results
        result = format_search_results([])
        assert "No search results" in result

    def test_format_search_results_non_empty(self):
        from tools import format_search_results
        results = [{"title": "Title", "href": "https://x.com", "body": "Body text."}]
        output = format_search_results(results)
        assert "Title" in output
        assert "https://x.com" in output
        assert "Body text." in output

    @patch("tools.DDGS")
    def test_web_search_returns_results(self, mock_ddgs_class):
        from tools import web_search
        mock_instance = MagicMock()
        mock_instance.__enter__ = MagicMock(return_value=mock_instance)
        mock_instance.__exit__ = MagicMock(return_value=False)
        mock_instance.text.return_value = [
            {"title": "T", "href": "https://test.com", "body": "Body."}
        ]
        mock_ddgs_class.return_value = mock_instance

        results = web_search("test query", max_results=1)
        assert len(results) == 1
        assert results[0]["href"] == "https://test.com"

    @patch("tools.DDGS")
    def test_web_search_handles_exception(self, mock_ddgs_class):
        from tools import web_search
        mock_ddgs_class.side_effect = Exception("Network error")
        results = web_search("query")
        assert results == []

    def test_web_search_empty_query(self):
        from tools import web_search
        results = web_search("")
        assert results == []
