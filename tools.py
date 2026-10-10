"""
tools.py
--------
Tool wrappers used by the Research Agent.

DuckDuckGo search is performed with the duckduckgo-search library (v8.x).
The public API uses DDGS as a context manager and returns results as dicts.
"""

from __future__ import annotations

import logging
from typing import Any

from ddgs import DDGS

from config import MAX_SEARCH_RESULTS, SEARCH_REGION

logger = logging.getLogger(__name__)


def web_search(query: str, max_results: int = MAX_SEARCH_RESULTS) -> list[dict[str, Any]]:
    """
    Perform a DuckDuckGo text search and return a list of result dicts.

    Each dict contains:
        - title  : str
        - href   : str  (source URL)
        - body   : str  (snippet)

    Returns an empty list on any failure.
    """
    if not query or not query.strip():
        logger.warning("web_search called with empty query – skipping.")
        return []

    try:
        with DDGS() as ddgs:
            results: list[dict[str, Any]] = list(
                ddgs.text(
                    keywords=query.strip(),
                    region=SEARCH_REGION,
                    max_results=max_results,
                )
            )
        logger.info("DuckDuckGo returned %d results for query: %s", len(results), query[:80])
        return results
    except Exception as exc:
        logger.error("DuckDuckGo search failed: %s", exc)
        return []


def filter_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Remove results that are obviously irrelevant (empty body or href).
    """
    filtered = [
        r for r in results
        if r.get("href") and r.get("body") and r.get("body").strip()
    ]
    logger.debug("filter_results: %d → %d results", len(results), len(filtered))
    return filtered


def extract_sources(results: list[dict[str, Any]]) -> list[str]:
    """Return a deduplicated, ordered list of source URLs from search results."""
    seen: set[str] = set()
    urls: list[str] = []
    for r in results:
        href = r.get("href", "").strip()
        if href and href not in seen:
            seen.add(href)
            urls.append(href)
    return urls


def format_search_results(results: list[dict[str, Any]]) -> str:
    """
    Format a list of search result dicts into a readable text block
    suitable for including in a model prompt.
    """
    if not results:
        return "No search results available."

    lines: list[str] = []
    for i, r in enumerate(results, start=1):
        title = r.get("title", "Untitled")
        href = r.get("href", "")
        body = r.get("body", "").strip()
        lines.append(f"[{i}] {title}")
        lines.append(f"    URL: {href}")
        lines.append(f"    Summary: {body}")
        lines.append("")
    return "\n".join(lines)
