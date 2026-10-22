"""
tools.py
--------
Tool wrappers used by the Research Agent.
Supports Serper API (google-serper) with seamless fallback to DuckDuckGo search.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import requests
from ddgs import DDGS

from config import MAX_SEARCH_RESULTS, SEARCH_REGION, get_secret

logger = logging.getLogger(__name__)


def serper_search(query: str, max_results: int = MAX_SEARCH_RESULTS) -> list[dict[str, Any]]:
    """
    Perform a fast Google web search via Serper API.
    Returns a list of dicts with title, href, and body.
    """
    api_key = get_secret("SERPER_API_KEY")
    if not api_key:
        return []

    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": query.strip(), "num": max_results})
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(url, headers=headers, data=payload, timeout=8)
        if response.status_code == 200:
            data = response.json()
            results: list[dict[str, Any]] = []
            for item in data.get("organic", [])[:max_results]:
                results.append({
                    "title": item.get("title", "Untitled"),
                    "href": item.get("link", ""),
                    "body": item.get("snippet", ""),
                })
            logger.info("Serper returned %d results for query: %s", len(results), query[:80])
            return results
        else:
            logger.warning("Serper request returned status %d: %s", response.status_code, response.text[:200])
    except Exception as exc:
        logger.warning("Serper search failed, falling back to DuckDuckGo: %s", exc)

    return []


def web_search(query: str, max_results: int = MAX_SEARCH_RESULTS) -> list[dict[str, Any]]:
    """
    Perform a web search using Serper API (if configured) or DuckDuckGo.
    Returns a list of result dicts:
        - title  : str
        - href   : str  (source URL)
        - body   : str  (snippet)
    """
    if not query or not query.strip():
        logger.warning("web_search called with empty query – skipping.")
        return []

    # 1. Try Serper API first if key is present
    serper_results = serper_search(query, max_results=max_results)
    if serper_results:
        return serper_results

    # 2. Fallback to DuckDuckGo search
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
