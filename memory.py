"""
memory.py
---------
Conversation memory helpers.

Uses an in-memory list (session-level) backed by st.session_state when
running inside Streamlit.  Can also be used standalone for tests.
"""

from __future__ import annotations

from typing import Any


def make_turn(role: str, content: str) -> dict[str, Any]:
    """Create a single conversation turn dict."""
    return {"role": role, "content": content}


def append_turn(
    history: list[dict[str, Any]],
    role: str,
    content: str,
) -> list[dict[str, Any]]:
    """Return a new history list with the turn appended (immutable update)."""
    return history + [make_turn(role, content)]


def format_history_for_prompt(history: list[dict[str, Any]], max_turns: int = 6) -> str:
    """
    Format the last *max_turns* conversation turns into a prompt-friendly
    string.  Returns an empty string when history is empty.
    """
    if not history:
        return ""

    recent = history[-max_turns:]
    lines: list[str] = ["=== Conversation History ==="]
    for turn in recent:
        role = turn.get("role", "unknown").capitalize()
        content = turn.get("content", "")
        lines.append(f"{role}: {content}")
    lines.append("=== End of History ===")
    return "\n".join(lines)


def clear_history() -> list[dict[str, Any]]:
    """Return a fresh empty history list."""
    return []
