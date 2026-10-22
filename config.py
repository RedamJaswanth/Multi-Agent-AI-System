"""
config.py
---------
Centralised configuration and environment loading.
All sensitive values are read exclusively from environment variables or Streamlit secrets.
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv

# Load .env from the project root (safe even when file is absent)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=False)

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
# Suppress noisy third-party loggers
for _noisy in ("httpx", "httpcore", "urllib3", "duckduckgo_search"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def get_secret(key_name: str, fallback_key: str | None = None) -> str | None:
    """
    Safely retrieve a configuration value or secret from environment variables
    or Streamlit Secrets (for Streamlit Community Cloud).
    """
    # 1. Environment variable
    val = os.getenv(key_name)
    if not val and fallback_key:
        val = os.getenv(fallback_key)
    if val:
        return val

    # 2. Streamlit secrets
    try:
        import streamlit as st

        if hasattr(st, "secrets"):
            if key_name in st.secrets:
                return str(st.secrets[key_name])
            if fallback_key and fallback_key in st.secrets:
                return str(st.secrets[fallback_key])
    except Exception:
        pass

    return None


# ── API keys ──────────────────────────────────────────────────────────────────
GOOGLE_API_KEY: str | None = get_secret("GEMINI_API_KEY", "GOOGLE_API_KEY")
SERPER_API_KEY: str | None = get_secret("SERPER_API_KEY")

# ── Model configuration ───────────────────────────────────────────────────────
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
MODEL_TEMPERATURE: float = float(os.getenv("MODEL_TEMPERATURE", "0.3"))
MODEL_MAX_TOKENS: int = int(os.getenv("MODEL_MAX_TOKENS", "4096"))

# ── Quota & Safeguards ───────────────────────────────────────────────────────
MAX_TURNS_LIMIT: int = int(os.getenv("MAX_TURNS_LIMIT", "10"))
MAX_SEARCH_RESULTS: int = int(os.getenv("MAX_SEARCH_RESULTS", "5"))
SEARCH_REGION: str = os.getenv("SEARCH_REGION", "wt-wt")   # worldwide
MAX_RECURSION_LIMIT: int = int(os.getenv("MAX_RECURSION_LIMIT", "25"))

# ── Application metadata ──────────────────────────────────────────────────────
APP_TITLE: str = "AI Multi-Agent Research and Analysis System"
APP_VERSION: str = "1.0.0"
