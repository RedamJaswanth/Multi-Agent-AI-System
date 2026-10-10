"""
config.py
---------
Centralised configuration and environment loading.
All sensitive values are read exclusively from environment variables.
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

# ── API keys ──────────────────────────────────────────────────────────────────
GOOGLE_API_KEY: str | None = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    logger.warning(
        "GOOGLE_API_KEY is not set. "
        "Create a .env file with GOOGLE_API_KEY=<your-key> to enable model inference."
    )

# ── Model configuration ───────────────────────────────────────────────────────
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
MODEL_TEMPERATURE: float = float(os.getenv("MODEL_TEMPERATURE", "0.3"))
MODEL_MAX_TOKENS: int = int(os.getenv("MODEL_MAX_TOKENS", "4096"))

# ── Research configuration ────────────────────────────────────────────────────
MAX_SEARCH_RESULTS: int = int(os.getenv("MAX_SEARCH_RESULTS", "5"))
SEARCH_REGION: str = os.getenv("SEARCH_REGION", "wt-wt")   # worldwide

# ── Workflow safeguards ───────────────────────────────────────────────────────
MAX_RECURSION_LIMIT: int = int(os.getenv("MAX_RECURSION_LIMIT", "25"))

# ── Application metadata ──────────────────────────────────────────────────────
APP_TITLE: str = "AI Multi-Agent Research and Analysis System"
APP_VERSION: str = "1.0.0"
