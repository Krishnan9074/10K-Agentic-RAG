"""
Central configuration. Every value can be overridden with an env var, a .env
file, or Streamlit secrets (in that order of precedence: secrets > env).
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent


def secret(key: str, default: str = "") -> str:
    try:
        import streamlit as st
        return st.secrets.get(key, os.environ.get(key, default))
    except Exception:
        return os.environ.get(key, default)


# --------------------------------------------------------------------------- #
#  LLM (OpenRouter, OpenAI-compatible API)                                      #
# --------------------------------------------------------------------------- #
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def openrouter_api_key() -> str:
    return secret("OPENROUTER_API_KEY")


# Answering model (user can change it in the UI)
CHAT_MODEL = secret("OPENROUTER_MODEL", "anthropic/claude-sonnet-5.5")
# Cheap/fast model for routing, query planning and grounding checks
FAST_MODEL = secret("OPENROUTER_FAST_MODEL", "google/gemini-3.8-flash")

CHAT_MODEL_CHOICES = [
    "anthropic/claude-sonnet-5.5",
    "anthropic/claude-opus-5.5",
    "openai/gpt-5.6-sol",
    "google/gemini-3.8-flash",
    "qwen/qwen3.8-max-0902",
]

# --------------------------------------------------------------------------- #
#  SEC EDGAR                                                                    #
# --------------------------------------------------------------------------- #
# SEC requires a descriptive User-Agent with contact info:
# https://www.sec.gov/os/accessing-edgar-data
SEC_USER_AGENT = secret("SEC_USER_AGENT", "10K-Agentic-RAG research admin@example.com")
SEC_MAX_RPS = 8.0  # SEC hard limit is 10 req/s

DEFAULT_FORMS = ["10-K", "10-Q", "8-K", "3", "4"]
# Per-form cap on how many filings to pull per company on first ingest
DEFAULT_LIMITS = {"10-K": 3, "10-Q": 4, "8-K": 12, "3": 25, "4": 100}
DEFAULT_SINCE_YEARS = 3

# Pulled automatically the first time the app starts with an empty database.
WATCHLIST = [t.strip().upper() for t in secret("SECRAG_WATCHLIST", "AAPL,MSFT,GOOGL,AMZN").split(",")
             if t.strip()]
# Background refresh of every tracked company (0 disables).
AUTO_REFRESH_HOURS = float(secret("SECRAG_AUTO_REFRESH_HOURS", "12"))

# --------------------------------------------------------------------------- #
#  Storage                                                                      #
# --------------------------------------------------------------------------- #
DATA_DIR = Path(secret("SECRAG_DATA_DIR", str(ROOT / "data")))
DB_PATH = DATA_DIR / "secrag.sqlite3"
CACHE_DIR = DATA_DIR / "edgar_cache"

# Qdrant: Cloud when QDRANT_URL is set, otherwise an embedded on-disk instance.
QDRANT_LOCAL_PATH = DATA_DIR / "qdrant"
COLLECTION_NAME = secret("QDRANT_COLLECTION", "sec_filings")


def qdrant_url() -> str:
    return secret("QDRANT_URL")


def qdrant_api_key() -> str:
    return secret("QDRANT_API_KEY")


# --------------------------------------------------------------------------- #
#  Embeddings / chunking / retrieval                                            #
# --------------------------------------------------------------------------- #
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150
TOP_K = 8

MAX_REQUESTS_PER_MINUTE = 20
