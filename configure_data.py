import os
from urllib.parse import urlparse, urlunparse
from dotenv import load_dotenv
load_dotenv()

# Pull from Streamlit secrets when running in Streamlit Cloud,
# otherwise fall back to environment variables / .env file
def _secret(key: str) -> str:
    try:
        import streamlit as st
        return st.secrets.get(key, os.environ.get(key, ""))
    except Exception:
        return os.environ.get(key, "")

md5_path = "./md5.text"

# Qdrant — reads at call time so Streamlit secrets are always initialised first
def _get_qdrant_url() -> str:
    return _secret("QDRANT_URL")

def _get_qdrant_api_key() -> str:
    return _secret("QDRANT_API_KEY")

# Legacy module-level aliases (used by ingest_10k and local scripts)
qdrant_url     = _get_qdrant_url
qdrant_api_key = _get_qdrant_api_key
collection_name = "rag"


def make_qdrant_client():
    """Build a QdrantClient that works on both local and Streamlit Cloud.

    Streamlit Cloud blocks outbound port 6333.  Qdrant Cloud also listens on
    port 443, so we rewrite the URL to force HTTPS/443.
    """
    from qdrant_client import QdrantClient
    url = _get_qdrant_url()
    api_key = _get_qdrant_api_key()
    # If no port is specified, qdrant-client defaults to 6333 which is
    # often blocked on shared cloud runners — force 443 instead.
    parsed = urlparse(url)
    if not parsed.port:
        netloc = f"{parsed.hostname}:443"
        url = urlunparse(parsed._replace(netloc=netloc, scheme="https"))
    return QdrantClient(url=url, api_key=api_key)


# Groq API key — set as env var so langchain-groq picks it up automatically
os.environ.setdefault("GROQ_API_KEY", _secret("GROQ_API_KEY"))


#spliter
chunk_size = 1000
chunk_overlap = 100
separators = ["\n\n","\n","。","？","?","！","!"]
max_split_char_number = 1000

similarity_threshold = 6

# Local embedding model via fastembed (no API key needed)
embedding_model_name = "BAAI/bge-small-en-v1.5"

# Groq chat model
chat_model_name = "llama-3.3-70b-versatile"



session_con = {
        "configurable":{
            "session_id":"user_002"
        }
    }