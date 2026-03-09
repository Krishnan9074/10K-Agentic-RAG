import os
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

# Qdrant — reads from .env locally, from Streamlit secrets in cloud
qdrant_url     = _secret("QDRANT_URL")
qdrant_api_key = _secret("QDRANT_API_KEY")
collection_name = "rag"

# Groq API key — expose as env var so langchain-groq picks it up automatically
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