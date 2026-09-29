"""OpenRouter chat models (OpenAI-compatible endpoint via langchain-openai)."""
from __future__ import annotations

from langchain_openai import ChatOpenAI

from secrag import config


class MissingAPIKey(RuntimeError):
    pass


def make_llm(model: str | None = None, *, temperature: float = 0.0, streaming: bool = False) -> ChatOpenAI:
    key = config.openrouter_api_key()
    if not key:
        raise MissingAPIKey(
            "OPENROUTER_API_KEY is not set. Add it to .env or Streamlit secrets "
            "(get one at https://openrouter.ai/keys)."
        )
    return ChatOpenAI(
        model=model or config.CHAT_MODEL,
        base_url=config.OPENROUTER_BASE_URL,
        api_key=key,
        temperature=temperature,
        streaming=streaming,
        timeout=120,
        max_retries=2,
        default_headers={
            "HTTP-Referer": "https://github.com/Krishnan9074/10K-Agentic-RAG",
            "X-Title": "10K Agentic RAG",
        },
    )


def fast_llm() -> ChatOpenAI:
    return make_llm(config.FAST_MODEL)
