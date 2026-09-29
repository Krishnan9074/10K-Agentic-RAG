"""Shared Streamlit helpers."""
from __future__ import annotations

import streamlit as st

from secrag import config
from secrag.store import db

# Validated categorical order (dataviz reference palette, light steps).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
BUY, SELL, NEUTRAL = "#2a78d6", "#eb6834", "#9a9890"


@st.cache_resource(show_spinner=False)
def manager():
    from secrag.pipeline.jobs import get_manager
    return get_manager(start_scheduler=True)


@st.cache_resource(show_spinner="Loading embedding model and vector store...")
def store():
    from secrag.store.vectors import get_store
    return get_store()


def require_api_key() -> bool:
    if config.openrouter_api_key():
        return True
    st.warning(
        "**OPENROUTER_API_KEY is not set.** Add it to `.env` (local) or Streamlit secrets "
        "(cloud), then reload. Ingestion and dashboards work without it; chat, summaries and "
        "grounding checks need it. Get a key at https://openrouter.ai/keys", icon="🔑")
    return False


def jobs_banner() -> None:
    active = manager().active()
    if not active:
        return
    with st.container(border=True):
        st.caption("Background ingestion from SEC EDGAR")
        for j in active:
            st.progress(min(j.progress, 1.0), text=f"**{j.ticker}** · {j.status} · {j.message}")


def ticker_picker(label: str = "Company", key: str = "ticker", multi: bool = False,
                  max_selections: int = 4):
    tickers = sorted(db.known_tickers())
    if not tickers:
        st.info("No companies indexed yet. Add one on the **Companies** page, or just ask "
                "about any company in **Ask** and it will be pulled from EDGAR automatically.")
        return [] if multi else None
    if multi:
        return st.multiselect(label, tickers, default=tickers[:1], key=key,
                              max_selections=max_selections)
    return st.selectbox(label, tickers, key=key)


def usd(x) -> str:
    if x is None or x != x:
        return "-"
    a = abs(x)
    for d, s in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if a >= d:
            return f"{'-' if x < 0 else ''}${a / d:,.1f}{s}"
    return f"{'-' if x < 0 else ''}${a:,.0f}"
