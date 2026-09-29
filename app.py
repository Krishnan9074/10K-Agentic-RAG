"""
10K Agentic RAG -- Streamlit entry point.

    streamlit run app.py
"""
import logging

import streamlit as st

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
for noisy in ("httpx", "urllib3", "fastembed", "huggingface_hub"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

st.set_page_config(page_title="10K Agentic RAG", page_icon="📊", layout="wide")

from views.common import manager  # noqa: E402

manager()  # starts background ingestion + auto-refresh once per process

nav = st.navigation({
    "Research": [
        st.Page("views/ask.py", title="Ask", icon="💬", default=True),
        st.Page("views/financials.py", title="Financials & Events", icon="📈"),
        st.Page("views/insiders.py", title="Insider Radar", icon="🕵️"),
        st.Page("views/diff.py", title="Filing Diff", icon="🧬"),
    ],
    "Data": [
        st.Page("views/companies.py", title="Companies", icon="🏢"),
        st.Page("views/upload.py", title="Upload documents", icon="📤"),
    ],
})
nav.run()
