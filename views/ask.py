"""Chat page: agentic RAG over SEC filings with auto-ingest and citations."""
import time
import uuid

import streamlit as st

import file_history_store
from secrag import config
from secrag.agent.llm import MissingAPIKey
from views.common import jobs_banner, require_api_key, store

st.title("💬 Ask SEC filings")
st.caption("10-K · 10-Q · 8-K · Forms 3/4 · XBRL financials, pulled live from SEC EDGAR. "
           "Ask about any US-listed company; it is fetched and indexed automatically.")
jobs_banner()
has_key = require_api_key()

with st.sidebar:
    st.subheader("Model")
    choice = st.selectbox("Answer model (OpenRouter)", config.CHAT_MODEL_CHOICES + ["Custom..."],
                          index=config.CHAT_MODEL_CHOICES.index(config.CHAT_MODEL)
                          if config.CHAT_MODEL in config.CHAT_MODEL_CHOICES else 0)
    model = st.text_input("OpenRouter model slug", config.CHAT_MODEL) if choice == "Custom..." else choice
    auto_ingest = st.toggle("Auto-ingest unknown companies", value=True,
                            help="Pull a company's filings from EDGAR the first time you ask about it.")
    show_plan = st.toggle("Show query plan", value=False)
    if st.button("New conversation", width="stretch"):
        st.session_state.pop("messages", None)
        st.session_state["session_id"] = "user_" + uuid.uuid4().hex[:12]
        st.rerun()

st.session_state.setdefault("session_id", "user_" + uuid.uuid4().hex[:12])
st.session_state.setdefault("messages", [])
st.session_state.setdefault("request_timestamps", [])


@st.cache_resource(show_spinner="Initializing agent...")
def get_agent(model_name: str):
    from secrag.agent.rag import SecAgent
    store()
    return SecAgent(model_name)


def render_assistant(msg: dict) -> None:
    for note in msg.get("notes", []):
        st.info(note, icon="📥")
    if show_plan and msg.get("plan"):
        with st.expander("🧭 Query plan"):
            st.json(msg["plan"])
    st.markdown(msg["content"])
    if msg.get("grounded") is False:
        claims = "\n".join(f"- {c}" for c in msg.get("unsupported", []))
        st.warning(f"**Possibly unsupported by sources:**\n{claims}", icon="⚠️")
    sources = msg.get("sources", [])
    if sources:
        with st.expander(f"📄 Sources ({len(sources)})"):
            for s in sources:
                link = f" · [open on EDGAR]({s['url']})" if s.get("url") else ""
                st.markdown(f"**[{s['ref']}]** {s['title']}{link}")
                if s["ref"].startswith("T"):
                    st.markdown(s["text"][:6000])
                else:
                    st.caption(s["text"][:500].replace("\n", " ") + "…")


for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        render_assistant(m) if m["role"] == "assistant" else st.markdown(m["content"])

EXAMPLES = [
    "Compare Apple and Microsoft revenue, margins and free cash flow over the last 3 years.",
    "What new risk factors did NVIDIA add in its latest 10-K?",
    "Have Tesla insiders been buying or selling stock this year? Any 10b5-1 plans?",
    "Summarize Amazon's last earnings 8-K press release.",
]
prompt = st.chat_input("Ask about any public company's SEC filings...", disabled=not has_key)
if not st.session_state["messages"] and has_key:
    cols = st.columns(2)
    for i, ex in enumerate(EXAMPLES):
        if cols[i % 2].button(ex, key=f"ex{i}", width="stretch"):
            prompt = ex

if prompt:
    now = time.time()
    ts = [t for t in st.session_state["request_timestamps"] if now - t < 60]
    if len(ts) >= config.MAX_REQUESTS_PER_MINUTE:
        st.error("Too many requests. Please wait a moment.")
        st.stop()
    st.session_state["request_timestamps"] = ts + [now]

    history_msgs = [dict(m) for m in st.session_state["messages"]]
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    from secrag.agent.rag import to_messages
    history = to_messages(history_msgs)

    with st.chat_message("assistant"):
        try:
            agent = get_agent(model)
            with st.status("Planning and retrieving...", expanded=False) as status:
                bar = st.progress(0.0)

                def progress(frac, msg):
                    status.update(label=f"Fetching from SEC EDGAR: {msg}", expanded=True)
                    bar.progress(min(frac, 1.0), text=msg)

                prep = agent.prepare(prompt, history, auto_ingest=auto_ingest, progress=progress)
                p = prep.plan
                status.update(label=f"Plan: {', '.join(p.tickers) or 'all companies'} · "
                                    f"tools {', '.join(p.tools)} · {len(prep.sources)} sources",
                              state="complete", expanded=False)
            for note in prep.notes:
                st.info(note, icon="📥")
            answer = st.write_stream(agent.stream_answer(prep, history))
            with st.spinner("Verifying against sources..."):
                check = agent.check_grounding(answer, prep)
        except MissingAPIKey as e:
            st.error(str(e))
            st.stop()
        except Exception as e:
            st.error(f"Something went wrong: {type(e).__name__}: {e}")
            st.stop()

        msg = {
            "role": "assistant", "content": answer, "notes": prep.notes,
            "plan": prep.plan.model_dump(), "grounded": check.grounded,
            "unsupported": check.unsupported_claims,
            "sources": [{"ref": s.ref, "title": s.title, "text": s.text, "url": s.url}
                        for s in prep.sources],
        }
        if not check.grounded:
            claims = "\n".join(f"- {c}" for c in check.unsupported_claims)
            st.warning(f"**Possibly unsupported by sources:**\n{claims}", icon="⚠️")
        if msg["sources"]:
            with st.expander(f"📄 Sources ({len(msg['sources'])})"):
                for s in msg["sources"]:
                    link = f" · [open on EDGAR]({s['url']})" if s.get("url") else ""
                    st.markdown(f"**[{s['ref']}]** {s['title']}{link}")
    st.session_state["messages"].append(msg)

    try:
        from langchain_core.messages import AIMessage, HumanMessage
        file_history_store.get_his(st.session_state["session_id"]).add_messages(
            [HumanMessage(prompt), AIMessage(answer)])
    except Exception:
        pass
