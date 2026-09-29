"""Filing Diff: what changed in a 10-K / 10-Q section year over year."""
import streamlit as st

from secrag import config
from secrag.analytics import diff
from views.common import jobs_banner, require_api_key, ticker_picker

st.title("🧬 Filing Diff")
st.caption("Paragraph-level comparison of the same Item across two filings: new risks, "
           "dropped language, and quiet rewording.")
jobs_banner()

SECTIONS = {
    "10-K": {"item_1a": "Item 1A Risk Factors", "item_7": "Item 7 MD&A", "item_1": "Item 1 Business",
             "item_3": "Item 3 Legal Proceedings", "item_1c": "Item 1C Cybersecurity",
             "item_7a": "Item 7A Market Risk", "item_9a": "Item 9A Controls"},
    "10-Q": {"part2_item1a": "Part II Item 1A Risk Factors", "part1_item2": "Part I Item 2 MD&A",
             "part2_item1": "Part II Item 1 Legal Proceedings"},
}

ticker = ticker_picker(key="diff_ticker")
if not ticker:
    st.stop()
c1, c2 = st.columns(2)
form = c1.radio("Form", ["10-K", "10-Q"], horizontal=True)
section = c2.selectbox("Section", list(SECTIONS[form]), format_func=SECTIONS[form].get)

rows = diff.comparable_filings(ticker, form)
if len(rows) < 2:
    st.info(f"Need at least two ingested {form} filings for {ticker} (have {len(rows)}). "
            "Ingest more history from the Companies page or the CLI: "
            f"`python -m secrag ingest {ticker} --forms {form} --years 5`.")
    st.stop()

label = {r["accession"]: f"period {r['report_date']} · filed {r['filed_date']}" for r in rows}
c3, c4 = st.columns(2)
new_acc = c3.selectbox("Newer filing", list(label), index=0, format_func=label.get)
old_acc = c4.selectbox("Older filing", list(label), index=1, format_func=label.get)


@st.cache_data(show_spinner="Diffing sections (documents come from the local EDGAR cache)...")
def run_diff(ticker, section, form, new_acc, old_acc):
    return diff.diff_section(ticker, section, form, new_acc, old_acc)


d = run_diff(ticker, section, form, new_acc, old_acc)
s = d.stats()
m = st.columns(4)
m[0].metric("New paragraphs", s["added"])
m[1].metric("Removed", s["removed"])
m[2].metric("Reworded", s["modified"])
m[3].metric("Unchanged", s["unchanged"])

if require_api_key() and st.button("✨ AI summary of what changed", type="primary"):
    with st.spinner(f"Summarizing with {config.CHAT_MODEL}..."):
        st.markdown(diff.summarize(d))

t1, t2, t3 = st.tabs([f"🆕 New ({s['added']})", f"🗑️ Removed ({s['removed']})",
                      f"✏️ Reworded ({s['modified']})"])
with t1:
    for p in d.added:
        st.markdown(f"> {p}")
with t2:
    for p in d.removed:
        st.markdown(f"> ~~{p}~~")
with t3:
    for old, new, score in d.modified:
        a, b = st.columns(2)
        a.caption(f"Before (similarity {score})")
        a.markdown(old)
        b.caption("After")
        b.markdown(new)
        st.divider()
