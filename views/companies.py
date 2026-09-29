"""Companies: add tickers, watch background ingestion, browse what is indexed."""

import streamlit as st

from secrag import config
from secrag.edgar import search_companies
from secrag.store import db
from views.common import manager, store

st.title("🏢 Companies")
st.caption("Everything here is pulled straight from SEC EDGAR: filings are parsed, split into "
           "sections, embedded into Qdrant, and structured data lands in SQLite.")

mgr = manager()

with st.container(border=True):
    st.subheader("Add companies")
    c1, c2 = st.columns([3, 1])
    raw = c1.text_input("Tickers or company names (comma-separated)", placeholder="NVDA, Tesla, JPM")
    c2.write("")
    c2.write("")
    if c2.button("Ingest", type="primary", width="stretch") and raw.strip():
        for token in [t.strip() for t in raw.split(",") if t.strip()]:
            hits = search_companies(token, 1)
            if not hits:
                st.error(f"No SEC registrant matches {token!r}.")
                continue
            job = mgr.submit(hits[0][0])
            st.success(f"Queued {hits[0][0]} ({hits[0][1]}) · job {job.id}")
    st.caption(f"Default pull per company: last {config.DEFAULT_SINCE_YEARS} years, up to "
               + ", ".join(f"{v}× {k}" for k, v in config.DEFAULT_LIMITS.items())
               + " + full XBRL history. New filings are picked up automatically every "
               f"{config.AUTO_REFRESH_HOURS:g}h.")
    if st.button("Refresh all tracked companies now"):
        for t in sorted(db.known_tickers()):
            mgr.submit(t, kind="refresh")
        st.success("Refresh queued.")


@st.fragment(run_every=2)
def jobs_panel():
    jobs = mgr.recent(12)
    if not jobs:
        return
    st.subheader("Ingestion jobs")
    for j in jobs:
        icon = {"queued": "⏳", "running": "⚙️", "done": "✅", "error": "❌"}[j.status]
        with st.expander(f"{icon} {j.ticker} · {j.kind} · {j.status}", expanded=j.status == "running"):
            st.progress(min(j.progress, 1.0), text=j.message or j.status)
            if j.summary:
                st.write(j.summary)
            if j.finished:
                st.caption(f"took {j.finished - j.started:,.0f}s")
            if j.log:
                st.code("\n".join(j.log[-15:]), language=None)


jobs_panel()

st.subheader("Indexed")
cos = db.companies()
if cos.empty:
    st.info("Nothing indexed yet. On first launch the watchlist "
            f"({', '.join(config.WATCHLIST)}) is ingested automatically.")
else:
    m1, m2, m3 = st.columns(3)
    m1.metric("Companies", len(cos))
    m2.metric("Filings", int(cos["filings"].sum()))
    try:
        m3.metric("Vector chunks", f"{store().count():,}")
    except Exception:
        m3.metric("Vector chunks", "-")
    st.dataframe(cos[["ticker", "name", "sic_description", "fiscal_year_end", "filings",
                      "last_ingested"]], hide_index=True, width="stretch")

    t = st.selectbox("Filings for", sorted(cos["ticker"]))
    f = db.filings(t)
    if not f.empty:
        f["form_type"] = f["form_type"].astype(str)
        counts = f[f["status"] == "ok"].groupby("form_type").size()
        st.caption(" · ".join(f"{k}: {v}" for k, v in counts.items()))
        st.dataframe(
            f[["filed_date", "form_type", "report_date", "status", "chunks", "url", "error"]],
            hide_index=True, width="stretch",
            column_config={"url": st.column_config.LinkColumn("EDGAR", display_text="open")},
        )
