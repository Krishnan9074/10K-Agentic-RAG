"""Financials (XBRL) + 8-K event timeline."""
import altair as alt
import pandas as pd
import streamlit as st

from secrag.analytics import financials
from secrag.parsers.xbrl import METRICS
from views.common import SERIES, jobs_banner, ticker_picker

st.title("📈 Financials & Events")
st.caption("Exact as-reported numbers from SEC XBRL companyfacts, plus every 8-K event.")
jobs_banner()

tickers = ticker_picker("Companies (compare up to 4)", key="fin_tickers", multi=True)
if not tickers:
    st.stop()

c1, c2 = st.columns([2, 1])
metric = c1.selectbox("Metric", list(METRICS), format_func=lambda m: METRICS[m][0],
                      index=list(METRICS).index("revenue"))
period = c2.radio("Period", ["annual", "quarterly"], horizontal=True)

frames = []
for t in tickers:
    s = financials.series(t, [metric], period)
    kind = "instant" if METRICS[metric][2] == "dei" or metric in (
        "cash", "total_assets", "total_liabilities", "long_term_debt", "equity",
        "deferred_tax_assets", "shares_outstanding", "nol_carryforwards") else period
    s = s[s["period_type"] == kind]
    if not s.empty:
        frames.append(s.assign(ticker=t))

if frames:
    df = pd.concat(frames)
    df["period_end"] = pd.to_datetime(df["period_end"])
    df = df.sort_values("period_end").groupby("ticker").tail(12 if period == "quarterly" else 8)
    unit = df["unit"].iloc[0]
    scale = 1e9 if unit == "USD" else 1e6 if unit == "shares" else 1
    suffix = " ($B)" if unit == "USD" else " (M shares)" if unit == "shares" else ""
    df["value_scaled"] = df["value"] / scale
    color = alt.Color("ticker:N", scale=alt.Scale(domain=tickers, range=SERIES[:len(tickers)]),
                      legend=alt.Legend(title=None, orient="top") if len(tickers) > 1 else None)
    tooltip = [alt.Tooltip("ticker:N"), alt.Tooltip("period_end:T", title="Period end"),
               alt.Tooltip("value_scaled:Q", title=METRICS[metric][0] + suffix, format=",.2f"),
               alt.Tooltip("form:N", title="Reported in")]
    base = alt.Chart(df).encode(
        x=alt.X("period_end:T", title=None),
        y=alt.Y("value_scaled:Q", title=METRICS[metric][0] + suffix),
        color=color, tooltip=tooltip,
    )
    chart = base.mark_line(strokeWidth=2) + base.mark_point(size=64, filled=True)
    st.altair_chart(chart.properties(height=340), width="stretch")
else:
    st.info(f"No {METRICS[metric][0]} facts reported for the selected companies.")

for t in tickers:
    st.subheader(t)
    tab1, tab2, tab3 = st.tabs(["Statement", "Ratios", "8-K events"])
    with tab1:
        tbl = financials.table(t, period, 5 if period == "annual" else 6)
        st.dataframe(tbl, width="stretch") if not tbl.empty else st.caption("No data.")
    with tab2:
        r = financials.ratios(t)
        st.dataframe(r.tail(6), width="stretch") if not r.empty else st.caption("No data.")
    with tab3:
        ev = financials.events(t)
        if ev.empty:
            st.caption("No 8-K events indexed.")
        else:
            flags = int(ev["red_flag"].sum())
            if flags:
                st.warning(f"⚠️ {flags} red-flag 8-K item(s): impairments, restatements, auditor "
                           "changes, defaults, delisting, cyber incidents or restructurings.")
            ev["red_flag"] = ev["red_flag"].map({1: "⚠️ red flag", 0: ""})
            st.dataframe(ev[["filed_date", "item", "title", "red_flag", "summary"]],
                         hide_index=True, width="stretch")
