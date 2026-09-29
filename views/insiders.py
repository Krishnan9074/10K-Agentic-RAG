"""Insider Radar: Form 3/4 open-market activity, 10b5-1 share, pre-event selling."""
import altair as alt
import pandas as pd
import streamlit as st

from secrag.analytics import insiders
from secrag.parsers.form345 import TRANSACTION_CODES
from views.common import BUY, NEUTRAL, SELL, jobs_banner, ticker_picker, usd

st.title("🕵️ Insider Radar")
st.caption("Parsed from SEC Forms 3 and 4. Open-market trades (codes P/S) are the "
           "discretionary signal; grants, option exercises and tax withholding are shown "
           "but not counted as buying/selling.")
jobs_banner()

ticker = ticker_picker(key="ins_ticker")
if not ticker:
    st.stop()
days = st.select_slider("Look-back", options=[90, 180, 365, 730, 1095], value=365,
                        format_func=lambda d: f"{d // 365}y" if d >= 365 else f"{d}d")

tx = insiders.transactions(ticker, days)
summary = insiders.summary_by_insider(ticker, days)
if tx.empty:
    st.info(f"No insider transactions indexed for {ticker} in this window.")
    st.stop()

om = tx[tx["transaction_code"].isin(["P", "S"]) & (tx["derivative"] == 0)]
buys, sells = om[om["transaction_code"] == "P"], om[om["transaction_code"] == "S"]
k1, k2, k3, k4 = st.columns(4)
k1.metric("Open-market buys", usd(buys["value"].sum()), f"{len(buys)} trades", delta_color="off")
k2.metric("Open-market sells", usd(sells["value"].sum()), f"{len(sells)} trades", delta_color="off")
k3.metric("Insiders selling", sells["owner_name"].nunique())
k4.metric("Sells under 10b5-1 plans",
          f"{100 * sells['aff10b5_one'].mean():.0f}%" if len(sells) else "-")

st.subheader("Trade timeline")
plot = tx[tx["shares"].notna()].copy()
plot["side"] = plot["transaction_code"].map({"P": "Open-market buy", "S": "Open-market sale"}).fillna("Other")
plot["code_desc"] = plot["transaction_code"].map(TRANSACTION_CODES).fillna("Other")
plot["transaction_date"] = pd.to_datetime(plot["transaction_date"])
plot["plan"] = plot["aff10b5_one"].map({1: "10b5-1 plan", 0: "discretionary"})
chart = alt.Chart(plot).mark_circle(opacity=0.85, stroke="white", strokeWidth=2).encode(
    x=alt.X("transaction_date:T", title=None),
    y=alt.Y("owner_name:N", title=None, sort="-x"),
    size=alt.Size("shares:Q", legend=None, scale=alt.Scale(range=[64, 900])),
    color=alt.Color("side:N", scale=alt.Scale(
        domain=["Open-market buy", "Open-market sale", "Other"], range=[BUY, SELL, NEUTRAL]),
        legend=alt.Legend(title=None, orient="top")),
    tooltip=[alt.Tooltip("transaction_date:T", title="Date"), "owner_name:N", "owner_role:N",
             alt.Tooltip("code_desc:N", title="Transaction"),
             alt.Tooltip("shares:Q", format=",.0f"), alt.Tooltip("price:Q", format="$,.2f"),
             alt.Tooltip("value:Q", format="$,.0f"), alt.Tooltip("shares_after:Q", format=",.0f",
                                                                   title="Held after"), "plan:N"],
)
st.altair_chart(chart.properties(height=max(220, 34 * plot["owner_name"].nunique())),
                width="stretch")

st.subheader("By insider")
st.dataframe(
    summary, hide_index=True, width="stretch",
    column_config={c: st.column_config.NumberColumn(format="dollar")
                   for c in ["open_mkt_buys_usd", "open_mkt_sells_usd", "net_usd"]}
    | {"pct_sells_10b5_1": st.column_config.NumberColumn("% sells 10b5-1", format="%.0f%%")},
)

st.subheader("Selling before 8-K events")
st.caption("Open-market insider sales in the 30 days before each 8-K. Heavy discretionary "
           "selling ahead of a red-flag event is worth a closer look; 10b5-1 sales were "
           "scheduled in advance.")
pe = insiders.pre_event_selling(ticker, 30)
if pe.empty:
    st.caption("No 8-K events indexed yet.")
else:
    pe["red_flag"] = pe["red_flag"].map({1: "⚠️", 0: ""})
    st.dataframe(pe, hide_index=True, width="stretch",
                 column_config={"usd_sold": st.column_config.NumberColumn(format="dollar"),
                                "usd_sold_10b5_1": st.column_config.NumberColumn(format="dollar")})

with st.expander("All transactions"):
    st.dataframe(tx, hide_index=True, width="stretch")
