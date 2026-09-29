"""XBRL financial series, tables, ratios and events for dashboards + the agent."""
from __future__ import annotations

import pandas as pd

from secrag.parsers.xbrl import METRICS, fmt_value
from secrag.store import db


def cik_for(ticker: str) -> int | None:
    df = db.query_df("SELECT cik FROM companies WHERE ticker = ?", (ticker,))
    return int(df["cik"].iloc[0]) if len(df) else None


def series(ticker: str, metrics: list[str] | None = None, period_type: str = "annual") -> pd.DataFrame:
    cik = cik_for(ticker)
    if cik is None:
        return pd.DataFrame()
    metrics = metrics or list(METRICS)
    q = (f"SELECT metric, label, unit, value, period_start, period_end, period_type, frame, form, "
         f"accession FROM xbrl_facts WHERE cik = ? AND metric IN ({','.join('?' * len(metrics))})")
    params: list = [cik, *metrics]
    if period_type == "annual":
        # balance-sheet items are instants; show them alongside annual flows
        q += " AND period_type IN ('annual', 'instant')"
    elif period_type:
        q += " AND period_type IN (?, 'instant')"
        params.append(period_type)
    return db.query_df(q + " ORDER BY period_end", tuple(params))


def table(ticker: str, period_type: str = "annual", n_periods: int = 5,
          metrics: list[str] | None = None) -> pd.DataFrame:
    """Metric x period table (formatted strings), latest periods on the right."""
    df = series(ticker, metrics, period_type)
    if df.empty:
        return df
    flow = df[df["period_type"] == period_type]
    ends = sorted(flow["period_end"].unique())[-n_periods:]
    df = df[df["period_end"].isin(ends)]
    df = df.assign(display=[fmt_value(v, u) for v, u in zip(df["value"], df["unit"])])
    pivot = df.pivot_table(index="label", columns="period_end", values="display", aggfunc="first")
    order = [METRICS[m][0] for m in METRICS if METRICS[m][0] in pivot.index]
    return pivot.reindex(order)


def ratios(ticker: str) -> pd.DataFrame:
    df = series(ticker, ["revenue", "gross_profit", "operating_income", "net_income",
                         "operating_cash_flow", "capex"], "annual")
    if df.empty:
        return df
    w = df[df["period_type"] == "annual"].pivot_table(index="period_end", columns="metric",
                                                      values="value", aggfunc="first")
    out = pd.DataFrame(index=w.index)
    if "revenue" in w:
        out["revenue_growth_%"] = w["revenue"].pct_change() * 100
        for m, name in [("gross_profit", "gross_margin_%"), ("operating_income", "operating_margin_%"),
                        ("net_income", "net_margin_%")]:
            if m in w:
                out[name] = w[m] / w["revenue"] * 100
    if "operating_cash_flow" in w and "capex" in w:
        out["free_cash_flow"] = w["operating_cash_flow"] - w["capex"]
    return out.round(2)


def events(ticker: str, limit: int = 50) -> pd.DataFrame:
    return db.query_df(
        "SELECT filed_date, item, title, red_flag, summary, accession FROM events_8k "
        "WHERE ticker = ? ORDER BY filed_date DESC, item LIMIT ?", (ticker, limit))


def to_context(ticker: str, metrics: list[str] | None = None) -> str:
    parts = []
    annual = table(ticker, "annual", 5, metrics)
    if not annual.empty:
        parts += [f"{ticker} annual financials (SEC XBRL companyfacts, fiscal years by period-end date):",
                  annual.to_markdown()]
    quarterly = table(ticker, "quarterly", 6, metrics)
    if not quarterly.empty:
        parts += [f"{ticker} quarterly financials (discrete quarters):", quarterly.to_markdown()]
    r = ratios(ticker)
    if not r.empty:
        parts += [f"{ticker} derived ratios:", r.tail(5).to_markdown()]
    return "\n\n".join(parts) or f"No XBRL financials indexed for {ticker}."


def events_context(ticker: str) -> str:
    e = events(ticker, 30)
    if e.empty:
        return f"No 8-K events indexed for {ticker}."
    e = e.assign(red_flag=e["red_flag"].map({1: "YES", 0: ""}))
    return f"{ticker} 8-K events (most recent first):\n\n" + e.drop(columns=["summary"]).to_markdown(index=False)
