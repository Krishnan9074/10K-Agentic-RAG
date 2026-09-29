"""
Insider-trading analytics over Form 3/4 data.

Highlights:
  * per-insider open-market buy/sell totals (codes P/S -- the discretionary
    trades; grants, tax withholding and option exercises are noise),
  * 10b5-1 plan share of selling (pre-scheduled vs discretionary),
  * "pre-event selling": open-market insider sales in the N days BEFORE an
    8-K red-flag or earnings event -- the classic insider-signal screen.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from secrag.store import db


def transactions(ticker: str, days: int | None = 365) -> pd.DataFrame:
    sql = "SELECT * FROM insider_transactions WHERE ticker = ? AND kind = 'transaction'"
    params: list = [ticker]
    if days:
        sql += " AND transaction_date >= ?"
        params.append((date.today() - timedelta(days=days)).isoformat())
    return db.query_df(sql + " ORDER BY transaction_date DESC", tuple(params))


def summary_by_insider(ticker: str, days: int | None = 365) -> pd.DataFrame:
    df = transactions(ticker, days)
    if df.empty:
        return df
    om = df[df["transaction_code"].isin(["P", "S"]) & (df["derivative"] == 0)].copy()
    rows = []
    for (name, role), g in df.groupby(["owner_name", "owner_role"]):
        o = om[(om["owner_name"] == name)]
        buys, sells = o[o["transaction_code"] == "P"], o[o["transaction_code"] == "S"]
        latest = g.sort_values("transaction_date").dropna(subset=["shares_after"])
        rows.append({
            "insider": name, "role": role,
            "open_mkt_buys_shares": buys["shares"].sum(),
            "open_mkt_buys_usd": buys["value"].sum(),
            "open_mkt_sells_shares": sells["shares"].sum(),
            "open_mkt_sells_usd": sells["value"].sum(),
            "pct_sells_10b5_1": (100 * sells["aff10b5_one"].mean()) if len(sells) else None,
            "last_trade": g["transaction_date"].max(),
            "holdings_after_last": latest["shares_after"].iloc[-1] if len(latest) else None,
            "filings": g["accession"].nunique(),
        })
    out = pd.DataFrame(rows)
    out["net_usd"] = out["open_mkt_buys_usd"] - out["open_mkt_sells_usd"]
    return out.sort_values("open_mkt_sells_usd", ascending=False)


def pre_event_selling(ticker: str, days_before: int = 30) -> pd.DataFrame:
    """Open-market insider sales in the window before each 8-K event."""
    return db.query_df(
        """
        SELECT e.filed_date AS event_date, e.item, e.title, e.red_flag,
               COUNT(t.accession) AS sell_trades,
               COUNT(DISTINCT t.owner_name) AS sellers,
               COALESCE(SUM(t.shares), 0) AS shares_sold,
               COALESCE(SUM(t.value), 0) AS usd_sold,
               COALESCE(SUM(CASE WHEN t.aff10b5_one = 1 THEN t.value END), 0) AS usd_sold_10b5_1
        FROM events_8k e
        LEFT JOIN insider_transactions t
          ON t.ticker = e.ticker
         AND t.transaction_code = 'S' AND t.derivative = 0
         AND t.transaction_date <  e.filed_date
         AND t.transaction_date >= date(e.filed_date, ?)
        WHERE e.ticker = ? AND e.item NOT IN ('9.01', '7.01')
        GROUP BY e.accession, e.item
        ORDER BY e.filed_date DESC
        """,
        (f"-{int(days_before)} days", ticker),
    )


def to_context(ticker: str, days: int = 365, max_rows: int = 25) -> str:
    """Compact markdown summary the LLM can cite."""
    s = summary_by_insider(ticker, days)
    if s.empty:
        return f"No Form 3/4 insider transactions indexed for {ticker} in the last {days} days."
    t = transactions(ticker, days).head(max_rows)
    cols = ["transaction_date", "owner_name", "owner_role", "transaction_code", "acquired_disposed",
            "shares", "price", "value", "shares_after", "aff10b5_one", "accession"]
    parts = [
        f"Insider activity for {ticker}, last {days} days (SEC Forms 3/4). "
        "Codes: P=open-market buy, S=open-market sale, A=grant, M=option exercise, "
        "F=shares withheld for tax, G=gift.",
        "Per-insider open-market totals:",
        s.round(2).to_markdown(index=False),
        f"Most recent {len(t)} transactions:",
        t[cols].round(2).to_markdown(index=False),
    ]
    pe = pre_event_selling(ticker)
    pe = pe[pe["sell_trades"] > 0]
    if not pe.empty:
        parts += ["Open-market insider sales in the 30 days before 8-K events:",
                  pe.head(10).round(0).to_markdown(index=False)]
    return "\n\n".join(parts)
