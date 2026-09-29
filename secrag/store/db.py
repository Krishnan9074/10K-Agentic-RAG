"""
SQLite store for structured SEC data + the ingestion registry.

Vectors answer "what did they say"; this answers "exactly how much / when":
insider trades, XBRL financials and 8-K events are queried with SQL, never
re-derived by the LLM from text chunks.
"""
from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime

import pandas as pd

from secrag import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS companies (
    cik INTEGER PRIMARY KEY,
    ticker TEXT, name TEXT, fiscal_year_end TEXT, sic_description TEXT,
    last_ingested TEXT
);
CREATE TABLE IF NOT EXISTS filings (
    accession TEXT PRIMARY KEY,
    cik INTEGER, ticker TEXT, form_type TEXT, filed_date TEXT, report_date TEXT,
    url TEXT, primary_document TEXT, status TEXT, chunks INTEGER, error TEXT, ingested_at TEXT
);
CREATE INDEX IF NOT EXISTS ix_filings_cik ON filings(cik, form_type, filed_date);
CREATE TABLE IF NOT EXISTS insider_transactions (
    accession TEXT, entry_idx INTEGER, owner_cik TEXT,
    cik INTEGER, ticker TEXT, form_type TEXT, filed_date TEXT,
    owner_name TEXT, owner_role TEXT, is_officer INTEGER, is_director INTEGER,
    is_ten_pct_owner INTEGER, kind TEXT, derivative INTEGER, security_title TEXT,
    transaction_date TEXT, transaction_code TEXT, acquired_disposed TEXT,
    shares REAL, price REAL, value REAL, shares_after REAL, ownership_type TEXT,
    aff10b5_one INTEGER,
    PRIMARY KEY (accession, owner_cik, entry_idx)
);
CREATE INDEX IF NOT EXISTS ix_insider_cik ON insider_transactions(cik, transaction_date);
CREATE TABLE IF NOT EXISTS xbrl_facts (
    cik INTEGER, metric TEXT, frame TEXT, label TEXT, concept TEXT, unit TEXT,
    value REAL, period_start TEXT, period_end TEXT, period_type TEXT,
    fiscal_year INTEGER, fiscal_period TEXT, form TEXT, filed TEXT, accession TEXT,
    PRIMARY KEY (cik, metric, frame)
);
CREATE TABLE IF NOT EXISTS events_8k (
    accession TEXT, item TEXT, cik INTEGER, ticker TEXT, filed_date TEXT,
    title TEXT, red_flag INTEGER, summary TEXT,
    PRIMARY KEY (accession, item)
);
CREATE INDEX IF NOT EXISTS ix_events_cik ON events_8k(cik, filed_date);
"""

_lock = threading.Lock()
_initialized: set[str] = set()


@contextmanager
def connect():
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        if str(config.DB_PATH) not in _initialized:
            with _lock:
                conn.executescript(_SCHEMA)
                _initialized.add(str(config.DB_PATH))
        yield conn
        conn.commit()
    finally:
        conn.close()


def _upsert(conn, table: str, rows: list[dict]) -> None:
    if not rows:
        return
    cols = list(rows[0].keys())
    sql = (f"INSERT OR REPLACE INTO {table} ({', '.join(cols)}) "
           f"VALUES ({', '.join('?' for _ in cols)})")
    conn.executemany(sql, [tuple(r[c] for c in cols) for r in rows])


def upsert(table: str, rows: list[dict]) -> None:
    with connect() as conn:
        _upsert(conn, table, rows)


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ── registry ────────────────────────────────────────────────────────────────

def ingested_accessions(cik: int) -> set[str]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT accession FROM filings WHERE cik = ? AND status = 'ok'", (cik,)
        ).fetchall()
    return {r["accession"] for r in rows}


def companies() -> pd.DataFrame:
    with connect() as conn:
        return pd.read_sql_query(
            "SELECT c.*, (SELECT COUNT(*) FROM filings f WHERE f.cik = c.cik AND f.status='ok') "
            "AS filings FROM companies c ORDER BY ticker", conn)


def known_tickers() -> set[str]:
    with connect() as conn:
        return {r["ticker"] for r in conn.execute("SELECT ticker FROM companies")}


def filings(ticker: str | None = None) -> pd.DataFrame:
    q = "SELECT * FROM filings"
    args: tuple = ()
    if ticker:
        q += " WHERE ticker = ?"
        args = (ticker,)
    with connect() as conn:
        return pd.read_sql_query(q + " ORDER BY filed_date DESC", conn, params=args)


def query_df(sql: str, params: tuple = ()) -> pd.DataFrame:
    with connect() as conn:
        return pd.read_sql_query(sql, conn, params=params)
