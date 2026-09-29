"""
Live end-to-end test against SEC EDGAR -> SQLite -> embedded Qdrant.
Opt-in (network + ~1 min):  RUN_LIVE=1 pytest tests/test_live_edgar.py -s
"""
import os

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("RUN_LIVE"), reason="set RUN_LIVE=1")


def test_ingest_and_search(tmp_path, monkeypatch):
    from secrag import config
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "t.sqlite3")
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(config, "QDRANT_LOCAL_PATH", tmp_path / "qdrant")
    monkeypatch.setattr(config, "qdrant_url", lambda: "")

    from secrag.analytics import financials, insiders
    from secrag.pipeline.ingest import ingest_company
    from secrag.store import db
    from secrag.store.vectors import get_store

    r = ingest_company("MSFT", limits={"10-K": 1, "10-Q": 1, "8-K": 1, "3": 1, "4": 3})
    assert r.failed == 0, r.errors
    assert r.by_form.get("10-K") == 1 and r.xbrl_facts > 100

    hits = get_store().search("risks from artificial intelligence", tickers=["MSFT"],
                              form_types=["10-K"], sections=["item_1a"], k=3)
    assert hits and all(h.meta["section"] == "item_1a" for h in hits)
    assert not financials.table("MSFT").empty
    cik = int(db.query_df("SELECT cik FROM companies WHERE ticker='MSFT'")["cik"].iloc[0])
    assert len(db.ingested_accessions(cik)) == r.new_filings
    insiders.to_context("MSFT")  # must not raise

    again = ingest_company("MSFT", limits={"10-K": 1, "10-Q": 1, "8-K": 1, "3": 1, "4": 3})
    assert again.new_filings == 0 and again.skipped >= 3  # idempotent
