"""
End-to-end ingestion: EDGAR -> parse -> SQLite (structured) + Qdrant (vectors).

    ingest_company("AAPL")                      # everything, sensible defaults
    ingest_company("MSFT", forms=["10-K"], since_years=5)
    refresh_all()                               # pull only new filings for tracked companies

Idempotent: filings already ingested are skipped (registry in SQLite), and
vector ids are deterministic, so a crash mid-run is safe to re-run.
"""
from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Callable, Optional

from secrag import config
from secrag.edgar import Company, Filing, fetch_exhibits, fetch_primary, list_filings, resolve_company
from secrag.parsers import form345, xbrl
from secrag.parsers.html import html_to_text
from secrag.parsers.sections import EIGHTK_RED_FLAGS, split_sections
from secrag.pipeline.chunking import chunk_text, context_header
from secrag.store import db
from secrag.store.vectors import get_store

log = logging.getLogger(__name__)

Progress = Callable[[float, str], None]


@dataclass
class ParsedDoc:
    filing: Filing
    texts: list[str] = field(default_factory=list)
    metas: list[dict] = field(default_factory=list)
    insider_rows: list[dict] = field(default_factory=list)
    event_rows: list[dict] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class IngestReport:
    ticker: str
    company: str
    new_filings: int = 0
    skipped: int = 0
    failed: int = 0
    chunks: int = 0
    xbrl_facts: int = 0
    by_form: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)

    def summary(self) -> str:
        forms = ", ".join(f"{k}: {v}" for k, v in sorted(self.by_form.items())) or "none"
        return (f"{self.ticker}: {self.new_filings} new filings ({forms}), {self.chunks} chunks, "
                f"{self.xbrl_facts} XBRL facts, {self.skipped} already indexed, {self.failed} failed")


# --------------------------------------------------------------------------- #
#  Per-form parsing                                                             #
# --------------------------------------------------------------------------- #

def _base_meta(f: Filing) -> dict:
    period = f.report_date or f.filed_date
    return {
        "ticker": f.ticker, "cik": f.cik, "company": f.company,
        "form_type": f.form_type.replace("/A", ""), "is_amendment": f.form_type.endswith("/A"),
        "accession": f.accession, "filed_date": f.filed_date, "report_date": f.report_date,
        "fiscal_year": int(period[:4]), "url": f.viewer_url,
    }


def _add_section_chunks(doc: ParsedDoc, section_key: str, section_title: str, text: str) -> None:
    base = _base_meta(doc.filing)
    for chunk in chunk_text(text):
        meta = {**base, "section": section_key, "section_title": section_title,
                "chunk_index": len(doc.texts)}
        doc.texts.append(f"{context_header(meta)}\n{chunk}")
        doc.metas.append(meta)


def _parse_periodic(doc: ParsedDoc) -> None:
    raw = fetch_primary(doc.filing)
    if not raw:
        raise ValueError("primary document not found")
    text = html_to_text(raw)
    for sec in split_sections(text, doc.filing.form_type):
        _add_section_chunks(doc, sec.key, sec.title, sec.text)


def _parse_8k(doc: ParsedDoc) -> None:
    f = doc.filing
    raw = fetch_primary(f)
    if not raw:
        raise ValueError("primary document not found")
    for sec in split_sections(html_to_text(raw), f.form_type):
        _add_section_chunks(doc, sec.key, sec.title, sec.text)
        m = re.match(r"item_(\d\.\d\d)$", sec.key)
        if m:
            item = m.group(1)
            body = sec.text.split("\n", 1)[1].strip() if "\n" in sec.text else ""
            doc.event_rows.append({
                "accession": f.accession, "item": item, "cik": f.cik, "ticker": f.ticker,
                "filed_date": f.filed_date, "title": sec.title,
                "red_flag": int(item in EIGHTK_RED_FLAGS), "summary": body[:600],
            })
    # Press releases / investor decks (EX-99.x) carry the real 8-K content.
    for ex_type, filename, ex_raw in fetch_exhibits(f, "EX-99"):
        _add_section_chunks(doc, f"exhibit_{ex_type.lower()}", f"Exhibit {ex_type} ({filename})",
                            html_to_text(ex_raw))


def _parse_ownership(doc: ParsedDoc) -> None:
    f = doc.filing
    raw = fetch_primary(f)
    if not raw:
        raise ValueError("ownership XML not found")
    parsed = form345.parse_form345_xml(raw)
    for row in form345.transaction_rows(parsed):
        doc.insider_rows.append({
            "accession": f.accession, "cik": f.cik, "ticker": f.ticker,
            "form_type": f.form_type, "filed_date": f.filed_date, **row,
            "is_officer": int(row["is_officer"]), "is_director": int(row["is_director"]),
            "is_ten_pct_owner": int(row["is_ten_pct_owner"]), "derivative": int(row["derivative"]),
            "aff10b5_one": int(row["aff10b5_one"]),
        })
    _add_section_chunks(doc, "insider_filing", f"Form {parsed['form_type']} insider filing",
                        form345.narrative(parsed))


_HANDLERS = {"10-K": _parse_periodic, "10-Q": _parse_periodic, "8-K": _parse_8k,
             "3": _parse_ownership, "4": _parse_ownership, "5": _parse_ownership}


def parse_filing(filing: Filing) -> ParsedDoc:
    doc = ParsedDoc(filing)
    handler = _HANDLERS.get(filing.form_type.replace("/A", ""))
    if handler is None:
        doc.error = f"unsupported form {filing.form_type}"
        return doc
    try:
        handler(doc)
    except Exception as e:  # one bad filing must never kill a company ingest
        log.exception("parse failed for %s", filing.accession)
        doc.error = f"{type(e).__name__}: {e}"
    return doc


# --------------------------------------------------------------------------- #
#  Orchestration                                                                #
# --------------------------------------------------------------------------- #

def _ingest_xbrl(company: Company, since_year: int) -> int:
    facts = xbrl.fetch_facts(company.cik)
    if not facts:
        return 0
    db.upsert("xbrl_facts", [{
        "cik": f.cik, "metric": f.metric, "frame": f.frame, "label": f.label,
        "concept": f.concept, "unit": f.unit, "value": f.value, "period_start": f.period_start,
        "period_end": f.period_end, "period_type": f.period_type, "fiscal_year": f.fiscal_year,
        "fiscal_period": f.fiscal_period, "form": f.form, "filed": f.filed, "accession": f.accession,
    } for f in facts])

    texts, metas = [], []
    for text, m in xbrl.facts_to_narratives(company.name, company.ticker, facts, since_year):
        acc = f"xbrl-{company.cik}-{m['period_end']}"
        texts.append(text)
        metas.append({
            "ticker": company.ticker, "cik": company.cik, "company": company.name,
            "form_type": "XBRL", "is_amendment": False, "accession": acc,
            "filed_date": m["filed_date"], "report_date": m["period_end"],
            "fiscal_year": m["fiscal_year"], "section": "financial_summary",
            "section_title": "XBRL key financials", "chunk_index": 0,
            "url": f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={company.cik}",
        })
    get_store().upsert(texts, metas)
    return len(facts)


def ingest_company(
    identifier: str | int,
    *,
    forms: list[str] | None = None,
    since_years: int = config.DEFAULT_SINCE_YEARS,
    limits: dict[str, int] | None = None,
    include_xbrl: bool = True,
    include_amendments: bool = False,
    progress: Progress | None = None,
    workers: int = 4,
) -> IngestReport:
    progress = progress or (lambda frac, msg: log.info("[%3.0f%%] %s", frac * 100, msg))
    forms = forms or config.DEFAULT_FORMS
    limits = {**config.DEFAULT_LIMITS, **(limits or {})}

    progress(0.0, f"Resolving {identifier} on EDGAR...")
    company = resolve_company(identifier)
    report = IngestReport(ticker=company.ticker, company=company.name)
    db.upsert("companies", [{
        "cik": company.cik, "ticker": company.ticker, "name": company.name,
        "fiscal_year_end": company.fiscal_year_end, "sic_description": company.sic_description,
        "last_ingested": db.now(),
    }])

    since = date.today() - timedelta(days=365 * since_years)
    store = get_store()

    if include_xbrl:
        progress(0.03, f"Fetching XBRL financials for {company.name}...")
        try:
            report.xbrl_facts = _ingest_xbrl(company, since.year - 1)
        except Exception as e:
            log.exception("XBRL ingest failed")
            report.errors.append(f"XBRL: {e}")

    progress(0.08, "Listing filings...")
    all_filings = list_filings(company, forms, since=since, limits=limits,
                               include_amendments=include_amendments)
    done = db.ingested_accessions(company.cik)
    todo = [f for f in all_filings if f.accession not in done]
    report.skipped = len(all_filings) - len(todo)
    if not todo:
        progress(1.0, f"{company.ticker} is up to date ({report.skipped} filings indexed).")
        return report

    progress(0.1, f"Parsing {len(todo)} new filings...")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(parse_filing, f): f for f in todo}
        for n, fut in enumerate(as_completed(futures), 1):
            doc = fut.result()
            f = doc.filing
            status, err = "ok", None
            if doc.error:
                status, err = "error", doc.error
                report.failed += 1
                report.errors.append(f"{f.form_type} {f.accession}: {doc.error}")
            else:
                try:
                    store.delete_accession(f.accession)
                    report.chunks += store.upsert(doc.texts, doc.metas)
                    db.upsert("insider_transactions", doc.insider_rows)
                    db.upsert("events_8k", doc.event_rows)
                    report.new_filings += 1
                    base = f.form_type.replace("/A", "")
                    report.by_form[base] = report.by_form.get(base, 0) + 1
                except Exception as e:
                    log.exception("indexing failed for %s", f.accession)
                    status, err = "error", f"index: {e}"
                    report.failed += 1
            db.upsert("filings", [{
                "accession": f.accession, "cik": f.cik, "ticker": f.ticker,
                "form_type": f.form_type, "filed_date": f.filed_date, "report_date": f.report_date,
                "url": f.viewer_url, "primary_document": f.primary_document, "status": status, "chunks": len(doc.texts), "error": err,
                "ingested_at": db.now(),
            }])
            progress(0.1 + 0.9 * n / len(todo), f"[{n}/{len(todo)}] {f.form_type} filed {f.filed_date}")

    progress(1.0, report.summary())
    return report


def refresh_all(progress: Progress | None = None) -> list[IngestReport]:
    """Pull new filings for every company already in the database."""
    reports = []
    for ticker in sorted(db.known_tickers()):
        reports.append(ingest_company(ticker, progress=progress))
    return reports
