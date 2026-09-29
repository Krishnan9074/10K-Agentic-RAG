"""
Company lookup and filing discovery on EDGAR.

    resolve_company("AAPL")          -> Company(cik=320193, ticker="AAPL", name="Apple Inc.")
    list_filings(company, forms=...) -> [Filing, ...] newest first
    fetch_primary(filing)            -> raw text of the primary document
    fetch_exhibits(filing, "EX-99")  -> [(type, filename, text), ...]
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from datetime import date
from functools import lru_cache
from typing import Iterable, Optional

from bs4 import BeautifulSoup

from secrag.edgar.client import EdgarClient, default_client

ARCHIVES = "https://www.sec.gov/Archives/edgar/data"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"


@dataclass(frozen=True)
class Company:
    cik: int
    ticker: str
    name: str
    fiscal_year_end: str = ""   # "MMDD"
    sic_description: str = ""


@dataclass(frozen=True)
class Filing:
    cik: int
    ticker: str
    company: str
    form_type: str
    accession: str
    filed_date: str
    report_date: str
    primary_document: str

    @property
    def folder_url(self) -> str:
        return f"{ARCHIVES}/{self.cik}/{self.accession.replace('-', '')}"

    @property
    def primary_url(self) -> str:
        doc = self.primary_document
        # Forms 3/4/5 list the XSL-rendered view ("xslF345X06/form4.xml");
        # the raw ownershipDocument XML lives at the folder root.
        if re.match(r"^xsl[^/]+/", doc):
            doc = doc.split("/", 1)[1]
        return f"{self.folder_url}/{doc}"

    @property
    def index_url(self) -> str:
        return f"{self.folder_url}/{self.accession}-index.htm"

    @property
    def viewer_url(self) -> str:
        """Human-friendly link for citations."""
        if self.primary_url.endswith((".htm", ".html")):
            return f"https://www.sec.gov/ix?doc=/Archives/edgar/data/{self.cik}/" \
                   f"{self.accession.replace('-', '')}/{self.primary_document}"
        return self.index_url

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
#  Company resolution                                                           #
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=1)
def _ticker_table() -> dict:
    data = default_client().get_json(TICKERS_URL) or {}
    by_ticker, by_cik = {}, {}
    for row in data.values():
        entry = (int(row["cik_str"]), row["ticker"].upper(), row["title"])
        by_ticker[entry[1]] = entry
        by_cik.setdefault(entry[0], entry)
    return {"ticker": by_ticker, "cik": by_cik}


def search_companies(query: str, limit: int = 10) -> list[tuple[str, str]]:
    """Fuzzy lookup by ticker or name -> [(ticker, name)]."""
    q = query.strip().upper()
    if not q:
        return []
    table = _ticker_table()["ticker"]
    exact = [(t, e[2]) for t, e in table.items() if t == q]
    starts = [(t, e[2]) for t, e in table.items() if t != q and t.startswith(q)]
    names = [(t, e[2]) for t, e in table.items() if q in e[2].upper() and not t.startswith(q)]
    return (exact + starts + names)[:limit]


def resolve_company(identifier: str | int, client: EdgarClient | None = None) -> Company:
    """Resolve a ticker ('MSFT') or CIK ('789019') to a Company."""
    client = client or default_client()
    ident = str(identifier).strip().upper()
    table = _ticker_table()
    if ident.isdigit():
        cik = int(ident)
        ticker = table["cik"].get(cik, (cik, ident, ""))[1]
    else:
        ident = ident.replace(".", "-")  # BRK.B -> BRK-B (SEC's spelling)
        if ident not in table["ticker"]:
            raise ValueError(f"Unknown ticker {identifier!r}")
        cik, ticker, _ = table["ticker"][ident]

    subs = client.get_json(SUBMISSIONS_URL.format(cik=cik))
    if not subs:
        raise ValueError(f"No EDGAR submissions for CIK {cik}")
    return Company(
        cik=cik,
        ticker=ticker,
        name=subs.get("name", ""),
        fiscal_year_end=subs.get("fiscalYearEnd") or "",
        sic_description=subs.get("sicDescription") or "",
    )


# --------------------------------------------------------------------------- #
#  Filing listing                                                               #
# --------------------------------------------------------------------------- #

def _rows(block: dict) -> Iterable[dict]:
    keys = ("form", "accessionNumber", "filingDate", "reportDate", "primaryDocument")
    cols = [block.get(k, []) for k in keys]
    for vals in zip(*cols):
        yield dict(zip(keys, vals))


def list_filings(
    company: Company,
    forms: Iterable[str],
    *,
    since: Optional[date] = None,
    limits: Optional[dict[str, int]] = None,
    include_amendments: bool = False,
    client: EdgarClient | None = None,
) -> list[Filing]:
    """List a company's filings of the given form types, newest first.

    Walks the paginated submissions history until ``since`` or every
    per-form limit is satisfied.
    """
    client = client or default_client()
    wanted = set(forms)
    if include_amendments:
        wanted |= {f"{f}/A" for f in forms}
    limits = limits or {}
    counts: dict[str, int] = {}
    since_s = since.isoformat() if since else ""

    subs = client.get_json(SUBMISSIONS_URL.format(cik=company.cik)) or {}
    pages = [subs.get("filings", {}).get("recent", {})]
    older = [f["name"] for f in subs.get("filings", {}).get("files", [])]

    out: list[Filing] = []

    def base(form: str) -> str:
        return form[:-2] if form.endswith("/A") else form

    def satisfied() -> bool:
        return all(f in limits and counts.get(f, 0) >= limits[f] for f in forms)

    while pages:
        page = pages.pop(0)
        oldest_on_page = "9999"
        for r in _rows(page):
            oldest_on_page = min(oldest_on_page, r["filingDate"])
            form = r["form"]
            if form not in wanted:
                continue
            if since_s and r["filingDate"] < since_s:
                continue
            b = base(form)
            if b in limits and counts.get(b, 0) >= limits[b]:
                continue
            counts[b] = counts.get(b, 0) + 1
            out.append(Filing(
                cik=company.cik, ticker=company.ticker, company=company.name,
                form_type=form, accession=r["accessionNumber"], filed_date=r["filingDate"],
                report_date=r.get("reportDate") or "", primary_document=r["primaryDocument"],
            ))
        if satisfied() or (since_s and oldest_on_page < since_s):
            break
        if not pages and older:
            nxt = client.get_json(f"https://data.sec.gov/submissions/{older.pop(0)}")
            if nxt:
                pages.append(nxt)

    out.sort(key=lambda f: f.filed_date, reverse=True)
    return out


# --------------------------------------------------------------------------- #
#  Document fetching                                                            #
# --------------------------------------------------------------------------- #

def fetch_primary(filing: Filing, client: EdgarClient | None = None) -> Optional[str]:
    return (client or default_client()).get_text(filing.primary_url)


def list_documents(filing: Filing, client: EdgarClient | None = None) -> list[dict]:
    """Parse the filing index page -> [{seq, description, filename, type, url}]."""
    html = (client or default_client()).get_text(filing.index_url)
    if not html:
        return []
    soup = BeautifulSoup(html, "lxml")
    docs = []
    for table in soup.find_all("table", class_="tableFile"):
        for tr in table.find_all("tr")[1:]:
            tds = tr.find_all("td")
            if len(tds) < 4:
                continue
            a = tds[2].find("a")
            if not a:
                continue
            href = a["href"]
            if href.startswith("/ix?doc="):
                href = href[len("/ix?doc="):]
            docs.append({
                "seq": tds[0].get_text(strip=True),
                "description": tds[1].get_text(strip=True),
                "filename": a.get_text(strip=True),
                "type": tds[3].get_text(strip=True),
                "url": f"https://www.sec.gov{href}" if href.startswith("/") else href,
            })
    return docs


def fetch_exhibits(
    filing: Filing, type_prefix: str = "EX-99", client: EdgarClient | None = None,
) -> list[tuple[str, str, str]]:
    """Fetch exhibit documents (e.g. 8-K press releases) -> [(type, filename, text)]."""
    client = client or default_client()
    out = []
    for doc in list_documents(filing, client):
        if doc["type"].upper().startswith(type_prefix) and doc["filename"].lower().endswith(
            (".htm", ".html", ".txt")
        ):
            text = client.get_text(doc["url"])
            if text:
                out.append((doc["type"], doc["filename"], text))
    return out
