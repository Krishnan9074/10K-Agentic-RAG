"""
Structured financials from SEC's XBRL companyfacts API.

Each metric maps to an ordered list of us-gaap concepts (companies tag the
same idea differently, e.g. revenue). We keep only facts SEC assigned a
calendar ``frame`` to -- that is SEC's own de-duplication of the many
comparative restatements -- and classify them as annual / quarterly / instant.

Concept selection adapted from the Herbie SEC_Intelligence XBRL fetcher.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from secrag.edgar.client import EdgarClient, default_client

COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

# metric key -> (label, [concepts in priority order], taxonomy)
METRICS: dict[str, tuple[str, list[str], str]] = {
    "revenue": ("Revenue", [
        "RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues",
        "SalesRevenueNet", "RevenueFromContractWithCustomerIncludingAssessedTax"], "us-gaap"),
    "cost_of_revenue": ("Cost of revenue", [
        "CostOfGoodsAndServicesSold", "CostOfRevenue", "CostOfGoodsSold"], "us-gaap"),
    "gross_profit": ("Gross profit", ["GrossProfit"], "us-gaap"),
    "rnd": ("R&D expense", ["ResearchAndDevelopmentExpense"], "us-gaap"),
    "sga": ("SG&A expense", ["SellingGeneralAndAdministrativeExpense"], "us-gaap"),
    "operating_income": ("Operating income", ["OperatingIncomeLoss"], "us-gaap"),
    "net_income": ("Net income", ["NetIncomeLoss", "ProfitLoss"], "us-gaap"),
    "eps_diluted": ("Diluted EPS", ["EarningsPerShareDiluted"], "us-gaap"),
    "income_tax": ("Income tax expense", ["IncomeTaxExpenseBenefit"], "us-gaap"),
    "operating_cash_flow": ("Operating cash flow", [
        "NetCashProvidedByUsedInOperatingActivities"], "us-gaap"),
    "capex": ("Capital expenditures", ["PaymentsToAcquirePropertyPlantAndEquipment"], "us-gaap"),
    "buybacks": ("Share repurchases", ["PaymentsForRepurchaseOfCommonStock"], "us-gaap"),
    "dividends": ("Dividends paid", ["PaymentsOfDividends", "PaymentsOfDividendsCommonStock"], "us-gaap"),
    "cash": ("Cash & equivalents", [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"], "us-gaap"),
    "total_assets": ("Total assets", ["Assets"], "us-gaap"),
    "total_liabilities": ("Total liabilities", ["Liabilities"], "us-gaap"),
    "long_term_debt": ("Long-term debt", ["LongTermDebtNoncurrent", "LongTermDebt"], "us-gaap"),
    "equity": ("Stockholders' equity", ["StockholdersEquity"], "us-gaap"),
    "shares_outstanding": ("Shares outstanding", ["EntityCommonStockSharesOutstanding"], "dei"),
    "deferred_tax_assets": ("Deferred tax assets (net)", ["DeferredTaxAssetsNet"], "us-gaap"),
    "nol_carryforwards": ("Operating loss carryforwards", ["OperatingLossCarryforwards"], "us-gaap"),
}


@dataclass
class Fact:
    cik: int
    metric: str
    label: str
    concept: str
    unit: str
    value: float
    period_start: str | None
    period_end: str
    period_type: str      # annual | quarterly | instant
    frame: str
    fiscal_year: int | None
    fiscal_period: str
    form: str
    filed: str
    accession: str


def _period_type(frame: str) -> str:
    if frame.endswith("I"):
        return "instant"
    return "quarterly" if "Q" in frame else "annual"


def fetch_facts(cik: int, client: EdgarClient | None = None) -> list[Fact]:
    data = (client or default_client()).get_json(COMPANYFACTS_URL.format(cik=cik))
    if not data:
        return []
    facts_root = data.get("facts", {})
    out: list[Fact] = []
    for metric, (label, concepts, taxonomy) in METRICS.items():
        seen_frames: set[str] = set()
        for concept in concepts:  # earlier concepts win for a given frame
            node = facts_root.get(taxonomy, {}).get(concept)
            if not node:
                continue
            for unit, rows in node.get("units", {}).items():
                for r in rows:
                    frame = r.get("frame")
                    if not frame or frame in seen_frames:
                        continue
                    if not str(r.get("form", "")).startswith(("10-K", "10-Q", "20-F", "40-F")):
                        continue
                    seen_frames.add(frame)
                    out.append(Fact(
                        cik=cik, metric=metric, label=label, concept=f"{taxonomy}:{concept}",
                        unit=unit, value=float(r["val"]), period_start=r.get("start"),
                        period_end=r["end"], period_type=_period_type(frame), frame=frame,
                        fiscal_year=r.get("fy"), fiscal_period=r.get("fp") or "",
                        form=r.get("form", ""), filed=r.get("filed", ""), accession=r.get("accn", ""),
                    ))
    return out


def fmt_value(value: float, unit: str) -> str:
    if unit == "USD":
        a = abs(value)
        sign = "-" if value < 0 else ""
        for div, suf in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
            if a >= div:
                return f"{sign}${a / div:,.2f}{suf}"
        return f"{sign}${a:,.0f}"
    if unit == "USD/shares":
        return f"${value:,.2f}"
    if unit == "shares":
        return f"{value:,.0f} shares"
    return f"{value:,.2f} {unit}"


def facts_to_narratives(company_name: str, ticker: str, facts: list[Fact],
                        since_year: int | None = None) -> list[tuple[str, dict]]:
    """One text block per annual period (+ its balance-sheet instants) for embedding."""
    since_year = since_year or date.today().year - 6
    by_end: dict[str, list[Fact]] = {}
    for f in facts:
        if f.period_type == "annual":
            by_end.setdefault(f.period_end, []).append(f)
    instants = {}
    for f in facts:
        if f.period_type == "instant":
            instants.setdefault(f.period_end, []).append(f)

    out = []
    for end, rows in sorted(by_end.items()):
        if int(end[:4]) < since_year:
            continue
        rows = rows + instants.get(end, [])
        lines = [f"{company_name} ({ticker}) key financials from XBRL for the fiscal year "
                 f"ended {end} (source: SEC companyfacts, as reported in {rows[0].form}):"]
        for f in sorted(rows, key=lambda x: list(METRICS).index(x.metric)):
            lines.append(f"- {f.label}: {fmt_value(f.value, f.unit)}")
        out.append(("\n".join(lines), {
            "period_end": end, "fiscal_year": int(end[:4]),
            "accession": rows[0].accession, "filed_date": rows[0].filed,
        }))
    return out
