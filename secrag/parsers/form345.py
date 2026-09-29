"""
Form 3 / 4 / 5 (insider ownership) ownershipDocument XML parser.

Handles every schema version (X0202 ... X0609), SGML-wrapped or bare XML,
namespaces, footnote-referenced values, joint filers, and both derivative
and non-derivative tables (holdings for Form 3, transactions for Forms 4/5).

Core parsing ported from the Herbie SEC_Intelligence ``form345_parser``.
Adds ``transaction_rows`` (flat rows for SQL) and ``narrative`` (plain-English
text for embedding, so the RAG can answer "did the CFO sell in August?").
"""
from __future__ import annotations

import re
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Optional
from xml.etree import ElementTree as ET

TRANSACTION_CODES = {
    "P": "Open market or private purchase",
    "S": "Open market or private sale",
    "A": "Grant, award, or other acquisition",
    "D": "Disposition to the issuer",
    "F": "Payment of exercise price or tax liability by delivering securities",
    "I": "Discretionary transaction",
    "M": "Exercise or conversion of derivative security",
    "C": "Conversion of derivative security",
    "E": "Expiration of short derivative position",
    "H": "Expiration (or cancellation) of long derivative position",
    "O": "Exercise of out-of-the-money derivative",
    "X": "Exercise of in-the-money or at-the-money derivative",
    "G": "Bona fide gift",
    "L": "Small acquisition",
    "W": "Acquisition or disposition by will or laws of descent",
    "Z": "Deposit into or withdrawal from voting trust",
    "J": "Other acquisition or disposition",
    "K": "Equity swap or similar instrument",
    "U": "Disposition pursuant to tender of shares in change of control",
    "V": "Transaction voluntarily reported earlier than required",
}

# Codes that reflect a discretionary economic decision (the "signal" trades).
OPEN_MARKET_CODES = {"P", "S"}


# ── XML helpers ─────────────────────────────────────────────────────────────

def _text(el: Optional[ET.Element], path: str, default: str = "") -> str:
    if el is None:
        return default
    node = el.find(path)
    if node is not None and node.text:
        return node.text.strip()
    return default


def _value(el: Optional[ET.Element], path: str, default: str = "") -> str:
    """Form 3/4/5 wraps most data as <x><value>..</value></x>, or a footnoteId."""
    if el is None:
        return default
    node = el.find(path)
    if node is None:
        return default
    val = node.find("value")
    if val is not None and val.text:
        return val.text.strip()
    if node.text and node.text.strip():
        return node.text.strip()
    fn = node.find("footnoteId")
    if fn is not None:
        return f"(footnote:{fn.get('id', '')})"
    return default


def _bool(val: str) -> bool:
    return val.lower() in ("1", "true") if val else False


def _dec(val: str) -> Optional[Decimal]:
    if not val or val.startswith("(footnote:"):
        return None
    cleaned = re.sub(r"[,$]", "", val)
    cleaned = re.sub(r"\s*\(\d+\)\s*$", "", cleaned).strip()
    if not cleaned:
        return None
    try:
        return Decimal(cleaned)
    except InvalidOperation:
        return None


def _date(val: str) -> Optional[str]:
    if not val or val.startswith("(footnote:"):
        return None
    val = val.strip()
    if re.match(r"^\d{4}-\d{2}-\d{2}", val):
        return val[:10]
    m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{2,4})$", val)
    if m:
        mo, d, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if y < 100:
            y += 2000 if y < 50 else 1900
        try:
            return date(y, mo, d).isoformat()
        except ValueError:
            return None
    return None


def _extract_xml(text: str) -> Optional[str]:
    """Strip the SGML/<XML> wrapper around an ownershipDocument if present."""
    s = text.strip()
    if s.startswith("<?xml") or s.startswith("<ownershipDocument"):
        return s
    m = re.search(r"<XML>\s*(.*?)\s*</XML>", text, re.DOTALL | re.IGNORECASE)
    if m and "<ownershipDocument" in m.group(1):
        return re.sub(r"^<\?xml[^?]*\?>\s*", "", m.group(1).strip())
    m = re.search(r"(<ownershipDocument\b.*?</ownershipDocument>)", text, re.DOTALL)
    return m.group(1) if m else None


# ── Table parsers ───────────────────────────────────────────────────────────

def _ownership(el) -> dict:
    return {
        "shares_after": _dec(_value(el, "postTransactionAmounts/sharesOwnedFollowingTransaction")),
        "ownership_type": _value(el, "ownershipNature/directOrIndirectOwnership"),
        "nature_of_ownership": _value(el, "ownershipNature/natureOfOwnership"),
    }


def _coding(el) -> dict:
    coding = el.find("transactionCoding")
    return {
        "transaction_code": _text(coding, "transactionCode"),
        "equity_swap": _bool(_text(coding, "equitySwapInvolved")),
    }


def _amounts(el) -> dict:
    return {
        "shares": _dec(_value(el, "transactionAmounts/transactionShares")),
        "price_per_share": _dec(_value(el, "transactionAmounts/transactionPricePerShare")),
        "acquired_disposed": _value(el, "transactionAmounts/transactionAcquiredDisposedCode"),
    }


def _derivative_fields(el) -> dict:
    return {
        "conversion_or_exercise_price": _dec(_value(el, "conversionOrExercisePrice")),
        "exercise_date": _date(_value(el, "exerciseDate")),
        "expiration_date": _date(_value(el, "expirationDate")),
        "underlying_title": _value(el, "underlyingSecurity/underlyingSecurityTitle"),
        "underlying_shares": _dec(_value(el, "underlyingSecurity/underlyingSecurityShares")),
    }


def _parse_entry(el, *, derivative: bool, holding: bool) -> dict:
    row = {
        "security_title": _value(el, "securityTitle"),
        "derivative": derivative,
        "kind": "holding" if holding else "transaction",
        **_ownership(el),
    }
    if not holding:
        row["transaction_date"] = _date(_value(el, "transactionDate"))
        row.update(_coding(el))
        row.update(_amounts(el))
    if derivative:
        row.update(_derivative_fields(el))
    return row


def _footnotes(root) -> dict[str, str]:
    out = {}
    block = root.find("footnotes")
    if block is None:
        return out
    for fn in block.findall("footnote"):
        out[fn.get("id", "")] = "".join(fn.itertext()).strip()
    return out


def _owners(root) -> list[dict]:
    owners = []
    for o in root.findall("reportingOwner"):
        oid = o.find("reportingOwnerId")
        rel = o.find("reportingOwnerRelationship")
        owners.append({
            "name": _text(oid, "rptOwnerName"),
            "cik": _text(oid, "rptOwnerCik").lstrip("0"),
            "is_director": _bool(_text(rel, "isDirector")),
            "is_officer": _bool(_text(rel, "isOfficer")),
            "is_ten_pct_owner": _bool(_text(rel, "isTenPercentOwner")),
            "is_other": _bool(_text(rel, "isOther")),
            "officer_title": _text(rel, "officerTitle"),
            "other_text": _text(rel, "otherText"),
        })
    return owners


# ── Public API ──────────────────────────────────────────────────────────────

def parse_form345_xml(xml_text: str) -> dict:
    """Parse a Form 3/4/5 document. Raises ValueError if it is not one."""
    clean = _extract_xml(xml_text)
    if clean is None:
        raise ValueError("No ownershipDocument found")
    try:
        root = ET.fromstring(clean)
    except ET.ParseError:
        root = ET.fromstring(re.sub(r'\sxmlns="[^"]*"', "", clean))
    if "}" in root.tag:  # strip default namespace
        ns = root.tag[: root.tag.index("}") + 1]
        for el in root.iter():
            if isinstance(el.tag, str) and el.tag.startswith(ns):
                el.tag = el.tag[len(ns):]
    if root.tag != "ownershipDocument":
        raise ValueError(f"Root element is {root.tag!r}, expected ownershipDocument")

    entries: list[dict] = []
    nd = root.find("nonDerivativeTable")
    if nd is not None:
        entries += [_parse_entry(e, derivative=False, holding=False) for e in nd.findall("nonDerivativeTransaction")]
        entries += [_parse_entry(e, derivative=False, holding=True) for e in nd.findall("nonDerivativeHolding")]
    dt = root.find("derivativeTable")
    if dt is not None:
        entries += [_parse_entry(e, derivative=True, holding=False) for e in dt.findall("derivativeTransaction")]
        entries += [_parse_entry(e, derivative=True, holding=True) for e in dt.findall("derivativeHolding")]

    issuer = root.find("issuer")
    return {
        "form_type": _text(root, "documentType"),
        "schema_version": _text(root, "schemaVersion"),
        "period_of_report": _date(_text(root, "periodOfReport")),
        "issuer": {
            "name": _text(issuer, "issuerName"),
            "cik": _text(issuer, "issuerCik").lstrip("0"),
            "ticker": _text(issuer, "issuerTradingSymbol"),
        },
        "reporting_owners": _owners(root),
        "aff10b5_one": _bool(_text(root, "aff10b5One")),
        "no_securities_owned": _bool(_text(root, "noSecuritiesOwned")),
        "entries": entries,
        "footnotes": _footnotes(root),
        "remarks": _text(root, "remarks"),
    }


def owner_role(owner: dict) -> str:
    roles = []
    if owner.get("is_officer"):
        roles.append(owner.get("officer_title") or "Officer")
    if owner.get("is_director"):
        roles.append("Director")
    if owner.get("is_ten_pct_owner"):
        roles.append("10% Owner")
    if owner.get("is_other") and owner.get("other_text"):
        roles.append(owner["other_text"])
    return ", ".join(roles) or "Insider"


def transaction_rows(parsed: dict) -> list[dict]:
    """Flatten into one row per (owner, entry) for SQL storage."""
    rows = []
    owners = parsed["reporting_owners"] or [{"name": "", "cik": ""}]
    for owner in owners:
        for i, e in enumerate(parsed["entries"]):
            shares = e.get("shares")
            price = e.get("price_per_share")
            rows.append({
                "entry_idx": i,
                "owner_cik": owner.get("cik", ""),
                "owner_name": owner.get("name", ""),
                "owner_role": owner_role(owner),
                "is_officer": owner.get("is_officer", False),
                "is_director": owner.get("is_director", False),
                "is_ten_pct_owner": owner.get("is_ten_pct_owner", False),
                "kind": e["kind"],
                "derivative": e["derivative"],
                "security_title": e.get("security_title", ""),
                "transaction_date": e.get("transaction_date") or parsed.get("period_of_report"),
                "transaction_code": e.get("transaction_code", ""),
                "acquired_disposed": e.get("acquired_disposed", ""),
                "shares": float(shares) if shares is not None else None,
                "price": float(price) if price is not None else None,
                "value": float(shares * price) if shares is not None and price is not None else None,
                "shares_after": float(e["shares_after"]) if e.get("shares_after") is not None else None,
                "ownership_type": e.get("ownership_type", ""),
                "aff10b5_one": parsed.get("aff10b5_one", False),
            })
    return rows


def _fmt_num(x) -> str:
    if x is None:
        return "an unreported number of"
    x = float(x)
    return f"{x:,.0f}" if x == int(x) else f"{x:,.4f}".rstrip("0")


def narrative(parsed: dict) -> str:
    """Plain-English rendering of the filing for semantic search."""
    iss = parsed["issuer"]
    form = parsed["form_type"]
    lines = []
    owners = parsed["reporting_owners"]
    who = "; ".join(f"{o['name']} ({owner_role(o)})" for o in owners) or "Unknown insider"
    head = {"3": "Initial statement of beneficial ownership (Form 3)",
            "4": "Statement of changes in beneficial ownership (Form 4)",
            "5": "Annual statement of beneficial ownership (Form 5)"}.get(form, f"Form {form}")
    lines.append(f"{head} for {iss['name']} ({iss['ticker']}) filed by {who}. "
                 f"Period of report: {parsed.get('period_of_report')}.")
    if parsed.get("aff10b5_one"):
        lines.append("The reported transactions were made under a Rule 10b5-1 trading plan.")
    if parsed.get("no_securities_owned"):
        lines.append("The reporting person owns no securities of the issuer.")

    for e in parsed["entries"]:
        sec = e.get("security_title") or "securities"
        own = "directly" if e.get("ownership_type") == "D" else \
            f"indirectly ({e.get('nature_of_ownership') or 'indirect'})"
        after = e.get("shares_after")
        if e["kind"] == "holding":
            if e["derivative"] and e.get("underlying_title"):
                qty = f"{_fmt_num(after)} " if after is not None else ""
                s = (f"Holds {qty}{sec} {own} covering {_fmt_num(e.get('underlying_shares'))} "
                     f"{e['underlying_title']}")
                if e.get("conversion_or_exercise_price"):
                    s += f", exercise price ${float(e['conversion_or_exercise_price']):,.2f}"
                if e.get("expiration_date"):
                    s += f", expiring {e['expiration_date']}"
                s += "."
            else:
                s = f"Holds {_fmt_num(after)} {sec} {own}."
        else:
            code = e.get("transaction_code", "")
            verb = "acquired" if e.get("acquired_disposed") == "A" else "disposed of"
            price = e.get("price_per_share")
            s = (f"On {e.get('transaction_date')}, {verb} {_fmt_num(e.get('shares'))} {sec}"
                 f"{f' at ${float(price):,.2f} per share' if price else ''}"
                 f" (code {code}: {TRANSACTION_CODES.get(code, 'other')})."
                 f" Holdings after transaction: {_fmt_num(after)} {own}.")
        lines.append(s)

    if parsed["footnotes"]:
        lines.append("Footnotes: " + " ".join(f"[{k}] {v}" for k, v in parsed["footnotes"].items()))
    if parsed.get("remarks"):
        lines.append(f"Remarks: {parsed['remarks']}")
    return "\n".join(lines)
