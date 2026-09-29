"""
Split 10-K / 10-Q / 8-K text into their SEC-defined Items.

Works on the output of ``html_to_text`` where headings sit on their own lines.
A heading is accepted only if its title matches the canonical SEC title for
that Item, which filters out in-body cross references ("see Item 7 ...").
When an Item heading appears more than once (TOC + body), the occurrence that
opens the longest section wins -- that is the body, not the TOC line.

Item-title tables adapted from the Herbie SEC_Intelligence section parser.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

TENK_ITEMS = {
    "1": "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "1C": "Cybersecurity",
    "2": "Properties",
    "3": "Legal Proceedings",
    "4": "Mine Safety Disclosures",
    "5": "Market for Registrant's Common Equity",
    "6": "Reserved",  # was "Selected Financial Data" before FY2021
    "7": "Management's Discussion and Analysis",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data",
    "9": "Changes in and Disagreements with Accountants",
    "9A": "Controls and Procedures",
    "9B": "Other Information",
    "9C": "Disclosure Regarding Foreign Jurisdictions",
    "10": "Directors, Executive Officers and Corporate Governance",
    "11": "Executive Compensation",
    "12": "Security Ownership of Certain Beneficial Owners",
    "13": "Certain Relationships and Related Transactions",
    "14": "Principal Accountant Fees and Services",
    "15": "Exhibits and Financial Statement Schedules",
    "16": "Form 10-K Summary",
}
_TENK_ALIASES = {"6": ["Selected Financial Data", "[Reserved]"], "15": ["Exhibit"]}

# 10-Q item numbers collide across Parts, so keys are part-qualified.
TENQ_ITEMS = {
    ("I", "1"): "Financial Statements",
    ("I", "2"): "Management's Discussion and Analysis",
    ("I", "3"): "Quantitative and Qualitative Disclosures About Market Risk",
    ("I", "4"): "Controls and Procedures",
    ("II", "1"): "Legal Proceedings",
    ("II", "1A"): "Risk Factors",
    ("II", "2"): "Unregistered Sales of Equity Securities",
    ("II", "3"): "Defaults Upon Senior Securities",
    ("II", "4"): "Mine Safety Disclosures",
    ("II", "5"): "Other Information",
    ("II", "6"): "Exhibits",
}

EIGHTK_ITEMS = {
    "1.01": "Entry into a Material Definitive Agreement",
    "1.02": "Termination of a Material Definitive Agreement",
    "1.03": "Bankruptcy or Receivership",
    "1.04": "Mine Safety",
    "1.05": "Material Cybersecurity Incidents",
    "2.01": "Completion of Acquisition or Disposition of Assets",
    "2.02": "Results of Operations and Financial Condition",
    "2.03": "Creation of a Direct Financial Obligation",
    "2.04": "Triggering Events That Accelerate or Increase a Direct Financial Obligation",
    "2.05": "Costs Associated with Exit or Disposal Activities",
    "2.06": "Material Impairments",
    "3.01": "Notice of Delisting or Failure to Satisfy a Continued Listing Rule",
    "3.02": "Unregistered Sales of Equity Securities",
    "3.03": "Material Modification to Rights of Security Holders",
    "4.01": "Changes in Registrant's Certifying Accountant",
    "4.02": "Non-Reliance on Previously Issued Financial Statements",
    "5.01": "Changes in Control of Registrant",
    "5.02": "Departure or Election of Directors or Officers; Compensatory Arrangements",
    "5.03": "Amendments to Articles of Incorporation or Bylaws",
    "5.04": "Temporary Suspension of Trading Under Employee Benefit Plans",
    "5.05": "Amendments to the Code of Ethics",
    "5.06": "Change in Shell Company Status",
    "5.07": "Submission of Matters to a Vote of Security Holders",
    "5.08": "Shareholder Director Nominations",
    "6.01": "ABS Informational and Computational Material",
    "7.01": "Regulation FD Disclosure",
    "8.01": "Other Events",
    "9.01": "Financial Statements and Exhibits",
}

# 8-K items that usually signal material bad news / risk.
EIGHTK_RED_FLAGS = {"1.03", "1.05", "2.04", "2.05", "2.06", "3.01", "4.01", "4.02"}


@dataclass
class Section:
    key: str        # stable id, e.g. "item_1a", "part2_item1a", "item_2.02"
    title: str      # human title, e.g. "Item 1A. Risk Factors"
    text: str


_ITEM_LINE = re.compile(r"^item\s+(\d{1,2}(?:\.\d{2})?[a-c]?)\s*[.:\-—–]?\s*(.*)$", re.I)
_PART_LINE = re.compile(r"^part\s+(i{1,3}|iv)\b[.:\-—–\s]*(.*)$", re.I)


def _letters(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.lower())


def _title_matches(rest: str, titles: list[str]) -> bool:
    rest_l = _letters(rest)
    if not rest_l:
        return True  # bare "Item 7." line; title is on the next line
    for t in titles:
        tl = _letters(t)
        n = min(len(tl), 12)
        if rest_l[:n] == tl[:n]:
            return True
    return False


def _heading_hits(text: str, form: str):
    """Yield (offset, key, title) for every acceptable heading line."""
    part = "I"
    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if 0 < len(stripped) <= 200 and "|" not in stripped:
            pm = _PART_LINE.match(stripped)
            if pm and len(stripped) < 80:
                part = pm.group(1).upper()
            im = _ITEM_LINE.match(stripped)
            if im:
                num, rest = im.group(1).upper(), im.group(2)
                hit = _classify(form, part, num, rest)
                if hit:
                    yield (offset, *hit)
        offset += len(line)


def _classify(form: str, part: str, num: str, rest: str):
    if form == "8-K":
        if num in EIGHTK_ITEMS and _title_matches(rest, [EIGHTK_ITEMS[num]]):
            return f"item_{num}", f"Item {num}. {EIGHTK_ITEMS[num]}"
        return None
    if form == "10-Q":
        title = TENQ_ITEMS.get((part, num))
        if title and _title_matches(rest, [title]):
            p = "1" if part == "I" else "2"
            return f"part{p}_item{num.lower()}", f"Part {part}, Item {num}. {title}"
        return None
    title = TENK_ITEMS.get(num)
    if title and _title_matches(rest, [title, *_TENK_ALIASES.get(num, [])]):
        return f"item_{num.lower()}", f"Item {num}. {title}"
    return None


def split_sections(text: str, form_type: str) -> list[Section]:
    """Split a filing's text into Items. Text before the first Item is 'cover'."""
    form = form_type.upper().replace("/A", "")
    form = "10-K" if form.startswith("10-K") else "10-Q" if form.startswith("10-Q") else form
    hits = list(_heading_hits(text, form))
    if not hits:
        return [Section("full_text", "Full document", text)]

    # For each key, keep the occurrence that opens the longest section.
    ordered = sorted(hits)
    spans = []
    for i, (off, key, title) in enumerate(ordered):
        end = ordered[i + 1][0] if i + 1 < len(ordered) else len(text)
        spans.append((off, end, key, title))
    best: dict[str, tuple] = {}
    for span in spans:
        cur = best.get(span[2])
        if cur is None or (span[1] - span[0]) > (cur[1] - cur[0]):
            best[span[2]] = span
    chosen = sorted(best.values())

    sections = []
    first = chosen[0][0]
    if first > 200:
        sections.append(Section("cover", "Cover page & table of contents", text[:first].strip()))
    for i, (off, _end, key, title) in enumerate(chosen):
        end = chosen[i + 1][0] if i + 1 < len(chosen) else len(text)
        body = text[off:end].strip()
        if body:
            sections.append(Section(key, title, body))
    return sections
