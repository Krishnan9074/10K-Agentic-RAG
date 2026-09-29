"""
EDGAR HTML / iXBRL -> clean text.

Unlike a naive get_text(), this:
  * drops the hidden iXBRL header (<ix:header> inside display:none divs),
  * keeps block structure so section headings land on their own lines,
  * flattens tables to "cell | cell | cell" rows, gluing "$" / "%" / ")"
    fragments back onto their numbers so financial statements stay readable.
"""
from __future__ import annotations

import re
import warnings

from bs4 import BeautifulSoup, NavigableString, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

_BLOCK_TAGS = [
    "p", "div", "br", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6",
    "section", "article", "center", "blockquote", "pre", "hr",
]
_HIDDEN_STYLE = re.compile(r"display\s*:\s*none", re.I)
_PAGE_NUMBER_LINE = re.compile(r"^\s*(?:page\s+)?\d{1,3}\s*$", re.I)


def _cell_text(cell) -> str:
    return re.sub(r"\s+", " ", cell.get_text(" ", strip=True).replace("\xa0", " ")).strip()


def _table_to_text(table) -> str:
    lines = []
    for tr in table.find_all("tr"):
        cells = [_cell_text(td) for td in tr.find_all(["td", "th"])]
        cells = [c for c in cells if c]
        merged: list[str] = []
        for c in cells:
            if merged and merged[-1] in ("$", "(", "$(") :
                merged[-1] = merged[-1] + c
            elif merged and c in ("%", ")", ")%", "%)"):
                merged[-1] = merged[-1] + c
            else:
                merged.append(c)
        if merged:
            lines.append(" | ".join(merged))
    return "\n".join(lines)


def html_to_text(html: str) -> str:
    """Render an EDGAR HTML/iXBRL document to structured plain text."""
    if "<" not in html[:2000]:
        return normalize_text(html)  # already plain text (1990s filings)

    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style", "head", "title", "ix:header"]):
        tag.decompose()
    for tag in soup.find_all(style=_HIDDEN_STYLE):
        tag.decompose()

    for table in soup.find_all("table"):
        # Nested tables are rendered by their outermost ancestor.
        if table.find_parent("table") is not None:
            continue
        table.replace_with(NavigableString("\n\n" + _table_to_text(table) + "\n\n"))

    for tag in soup.find_all(_BLOCK_TAGS):
        tag.insert_before("\n")
        tag.insert_after("\n")

    return normalize_text(soup.get_text(""))


def normalize_text(text: str) -> str:
    text = text.replace("\xa0", " ").replace("​", "")
    lines = []
    for line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip()
        if _PAGE_NUMBER_LINE.match(line):
            continue
        lines.append(line)
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()
