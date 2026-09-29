"""
Structure-aware chunking.

Splits on paragraph boundaries, then table rows, then hard character cuts,
and prepends a context header to every chunk so each embedding knows which
company / form / period / section it came from.
"""
from __future__ import annotations

from secrag import config


def _pieces(text: str, size: int) -> list[str]:
    """Break text into units no longer than ``size``, preferring natural breaks."""
    out: list[str] = []
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        if len(para) <= size:
            out.append(para)
            continue
        for line in para.split("\n"):  # tables: one row per line
            while len(line) > size:
                cut = line.rfind(". ", 0, size)
                cut = cut + 1 if cut > size // 2 else size
                out.append(line[:cut].strip())
                line = line[cut:]
            if line.strip():
                out.append(line.strip())
    return out


def chunk_text(text: str, size: int = config.CHUNK_SIZE,
               overlap: int = config.CHUNK_OVERLAP) -> list[str]:
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for piece in _pieces(text, size):
        if cur and cur_len + len(piece) + 2 > size:
            chunks.append("\n\n".join(cur))
            # carry the tail of the previous chunk forward as overlap
            tail: list[str] = []
            tail_len = 0
            for p in reversed(cur):
                if tail_len + len(p) > overlap:
                    break
                tail.insert(0, p)
                tail_len += len(p) + 2
            cur, cur_len = tail, tail_len
        cur.append(piece)
        cur_len += len(piece) + 2
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


def context_header(meta: dict) -> str:
    parts = [meta.get("ticker", ""), meta.get("form_type", "")]
    if meta.get("fiscal_year"):
        parts.append(f"FY{meta['fiscal_year']}")
    if meta.get("report_date"):
        parts.append(f"period {meta['report_date']}")
    parts.append(f"filed {meta.get('filed_date', '')}")
    if meta.get("section_title"):
        parts.append(meta["section_title"])
    return "[" + " | ".join(p for p in parts if p) + "]"
