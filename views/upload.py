"""Upload your own .txt / .pdf documents into the same vector index."""
import hashlib
import io
from datetime import date

import filetype
import pdfplumber
import streamlit as st

from secrag.pipeline.chunking import chunk_text
from views.common import store

MAX_FILE_SIZE_MB = 10

st.title("📤 Upload documents")
st.caption("Add research notes, transcripts or PDFs. They are chunked, embedded and become "
           "searchable in Ask alongside SEC filings.")

up = st.file_uploader("Text or PDF file", type=["txt", "pdf"])
tag = st.text_input("Tag with a ticker (optional)", placeholder="AAPL").strip().upper()


def extract(uploaded) -> list[tuple[int | None, str]]:
    """Return [(page, text)] so citations can point to a page."""
    data = uploaded.getvalue()
    if uploaded.name.lower().endswith(".txt"):
        try:
            return [(None, data.decode("utf-8"))]
        except UnicodeDecodeError:
            raise ValueError("File is not valid UTF-8 text.")
    kind = filetype.guess(data)
    if kind is None or kind.mime != "application/pdf":
        raise ValueError("File content does not match a valid PDF.")
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        return [(i + 1, p.extract_text() or "") for i, p in enumerate(pdf.pages)]


if up is not None:
    if up.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        st.error(f"File too large. Maximum is {MAX_FILE_SIZE_MB} MB.")
        st.stop()
    try:
        pages = extract(up)
    except ValueError as e:
        st.error(str(e))
        st.stop()
    full = "\n".join(t for _, t in pages)
    if not full.strip():
        st.error("No readable text found.")
        st.stop()
    st.write(f"**{up.name}** · {up.size / 1024:,.1f} KB · {len(full):,} characters")
    with st.expander("Preview"):
        st.text(full[:1500])

    if st.button("Add to knowledge base", type="primary"):
        acc = "upload-" + hashlib.sha256(full.encode()).hexdigest()[:16]
        texts, metas = [], []
        for page, text in pages:
            for chunk in chunk_text(text):
                title = f"{up.name}" + (f" p.{page}" if page else "")
                metas.append({
                    "ticker": tag or "UPLOAD", "cik": 0, "company": tag or "Uploaded document",
                    "form_type": "UPLOAD", "is_amendment": False, "accession": acc,
                    "filed_date": date.today().isoformat(), "report_date": "",
                    "fiscal_year": date.today().year, "section": "upload",
                    "section_title": title, "chunk_index": len(texts), "url": "",
                })
                texts.append(f"[Uploaded: {title}]\n{chunk}")
        with st.spinner(f"Embedding {len(texts)} chunks..."):
            vs = store()
            vs.delete_accession(acc)  # re-uploading the same file replaces it
            vs.upsert(texts, metas)
        st.success(f"Added {len(texts)} chunks. Ask about it on the Ask page"
                   + (f" (mention {tag})." if tag else "."))
