"""
One-time script to ingest the three 10-K PDFs into ChromaDB.
"""
import warnings
warnings.filterwarnings("ignore")

import pdfplumber
from knowledge_base import KnowledgeBaseService
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma
import configure_data as config

FILES = [
    "Alphabet 10K 2024_compressed.pdf",
    "Amazon 10K 2024_compressed.pdf",
    "MSFT 10-K_compressed.pdf",
]

svc = KnowledgeBaseService()

for path in FILES:
    print(f"\nReading: {path}")
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    print(f"  Extracted {len(text):,} characters")
    result = svc.upload_by_str(text, path)
    print(f"  Status: {result}")

# Summary
db = Chroma(
    collection_name=config.collection_name,
    embedding_function=FastEmbedEmbeddings(model_name=config.embedding_model_name),
    persist_directory=config.persist_directory,
)
col = db._collection
print(f"\nTotal chunks in ChromaDB: {col.count()}")
metas = col.get(include=["metadatas"])["metadatas"]
sources = sorted(set(m.get("source", "?") for m in metas))
print("Indexed files:")
for s in sources:
    count = sum(1 for m in metas if m.get("source") == s)
    print(f"  - {s}  ({count} chunks)")
