"""
One-time script to ingest the three 10-K PDFs into Qdrant.
Run after setting QDRANT_URL and QDRANT_API_KEY in .env
"""
import warnings
warnings.filterwarnings("ignore")

import pdfplumber
from knowledge_base import KnowledgeBaseService
from qdrant_client import QdrantClient
import configure_data as config

FILES = [
    "Alphabet 10K 2024_compressed.pdf",
    "Amazon 10K 2024_compressed.pdf",
    "MSFT 10-K_compressed.pdf",
    "kb_company_policy.txt",
    "kb_python_basics.txt",
    "kb_rag_intro.txt",
]

svc = KnowledgeBaseService()

for path in FILES:
    print(f"\nReading: {path}")
    if path.endswith(".pdf"):
        text = ""
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
    else:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    print(f"  Extracted {len(text):,} characters")
    result = svc.upload_by_str(text, path)
    print(f"  Status: {result}")

# Summary
client = QdrantClient(url=config.qdrant_url, api_key=config.qdrant_api_key)
info = client.get_collection(config.collection_name)
print(f"\nTotal vectors in Qdrant: {info.points_count}")
