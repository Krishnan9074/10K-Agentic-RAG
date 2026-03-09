# 10K Agentic RAG

> An adaptive Retrieval-Augmented Generation (RAG) system for querying SEC 10-K annual reports and custom documents — powered by Groq LLMs, Qdrant vector search, and Streamlit.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://10k-agentic-rag.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-Orchestration-1C3C3C?logo=langchain)](https://langchain.com)
[![Groq](https://img.shields.io/badge/Groq-LLM%20Inference-F54E42)](https://groq.com)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20Search-DC244C)](https://qdrant.tech)

**[Try the live app →](https://10k-agentic-rag.streamlit.app/)**

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Environment Variables](#environment-variables)
  - [Ingest Documents](#ingest-documents)
  - [Run the App](#run-the-app)
- [Usage](#usage)
  - [QA Interface](#qa-interface)
  - [File Upload](#file-upload)
- [Configuration](#configuration)
- [Deployment](#deployment)
- [Security Considerations](#security-considerations)
- [License](#license)

---

## Overview

10K Agentic RAG is an intelligent document Q&A system that lets you ask natural-language questions about SEC 10-K filings from **Alphabet (2024)**, **Amazon (2024)**, and **Microsoft**, as well as any custom documents you upload. It uses an adaptive routing strategy to decide whether a question needs document retrieval or can be answered directly, then verifies answers for hallucinations before presenting them.

---

## Features

| Feature | Description |
|---|---|
| **Adaptive Query Routing** | Automatically classifies each question — routes factual/financial queries to RAG, conversational queries to direct LLM answer |
| **Grounding / Hallucination Check** | Post-generation verification that flags answers not fully supported by retrieved context |
| **Source Citations** | Every RAG answer includes clickable source snippets with document name and page number |
| **Multi-Model Selection** | Choose between Groq-hosted `llama-3.3-70b-versatile`, `llama-3.1-8b-instant`, or `gemma2-9b-it` at runtime |
| **File Upload** | Upload `.txt` or `.pdf` files (up to 10 MB) to extend the knowledge base on the fly |
| **Persistent Chat History** | Conversations are stored per session so context is preserved across page refreshes |
| **Duplicate Detection** | SHA-256 hashing prevents the same document from being indexed twice |
| **Rate Limiting** | 20 requests per minute per session to prevent API abuse |
| **Local Embeddings** | Uses `BAAI/bge-small-en-v1.5` via FastEmbed — no third-party embedding API key required |
| **Pre-loaded 10-K Reports** | Alphabet, Amazon, and Microsoft annual reports are ingested out of the box |

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     Streamlit Frontend                    │
│     ┌──────────────────┐   ┌──────────────────────┐      │
│     │  QA Interface    │   │  File Upload Page     │      │
│     │  (app_qa.py)     │   │ (app_file_uploader.py)│      │
│     └────────┬─────────┘   └──────────┬───────────┘      │
└──────────────┼──────────────────────  ┼ ───────────────── ┘
               │                        │
               ▼                        ▼
┌──────────────────────────────────────────────────────────┐
│                     RAG Service (rag.py)                  │
│                                                           │
│  ① Query Router        ② Retrieval        ③ Answer       │
│  (llama-3.1-8b)        (Qdrant k=6)       (chosen LLM)   │
│        │                    │                  │          │
│        ▼                    ▼                  ▼          │
│  direct_answer?     Vector similarity     ④ Grounding     │
│  vectorstore?       search → docs         check (LLM)     │
└──────────────────────────────────────────────────────────┘
               │                                │
               ▼                                ▼
┌─────────────────────────┐     ┌───────────────────────────┐
│  Qdrant Cloud           │     │  Groq Inference API       │
│  (384-dim cosine index) │     │  (LLaMA 3 / Gemma 2)      │
└─────────────────────────┘     └───────────────────────────┘
               ▲
               │  (ingest)
┌─────────────────────────┐
│  Knowledge Base         │
│  ├─ Alphabet 10-K 2024  │
│  ├─ Amazon 10-K 2024    │
│  ├─ Microsoft 10-K      │
│  ├─ Company Policy      │
│  ├─ Python Basics       │
│  └─ RAG Introduction    │
└─────────────────────────┘
```

### Adaptive RAG Flow

1. **Route** — A lightweight `llama-3.1-8b-instant` model classifies the query as `vectorstore` or `direct_answer`.
2. **Retrieve** — For `vectorstore` queries, the top-6 most relevant document chunks are fetched from Qdrant using cosine similarity.
3. **Answer** — The user-selected model generates an answer strictly grounded in the retrieved context.
4. **Verify** — A grounding-check call confirms whether the answer is fully supported; a warning is shown if not.
5. **Cite** — Source document filename, page number, and a text snippet are surfaced in the UI.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | [Streamlit](https://streamlit.io) |
| **LLM Inference** | [Groq](https://groq.com) — LLaMA 3.3 70B, LLaMA 3.1 8B, Gemma 2 9B |
| **Embeddings** | [FastEmbed](https://github.com/qdrant/fastembed) — `BAAI/bge-small-en-v1.5` (local, 384-dim) |
| **Vector Database** | [Qdrant Cloud](https://qdrant.tech) (cosine similarity) |
| **LLM Orchestration** | [LangChain](https://langchain.com) |
| **PDF Parsing** | [pdfplumber](https://github.com/jsvine/pdfplumber) |
| **Chat History** | File-based JSON store (`chat_history/`) |
| **Local Vector Store** | [ChromaDB](https://www.trychroma.com) (used by file upload page) |

---

## Project Structure

```
.
├── app_qa.py               # Streamlit QA chat interface (main page)
├── app_file_uploader.py    # Streamlit file upload page
├── rag.py                  # Core RAG service: router, retrieval, grounding
├── vector_stores.py        # Qdrant vector store wrapper
├── knowledge_base.py       # Document ingestion & dedup logic
├── ingest_10k.py           # One-time script to load 10-K PDFs into Qdrant
├── history_store.py        # File-based chat history (LangChain compatible)
├── file_history_store.py   # Per-session history helper
├── configure_data.py       # All configuration constants & secret loading
├── requirements.txt        # Python dependencies (pinned)
├── runtime.txt             # Python version for Streamlit Cloud
├── md5.text                # SHA-256 hashes of ingested documents (dedup store)
├── kb_company_policy.txt   # Sample company policy knowledge base doc
├── kb_python_basics.txt    # Sample Python basics knowledge base doc
├── kb_rag_intro.txt        # Sample RAG introduction knowledge base doc
├── chat_history/           # Persistent per-session chat history (JSON)
└── chroma_db/              # Local ChromaDB data (file upload page)
```

---

## Getting Started

### Prerequisites

- Python 3.11 or higher
- A [Qdrant Cloud](https://cloud.qdrant.io) account (free tier works)
- A [Groq](https://console.groq.com) API key (free tier available)

### Installation

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd <repo-directory>

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:

```env
QDRANT_URL=https://<your-cluster-id>.us-east4-0.gcp.cloud.qdrant.io
QDRANT_API_KEY=<your-qdrant-api-key>
GROQ_API_KEY=<your-groq-api-key>
```

| Variable | Description |
|---|---|
| `QDRANT_URL` | Your Qdrant Cloud cluster URL |
| `QDRANT_API_KEY` | Qdrant Cloud API key |
| `GROQ_API_KEY` | Groq API key for LLM inference |

> **Streamlit Cloud**: Add these as [Streamlit Secrets](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management) instead of a `.env` file.

### Ingest Documents

Place your 10-K PDF files in the project root, then run the ingestion script once:

```bash
python ingest_10k.py
```

Expected files (configured in `ingest_10k.py`):
- `Alphabet 10K 2024_compressed.pdf`
- `Amazon 10K 2024_compressed.pdf`
- `MSFT 10-K_compressed.pdf`
- `kb_company_policy.txt`
- `kb_python_basics.txt`
- `kb_rag_intro.txt`

The script chunks and embeds each document, uploads vectors to Qdrant, and uses SHA-256 hashing to skip any previously ingested file.

### Run the App

```bash
# QA Chat Interface
streamlit run app_qa.py

# File Upload Page
streamlit run app_file_uploader.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Usage

### QA Interface

1. **Select a model** from the dropdown (`llama-3.3-70b-versatile` is recommended for best accuracy).
2. **Type your question** in the chat input — e.g.:
   - *"What was Alphabet's total revenue in 2024?"*
   - *"Summarize Amazon's risk factors."*
   - *"Compare Microsoft and Alphabet's cloud revenue."*
3. The system automatically **routes** the query, **retrieves** relevant chunks, and **answers** with citations.
4. Expand **📄 Sources** under any answer to see the exact document and page a fact came from.
5. A **⚠️ Possible hallucination** warning appears if the answer isn't well-supported by the retrieved context.

### File Upload

1. Navigate to the **Information Upload System** page (`app_file_uploader.py`).
2. Upload a `.txt` or `.pdf` file (max **10 MB**).
3. The file is chunked and ingested into the vector store — it becomes immediately queryable in the QA interface.
4. Duplicate uploads are automatically detected and skipped.

---

## Configuration

All tunable parameters live in `configure_data.py`:

| Parameter | Default | Description |
|---|---|---|
| `chunk_size` | `1000` | Characters per document chunk |
| `chunk_overlap` | `100` | Overlap between consecutive chunks |
| `similarity_threshold` | `6` | Number of chunks retrieved per query (top-k) |
| `embedding_model_name` | `BAAI/bge-small-en-v1.5` | FastEmbed local embedding model |
| `chat_model_name` | `llama-3.3-70b-versatile` | Default Groq chat model |
| `collection_name` | `rag` | Qdrant collection name |
| `separators` | `["\n\n", "\n", "。", ...]` | Text split separators (supports CJK) |

---

## Deployment

This app is deployable to [Streamlit Community Cloud](https://streamlit.io/cloud) with zero infrastructure changes.

1. Push your code to a public or private GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo.
3. Set the **main file path** to `app_qa.py`.
4. Add `QDRANT_URL`, `QDRANT_API_KEY`, and `GROQ_API_KEY` under **Secrets**.
5. Deploy — Streamlit Cloud automatically reads `runtime.txt` for the Python version.

> The app is already live at **[https://10k-agentic-rag.streamlit.app/](https://10k-agentic-rag.streamlit.app/)**.

---

## Security Considerations

- **Prompt injection protection**: The RAG system prompt explicitly instructs the LLM to treat retrieved document content as read-only reference material and ignore any instructions embedded within it.
- **File validation**: Uploaded files are verified by both extension and magic bytes (for PDFs) before parsing.
- **File size limits**: Upload size is capped at 10 MB to prevent resource exhaustion.
- **Rate limiting**: Each session is limited to 20 requests per minute.
- **No credentials in code**: All secrets are loaded from environment variables or Streamlit Secrets — never hardcoded.
- **Duplicate prevention**: SHA-256 content hashing prevents re-ingestion of identical documents.

---

## License

This project is provided for educational and demonstration purposes. See [LICENSE](LICENSE) for details.
