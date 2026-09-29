# 10K Agentic RAG

> Ask anything about any US public company. It pulls **10-K, 10-Q, 8-K, Form 3, Form 4
> and XBRL financials** straight from SEC EDGAR, parses them into sections and structured
> tables, indexes them in Qdrant + SQLite, and answers with an agent that cites every claim.
> No data files, no manual ingestion.

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://python.org)
[![OpenRouter](https://img.shields.io/badge/LLM-OpenRouter-6566F1)](https://openrouter.ai)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20Search-DC244C)](https://qdrant.tech)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)

---

## What it does

| | |
|---|---|
| **Zero-touch data** | Ask about any company and it is fetched from EDGAR on the fly. On first launch the watchlist (`AAPL, MSFT, GOOGL, AMZN` by default) is ingested in the background, and every tracked company is refreshed every 12h. |
| **5 SEC forms + XBRL** | **10-K** and **10-Q** split into their SEC Items (Risk Factors, MD&A, Financial Statements…), **8-K** split by Item *plus* its EX-99 press releases, **Form 3/4** insider XML parsed into transactions and holdings, **XBRL companyfacts** into exact financial series. |
| **Agentic retrieval** | A planner extracts tickers, forms, sections, years and tools from the question, then does multi-query, per-company, soft-filtered vector search and calls SQL tools for exact numbers. |
| **Exact numbers, not guesses** | Revenue, margins, FCF, EPS etc. come from XBRL tables and insider stats from SQL, so the LLM doesn't read figures off text chunks. |
| **Citations + verification** | Every claim is cited `[S#]` (filing text, links to EDGAR) or `[T#]` (SQL tool). A second model then lists any unsupported claims. |
| **Insider Radar** | Open-market buys and sells by insider, share of sales under 10b5-1 plans, a trade timeline, and **insider selling in the 30 days before each 8-K**. |
| **Filing Diff** | Paragraph-level year-over-year diff of any Item (new, removed and *reworded* risk factors) with an AI "what changed" summary. |
| **Financials & Events** | Compare up to 4 companies on any metric, statements, derived ratios, 8-K timeline with red-flag items (impairments, restatements, auditor changes, defaults, cyber incidents). |
| **Bring your own docs** | Upload PDFs or text (transcripts, notes) into the same index. |

## Architecture

```
            ┌──────────────── SEC EDGAR (rate-limited ≤8 req/s, disk-cached) ───────────────┐
            │ submissions API · filing index · primary docs · EX-99 exhibits · companyfacts  │
            └───────────────────────────────────┬────────────────────────────────────────────┘
                                                ▼
 secrag/parsers      html.py      iXBRL/HTML → text (tables kept as "a | b | c" rows)
                     sections.py  10-K / 10-Q / 8-K → SEC Items (TOC + cross-ref safe)
                     form345.py   Form 3/4/5 ownershipDocument XML → rows + narrative
                     xbrl.py      companyfacts → deduplicated annual/quarterly/instant facts
                                                │
 secrag/pipeline     ingest.py    parse → chunk (context headers) → embed → upsert
                     jobs.py      background queue · watchlist bootstrap · auto-refresh
                        ┌───────────────────────┴───────────────────────┐
                        ▼                                               ▼
     Qdrant (Cloud or embedded)                              SQLite
     chunks + payload: ticker, form, FY,           filings registry · insider_transactions
     section, accession, EDGAR url                  xbrl_facts · events_8k · companies
                        └───────────────────────┬───────────────────────┘
                                                ▼
 secrag/agent/rag.py   plan (fast model) → auto-ingest → retrieve + SQL tools
                       → answer (chosen model, streamed, cited) → grounding check
                                                ▼
 Streamlit  app.py → Ask · Financials & Events · Insider Radar · Filing Diff · Companies · Upload
 CLI        python -m secrag ingest | refresh | watch | status | ask
```

## Quick start

```bash
git clone https://github.com/Krishnan9074/10K-Agentic-RAG.git
cd 10K-Agentic-RAG
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # add OPENROUTER_API_KEY and your SEC_USER_AGENT email
streamlit run app.py
```

That's it. No Qdrant server is needed (an embedded instance lives in `./data/qdrant`), and the
watchlist starts ingesting as soon as the app opens. Dashboards and ingestion work without an
API key; chat, summaries and grounding checks need one.

### Environment

| Variable | Required | Default |
|---|---|---|
| `OPENROUTER_API_KEY` | for chat/summaries | – |
| `SEC_USER_AGENT` | recommended (SEC asks for a contact email) | placeholder |
| `OPENROUTER_MODEL` | no | `anthropic/claude-sonnet-5.5` |
| `OPENROUTER_FAST_MODEL` | no (planner + fact-checker) | `google/gemini-3.8-flash` |
| `QDRANT_URL`, `QDRANT_API_KEY` | no (use Qdrant Cloud) | embedded Qdrant |
| `SECRAG_WATCHLIST` | no | `AAPL,MSFT,GOOGL,AMZN` |
| `SECRAG_AUTO_REFRESH_HOURS` | no (`0` disables) | `12` |
| `SECRAG_DATA_DIR` | no | `./data` |

Any OpenRouter model slug works, and you can switch models in the sidebar.

### CLI

```bash
python -m secrag ingest NVDA TSLA JPM                 # default: 3 years, 10-K/10-Q/8-K/3/4 + XBRL
python -m secrag ingest AAPL --forms 10-K --years 8   # deeper history for Filing Diff
python -m secrag refresh                              # new filings for every tracked company
python -m secrag watch --hours 6                      # long-running refresher (or use cron)
python -m secrag status
python -m secrag ask "Did Nvidia insiders sell before the last earnings 8-K?"
```

Default per-company pull: last 3 years, up to 3×10-K, 4×10-Q, 12×8-K, 25×Form 3, 100×Form 4,
plus the full XBRL history. A 125-filing company takes about 1–2 minutes. Re-runs are
incremental: an ingest registry skips filings already indexed, and deterministic vector ids
make re-indexing idempotent.

## Example questions

- *Compare Apple and Microsoft revenue, operating margin and free cash flow over the last 3 years.*
- *What new risk factors did Nvidia add in its latest 10-K?*
- *Have Tesla insiders been buying or selling? How much was under 10b5-1 plans?*
- *Summarize Amazon's latest earnings press release and the segment results.*
- *Any red-flag 8-Ks for Boeing in the last two years?*

## Tests

```bash
pip install -r requirements-dev.txt
pytest                                   # offline parser/diff/chunking tests
RUN_LIVE=1 pytest tests/test_live_edgar.py -s   # real EDGAR → SQLite → Qdrant round trip
```

## Deploying (Streamlit Community Cloud)

Set the main file to **`app.py`** and add `OPENROUTER_API_KEY`, `SEC_USER_AGENT` and
(recommended, since cloud disks are ephemeral) `QDRANT_URL` / `QDRANT_API_KEY` as secrets.

## Security notes

- Retrieved filing text is treated as untrusted: prompts forbid following instructions inside sources.
- SQL tools use parameterized queries only; the LLM never writes SQL.
- Uploads are validated by extension + magic bytes and capped at 10 MB; chat is rate limited per session.
- Secrets come from env / Streamlit secrets only.

## Roadmap

See [ROADMAP.md](ROADMAP.md).

## License

Educational and demonstration purposes. SEC data is public domain; respect SEC's
[fair access policy](https://www.sec.gov/os/accessing-edgar-data).
