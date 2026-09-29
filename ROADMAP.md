# Roadmap

Shipped in v2: EDGAR auto-ingest (10-K, 10-Q, 8-K + EX-99, Forms 3/4, XBRL), section-aware
chunking with payload filters, agentic planner + SQL tools, grounding check, Insider Radar
with pre-8-K selling screen, Filing Diff, multi-company financials, background refresh,
CLI, and OpenRouter for all LLM calls.

Ordered by impact over effort.

## Tier 1: Accuracy you can measure

1. **Eval harness + CI.** 150 golden questions (numeric, multi-hop, comparison, insider,
   "not in filings") with auto-grading on answer correctness, citation precision/recall and
   refusal quality. Runs on every PR with a model × retrieval-config leaderboard. Without
   this, every feature below is guesswork.
2. **Hybrid search + reranking.** Qdrant sparse vectors (BM25/SPLADE via fastembed) next to
   dense ones, fused with RRF, then a cross-encoder rerank (`bge-reranker-v2-m3`). This
   targets exact-term lookups ("ASC 842", "Item 1.05", tickers, product names) that dense
   search alone misses.
3. **Iterative research agent.** Replace single-pass plan→retrieve with a LangGraph
   tool-calling loop (search, SQL, fetch-filing, diff, compute) and a step budget, so the
   agent can do multi-hop work ("Which of my watchlist changed auditors and had insider
   selling within 60 days?").
4. **Read-only text-to-SQL** over the structured tables (allow-listed views, sqlglot
   validation, row caps) for arbitrary quantitative questions and cross-company screens.

## Tier 2: Signals nobody else surfaces

5. **Real-time filing alerts.** Poll EDGAR's latest-filings feed every minute and push to
   Slack/email/webhook when a watchlist company files a red-flag 8-K, a cluster of
   insider *buys* (3+ insiders, 10 days), a new 13D, or a Filing Diff that crosses a
   "new material risk" threshold.
6. **Insider signal backtest.** Join daily prices and measure forward 30/90/180-day
   excess returns after cluster buys, CEO/CFO open-market buys, and heavy discretionary
   selling before 8-Ks. Show each company's historical hit rate on the Insider Radar.
7. **Earnings brief generator.** On every 8-K Item 2.02: parse the EX-99.1 tables, compute
   deltas against XBRL prior-period facts, extract guidance changes, and flag
   non-GAAP-to-GAAP gaps. Output a one-page cited brief.
8. **Language-drift indices.** Track Loughran-McDonald sentiment, uncertainty, litigious
   and hedging ("may", "could", "materially") density per Item per filing. Chart the drift
   and alert on spikes in MD&A or Risk Factors.
9. **Accounting quality screen.** Accruals ratio, DSO/DIO trends, valuation-allowance
   swings, NOL usage, auditor tenure and changes, restatement (4.02) history, combined into
   a transparent red-flag score with drill-down to source facts.

## Tier 3: Coverage

10. **More forms.** SC 13D/13G (activists and 5% holders), DEF 14A (executive pay,
    say-on-pay, related-party), 13F-HR (institutional holdings), S-1/424B (IPOs and
    offerings), 20-F/6-K (foreign issuers), 10-K/A amendment chains (link to originals,
    no double counting).
11. **Dimensional XBRL.** Parse the full XBRL instance for segment and geographic revenue
    (axes/members), not just companyfacts totals.
12. **Ownership graph.** Insiders ↔ boards ↔ funds across companies (board interlocks,
    13D/13G holders, 13F positions), queryable in chat ("who sits on both boards?").
13. **Peer sets.** Auto-build peers from SIC code + revenue band, with one-click
    "compare to peers" on every metric and every Filing Diff.

## Tier 4: Product

14. **Investment memo export.** One click turns a conversation into a cited memo (PDF/Docs).
15. **Workspaces.** Saved watchlists, pinned questions that re-run on new filings, and
    shared threads.
16. **Cost and latency dashboard.** Tokens, OpenRouter spend and p50/p95 latency per
    model and per tool, plus prompt caching of the static system prompt.
17. **Postgres + pgvector option** for multi-user deployments, keeping Qdrant as default.
