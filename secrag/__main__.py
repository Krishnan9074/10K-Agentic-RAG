"""
Command line interface.

    python -m secrag ingest AAPL MSFT NVDA            # pull, parse, index
    python -m secrag ingest TSLA --forms 10-K,8-K --years 5
    python -m secrag refresh                          # new filings for tracked companies
    python -m secrag watch --hours 6                  # refresh forever (cron alternative)
    python -m secrag status
    python -m secrag ask "How did NVDA's gross margin trend and did insiders sell?"
"""
from __future__ import annotations

import argparse
import logging
import sys
import time


def _progress(frac: float, msg: str) -> None:
    print(f"  [{frac * 100:5.1f}%] {msg}", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="secrag", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ingest", help="ingest companies from EDGAR")
    p.add_argument("tickers", nargs="+")
    p.add_argument("--forms", default="", help="comma list, default 10-K,10-Q,8-K,3,4")
    p.add_argument("--years", type=int, default=None, help="look-back window in years")
    p.add_argument("--no-xbrl", action="store_true")
    p.add_argument("--amendments", action="store_true", help="include /A amendments")

    sub.add_parser("refresh", help="pull new filings for every tracked company")
    w = sub.add_parser("watch", help="refresh on an interval, forever")
    w.add_argument("--hours", type=float, default=6)
    sub.add_parser("status", help="show indexed companies")
    a = sub.add_parser("ask", help="ask a question from the terminal")
    a.add_argument("question")
    a.add_argument("--model", default=None)

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    from secrag import config
    from secrag.pipeline.ingest import ingest_company, refresh_all
    from secrag.store import db

    if args.cmd == "ingest":
        kw = {}
        if args.forms:
            kw["forms"] = [f.strip() for f in args.forms.split(",") if f.strip()]
        if args.years:
            kw["since_years"] = args.years
        for t in args.tickers:
            print(f"==> {t}")
            r = ingest_company(t, include_xbrl=not args.no_xbrl,
                               include_amendments=args.amendments, progress=_progress, **kw)
            for e in r.errors:
                print(f"  ! {e}")
        return 0

    if args.cmd == "refresh":
        for r in refresh_all(progress=_progress):
            print(r.summary())
        return 0

    if args.cmd == "watch":
        while True:
            for r in refresh_all(progress=_progress):
                print(r.summary())
            print(f"sleeping {args.hours}h...")
            time.sleep(args.hours * 3600)

    if args.cmd == "status":
        df = db.companies()
        print(df.to_string(index=False) if len(df) else "No companies indexed yet.")
        from secrag.store.vectors import get_store
        print(f"\nVectors: {get_store().count():,}  |  DB: {config.DB_PATH}")
        return 0

    if args.cmd == "ask":
        from secrag.agent.rag import SecAgent
        res = SecAgent(args.model).ask(args.question, progress=_progress)
        print("\n" + res["answer"])
        for n in res["notes"]:
            print(f"\n* {n}")
        if not res["grounding"].grounded:
            print("\n! Possibly unsupported:", *res["grounding"].unsupported_claims, sep="\n  - ")
        print("\nSources:")
        for s in res["sources"]:
            print(f"  [{s.ref}] {s.title} {s.url}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
