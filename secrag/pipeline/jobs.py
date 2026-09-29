"""
Background ingestion jobs + auto-refresh scheduler (one per process).

The Streamlit app starts this once: on an empty database it bootstraps the
watchlist, then every AUTO_REFRESH_HOURS it pulls new filings for every
tracked company. The UI enqueues ad-hoc ingests and polls job status.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field

from secrag import config
from secrag.store import db

log = logging.getLogger(__name__)


@dataclass
class Job:
    id: str
    ticker: str
    kind: str = "ingest"            # ingest | refresh
    status: str = "queued"          # queued | running | done | error
    progress: float = 0.0
    message: str = ""
    summary: str = ""
    started: float = 0.0
    finished: float = 0.0
    log: list[str] = field(default_factory=list)


class JobManager:
    def __init__(self):
        self.jobs: dict[str, Job] = {}
        self._q: queue.Queue[Job] = queue.Queue()
        self._lock = threading.Lock()
        threading.Thread(target=self._worker, daemon=True, name="secrag-ingest").start()

    def submit(self, ticker: str, kind: str = "ingest") -> Job:
        ticker = ticker.upper().strip()
        with self._lock:
            for j in self.jobs.values():  # de-dupe identical pending work
                if j.ticker == ticker and j.status in ("queued", "running"):
                    return j
            job = Job(id=uuid.uuid4().hex[:8], ticker=ticker, kind=kind)
            self.jobs[job.id] = job
        self._q.put(job)
        return job

    def active(self) -> list[Job]:
        return [j for j in self.jobs.values() if j.status in ("queued", "running")]

    def recent(self, n: int = 20) -> list[Job]:
        return sorted(self.jobs.values(), key=lambda j: j.started or 1e18, reverse=True)[:n]

    def _worker(self) -> None:
        from secrag.pipeline.ingest import ingest_company

        while True:
            job = self._q.get()
            job.status, job.started = "running", time.time()

            def progress(frac: float, msg: str, job=job):
                job.progress, job.message = frac, msg
                job.log.append(msg)

            try:
                report = ingest_company(job.ticker, progress=progress)
                job.summary = report.summary()
                job.status = "done"
            except Exception as e:
                log.exception("job %s failed", job.id)
                job.status, job.summary = "error", f"{type(e).__name__}: {e}"
            job.progress, job.finished = 1.0, time.time()

    # ------------------------------------------------------------------ #
    def start_scheduler(self) -> None:
        threading.Thread(target=self._schedule_loop, daemon=True, name="secrag-scheduler").start()

    def _schedule_loop(self) -> None:
        if not db.known_tickers():
            for t in config.WATCHLIST:
                self.submit(t)
        interval = config.AUTO_REFRESH_HOURS * 3600
        while interval > 0:
            time.sleep(interval)
            for t in sorted(db.known_tickers()):
                self.submit(t, kind="refresh")


_manager: JobManager | None = None
_manager_lock = threading.Lock()


def get_manager(start_scheduler: bool = True) -> JobManager:
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = JobManager()
            if start_scheduler:
                _manager.start_scheduler()
        return _manager
