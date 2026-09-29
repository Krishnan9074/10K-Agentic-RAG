"""
Rate-limited, retrying HTTP client for SEC EDGAR.

- Stays under SEC's 10 req/s limit across threads (shared lock).
- Retries 429/5xx with backoff.
- Caches immutable /Archives/ documents on disk so re-ingests are free.
"""
from __future__ import annotations

import hashlib
import logging
import threading
import time
from typing import Optional

import requests

from secrag import config

log = logging.getLogger(__name__)


class EdgarClient:
    def __init__(self, user_agent: str | None = None, max_rps: float = config.SEC_MAX_RPS):
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": user_agent or config.SEC_USER_AGENT,
            "Accept-Encoding": "gzip, deflate",
        })
        self._interval = 1.0 / max_rps
        self._lock = threading.Lock()
        self._last = 0.0
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    def _throttle(self) -> None:
        with self._lock:
            wait = self._interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()

    @staticmethod
    def _cacheable(url: str) -> bool:
        # Archive documents never change once filed; index/submission JSON does.
        return "/Archives/edgar/data/" in url and not url.endswith("index.json")

    def _cache_path(self, url: str):
        return config.CACHE_DIR / hashlib.sha256(url.encode()).hexdigest()

    def get(self, url: str, *, max_retries: int = 4) -> Optional[bytes]:
        cache = self._cache_path(url) if self._cacheable(url) else None
        if cache is not None and cache.exists():
            return cache.read_bytes()

        for attempt in range(max_retries):
            self._throttle()
            try:
                resp = self._session.get(url, timeout=45)
            except requests.RequestException as e:
                log.warning("EDGAR request failed (%s): %s", url, e)
                time.sleep(2 * (attempt + 1))
                continue
            if resp.status_code == 200:
                if cache is not None:
                    cache.write_bytes(resp.content)
                return resp.content
            if resp.status_code == 404:
                return None
            if resp.status_code in (403, 429) or resp.status_code >= 500:
                # 403 from SEC usually means "slow down" or a bad User-Agent.
                wait = 5 * (attempt + 1)
                log.warning("EDGAR %s on %s, retrying in %ss", resp.status_code, url, wait)
                time.sleep(wait)
                continue
            log.error("EDGAR HTTP %s on %s", resp.status_code, url)
            return None
        return None

    def get_text(self, url: str) -> Optional[str]:
        data = self.get(url)
        if data is None:
            return None
        for enc in ("utf-8", "latin-1"):
            try:
                return data.decode(enc)
            except UnicodeDecodeError:
                continue
        return data.decode("utf-8", errors="replace")

    def get_json(self, url: str) -> Optional[dict]:
        data = self.get(url)
        if data is None:
            return None
        import json
        try:
            return json.loads(data)
        except ValueError:
            log.error("Invalid JSON from %s", url)
            return None


_default: EdgarClient | None = None


def default_client() -> EdgarClient:
    global _default
    if _default is None:
        _default = EdgarClient()
    return _default
