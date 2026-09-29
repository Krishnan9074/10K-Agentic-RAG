"""
Qdrant vector store with SEC-aware payload filters.

- Qdrant Cloud when QDRANT_URL is set, otherwise an embedded on-disk
  instance under data/qdrant (zero setup).
- Deterministic point ids (accession + chunk index) => re-ingesting a
  filing overwrites instead of duplicating.
- Every chunk carries ticker / form_type / fiscal_year / section so the
  retriever can filter ("AAPL 10-K risk factors, 2024-2025") before ranking.
"""
from __future__ import annotations

import atexit
import threading
import uuid
from dataclasses import dataclass
from urllib.parse import urlparse, urlunparse

from qdrant_client import QdrantClient, models

from secrag import config

_NS = uuid.UUID("5d9f8a52-0c1e-4a57-9a33-6f0e8b7c2a10")
_INDEXED_FIELDS = {
    "ticker": models.PayloadSchemaType.KEYWORD,
    "form_type": models.PayloadSchemaType.KEYWORD,
    "section": models.PayloadSchemaType.KEYWORD,
    "accession": models.PayloadSchemaType.KEYWORD,
    "fiscal_year": models.PayloadSchemaType.INTEGER,
}


@dataclass
class Hit:
    text: str
    score: float
    meta: dict


def _make_client() -> QdrantClient:
    url = config.qdrant_url()
    if url:
        # Some hosts (e.g. Streamlit Cloud) block 6333; Qdrant Cloud also serves 443.
        parsed = urlparse(url)
        if not parsed.port:
            url = urlunparse(parsed._replace(netloc=f"{parsed.hostname}:443", scheme="https"))
        return QdrantClient(url=url, api_key=config.qdrant_api_key() or None, timeout=60)
    config.QDRANT_LOCAL_PATH.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(config.QDRANT_LOCAL_PATH))


class VectorStore:
    def __init__(self):
        from fastembed import TextEmbedding

        self.client = _make_client()
        self.collection = config.COLLECTION_NAME
        self._embedder = TextEmbedding(model_name=config.EMBEDDING_MODEL)
        self._embed_lock = threading.Lock()
        # Embedded Qdrant is not safe for concurrent writers/readers (background ingest + chat).
        self._io = threading.RLock()
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                self.collection,
                vectors_config=models.VectorParams(size=config.EMBEDDING_DIM,
                                                   distance=models.Distance.COSINE),
            )
        if not config.qdrant_url():
            return  # embedded Qdrant filters without indexes
        for field, schema in _INDEXED_FIELDS.items():
            try:
                self.client.create_payload_index(self.collection, field, field_schema=schema)
            except Exception:
                pass  # already exists / not supported in local mode

    # ------------------------------------------------------------------ #
    def embed(self, texts: list[str]) -> list[list[float]]:
        with self._embed_lock:
            return [v.tolist() for v in self._embedder.embed(texts, batch_size=64)]

    def embed_query(self, text: str) -> list[float]:
        with self._embed_lock:
            return next(iter(self._embedder.query_embed(text))).tolist()

    @staticmethod
    def point_id(accession: str, idx: int) -> str:
        return str(uuid.uuid5(_NS, f"{accession}:{idx}"))

    def upsert(self, texts: list[str], metas: list[dict], batch: int = 128) -> int:
        for i in range(0, len(texts), batch):
            t, m = texts[i:i + batch], metas[i:i + batch]
            vecs = self.embed(t)
            points = [
                models.PointStruct(
                    id=self.point_id(meta["accession"], meta["chunk_index"]),
                    vector=vec, payload={**meta, "text": text},
                )
                for text, meta, vec in zip(t, m, vecs)
            ]
            with self._io:
                self.client.upsert(self.collection, points=points, wait=True)
        return len(texts)

    def delete_accession(self, accession: str) -> None:
        with self._io:
            self._delete(accession)

    def _delete(self, accession: str) -> None:
        self.client.delete(self.collection, points_selector=models.FilterSelector(
            filter=models.Filter(must=[models.FieldCondition(
                key="accession", match=models.MatchValue(value=accession))])))

    def search(
        self,
        query: str,
        *,
        k: int = config.TOP_K,
        tickers: list[str] | None = None,
        form_types: list[str] | None = None,
        sections: list[str] | None = None,
        year_from: int | None = None,
        year_to: int | None = None,
    ) -> list[Hit]:
        must: list = []
        if tickers:
            must.append(models.FieldCondition(key="ticker", match=models.MatchAny(any=tickers)))
        if form_types:
            must.append(models.FieldCondition(key="form_type", match=models.MatchAny(any=form_types)))
        if sections:
            must.append(models.FieldCondition(key="section", match=models.MatchAny(any=sections)))
        if year_from or year_to:
            must.append(models.FieldCondition(key="fiscal_year", range=models.Range(
                gte=year_from, lte=year_to)))
        vec = self.embed_query(query)
        with self._io:
            res = self.client.query_points(
                self.collection, query=vec, limit=k,
                query_filter=models.Filter(must=must) if must else None, with_payload=True,
            )
        return [Hit(text=p.payload.pop("text", ""), score=p.score, meta=p.payload) for p in res.points]

    def count(self) -> int:
        with self._io:
            return self.client.count(self.collection, exact=False).count


_store: VectorStore | None = None
_store_lock = threading.Lock()


def get_store() -> VectorStore:
    """Process-wide singleton (embedded Qdrant allows one client per path)."""
    global _store
    with _store_lock:
        if _store is None:
            _store = VectorStore()
            # Close explicitly: embedded Qdrant's __del__ crashes at interpreter exit.
            atexit.register(_store.client.close)
        return _store
