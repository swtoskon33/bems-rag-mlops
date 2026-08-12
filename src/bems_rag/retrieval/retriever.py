"""FAISS-backed retriever with per-building (multi-tenant) isolation.

Retrieval is tenant-first: we restrict to a building's own chunks BEFORE ranking, so a
tenant always searches its full sub-corpus regardless of how the global index ranks
things. This avoids a tenant's chunks being crowded out of a global top-k by other
tenants (which starves retrieval when embeddings are noisy).

Implementation: one small FAISS index per building, built lazily. Correct and simple
for modest per-tenant corpora; a production system might use a single index with
metadata-filtered search (e.g. IVF with tenant partitions).
"""
from __future__ import annotations

import faiss
import numpy as np

from bems_rag.retrieval.embeddings import Embedder, get_embedder
from bems_rag.types import Chunk, Query, RetrievedChunk


def _as_faiss(v: np.ndarray) -> np.ndarray:
    """FAISS requires a C-contiguous float32 array."""
    return np.ascontiguousarray(v, dtype=np.float32)


class _TenantIndex:
    """A FAISS index over a single building's chunks."""

    def __init__(self, chunks: list[Chunk], embedder: Embedder) -> None:
        self.chunks = chunks
        vectors = _as_faiss(embedder.embed([c.text for c in chunks]))
        self.index = faiss.IndexFlatIP(embedder.dim)
        self.index.add(vectors)

    def search(self, qv: np.ndarray, k: int) -> list[RetrievedChunk]:
        n = min(len(self.chunks), k)
        scores, idxs = self.index.search(qv, n)
        out: list[RetrievedChunk] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            out.append(RetrievedChunk(chunk=self.chunks[idx], score=float(score)))
        return out


class Retriever:
    def __init__(self, embedder: Embedder | None = None) -> None:
        self.embedder = embedder or get_embedder()
        self._by_building: dict[str, _TenantIndex] = {}

    def index(self, chunks: list[Chunk]) -> None:
        """Group chunks by building and build one index per tenant."""
        if not chunks:
            raise ValueError("cannot index an empty chunk list")
        grouped: dict[str, list[Chunk]] = {}
        for c in chunks:
            grouped.setdefault(c.building_id, []).append(c)
        self._by_building = {
            bid: _TenantIndex(cs, self.embedder) for bid, cs in grouped.items()
        }

    def retrieve(self, query: Query, k: int = 4) -> list[RetrievedChunk]:
        """Return the top-k chunks for a query, from that building's index only."""
        tenant = self._by_building.get(query.building_id)
        if tenant is None:
            return []  # unknown building -> no context (never leak other tenants)
        qv = _as_faiss(self.embedder.embed([query.text]))
        return tenant.search(qv, k)
