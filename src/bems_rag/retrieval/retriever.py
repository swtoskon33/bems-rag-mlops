"""FAISS-backed retriever with per-building (multi-tenant) isolation.

Each query is scoped to a building_id; the retriever only returns chunks from that
building, so one tenant can never retrieve another tenant's telemetry or documents.
"""
from __future__ import annotations

import faiss
import numpy as np

from bems_rag.retrieval.embeddings import Embedder, get_embedder
from bems_rag.types import Chunk, Query, RetrievedChunk


def _as_faiss(v: np.ndarray) -> np.ndarray:
    """FAISS requires a C-contiguous float32 array."""
    return np.ascontiguousarray(v, dtype=np.float32)


class Retriever:
    def __init__(self, embedder: Embedder | None = None) -> None:
        self.embedder = embedder or get_embedder()
        self._index: faiss.Index | None = None
        self._chunks: list[Chunk] = []

    def index(self, chunks: list[Chunk]) -> None:
        """Build the vector index from a list of chunks."""
        if not chunks:
            raise ValueError("cannot index an empty chunk list")
        self._chunks = list(chunks)
        vectors = _as_faiss(self.embedder.embed([c.text for c in self._chunks]))
        # Inner product on L2-normalised vectors == cosine similarity.
        index = faiss.IndexFlatIP(self.embedder.dim)
        index.add(vectors)
        self._index = index

    def retrieve(self, query: Query, k: int = 4) -> list[RetrievedChunk]:
        """Return the top-k chunks for a query, restricted to its building."""
        if self._index is None:
            raise RuntimeError("index() must be called before retrieve()")

        qv = _as_faiss(self.embedder.embed([query.text]))
        # Over-fetch, then filter by building, then take k. Simple and correct for
        # modest corpora; a production index would use per-tenant partitions.
        n = min(len(self._chunks), max(k * 5, k))
        scores, idxs = self._index.search(qv, n)

        results: list[RetrievedChunk] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            chunk = self._chunks[idx]
            if chunk.building_id != query.building_id:
                continue  # tenant isolation
            results.append(RetrievedChunk(chunk=chunk, score=float(score)))
            if len(results) >= k:
                break
        return results
