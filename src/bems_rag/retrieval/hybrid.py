"""Hybrid retrieval: sparse (BM25) + dense (FAISS) fused with Reciprocal Rank Fusion.

Dense embeddings capture semantics; BM25 captures exact terms (equipment ids, units
like kWh/m2, building codes) that a lexical query needs. RRF combines the two rankings
without tuning score scales: an item's fused score is sum(1 / (rrf_k + rank)) over the
lists it appears in. Standard, robust, and hyperparameter-light.
"""
from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from bems_rag.types import Chunk, Query, RetrievedChunk

_TOKEN = re.compile(r"[a-z0-9]+")


def _tok(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Retriever:
    """Per-tenant BM25 over a building's chunks (same tenant-first isolation)."""

    def __init__(self) -> None:
        self._by_building: dict[str, tuple[BM25Okapi, list[Chunk]]] = {}

    def index(self, chunks: list[Chunk]) -> None:
        grouped: dict[str, list[Chunk]] = {}
        for c in chunks:
            grouped.setdefault(c.building_id, []).append(c)
        self._by_building = {
            bid: (BM25Okapi([_tok(c.text) for c in cs]), cs)
            for bid, cs in grouped.items()
        }

    def retrieve(self, query: Query, k: int = 4) -> list[RetrievedChunk]:
        entry = self._by_building.get(query.building_id)
        if entry is None:
            return []
        bm25, chunks = entry
        scores = bm25.get_scores(_tok(query.text))
        ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
        return [RetrievedChunk(chunk=c, score=float(s)) for s, c in ranked[:k]]


def rrf_fuse(
    dense: list[RetrievedChunk],
    sparse: list[RetrievedChunk],
    k: int = 4,
    rrf_k: int = 60,
) -> list[RetrievedChunk]:
    """Reciprocal Rank Fusion of two ranked lists -> top-k fused."""
    scores: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    for lst in (dense, sparse):
        for rank, rc in enumerate(lst, start=1):
            cid = rc.chunk.id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
            chunks[cid] = rc.chunk
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [RetrievedChunk(chunk=chunks[cid], score=s) for cid, s in fused[:k]]
