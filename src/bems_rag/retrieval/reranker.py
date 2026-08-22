"""Cross-encoder-style reranking stage.

Standard production RAG pattern: a fast bi-encoder (embeddings + FAISS) retrieves a
larger candidate set (top-N), then a slower, more precise reranker rescores those
candidates and keeps the top-k. The reranker sees the (query, chunk) pair together,
so it can weigh term overlap the bi-encoder misses.

Two backends, selected by RERANKER_BACKEND:
  - "none"    : identity (keep retriever order) -- default, reproducible
  - "lexical" : deterministic token-overlap rescoring (offline, no model download)

A real deployment would swap in a cross-encoder (e.g. a MiniLM cross-encoder) behind
the same interface; the point here is the two-stage retrieve->rerank machinery and
its measurable effect on the eval, not the specific scorer.
"""
from __future__ import annotations

import os
import re

from bems_rag.types import Query, RetrievedChunk

_TOKEN = re.compile(r"[a-z0-9]+")

# Small domain synonym map: paraphrased queries use different surface words than the
# metadata sentences. Normalising a few known pairs recovers lexical signal that a
# pure token match would miss (a lightweight stand-in for semantic matching).
_SYNONYMS = {
    "metres": "meters", "metre": "meters", "sqm": "meters", "big": "area",
    "size": "area", "large": "area", "old": "built", "age": "built",
    "utilities": "energy", "power": "energy", "fuel": "energy",
    "intensive": "eui", "premises": "facility", "property": "building",
    "place": "building", "site": "building",
}


def _tokens(text: str) -> set[str]:
    raw = _TOKEN.findall(text.lower())
    return {_SYNONYMS.get(t, t) for t in raw}


class Reranker:
    """Base: identity rerank (returns the retriever's own order)."""

    def rerank(self, query: Query, candidates: list[RetrievedChunk], k: int) -> list[RetrievedChunk]:
        return candidates[:k]


class LexicalReranker(Reranker):
    """Rescore (query, chunk) pairs by weighted token overlap.

    Score = |query ∩ chunk| / |query|  (recall of query terms in the chunk),
    blended with the retriever's own score so ties break sensibly. Deterministic
    and offline -- a stand-in for a cross-encoder that keeps CI reproducible.
    """

    def __init__(self, alpha: float = 0.7) -> None:
        self.alpha = alpha  # weight on the lexical signal vs the retriever score

    def rerank(self, query: Query, candidates: list[RetrievedChunk], k: int) -> list[RetrievedChunk]:
        q = _tokens(query.text)
        if not q:
            return candidates[:k]

        rescored: list[RetrievedChunk] = []
        for rc in candidates:
            overlap = len(q & _tokens(rc.chunk.text)) / len(q)
            blended = self.alpha * overlap + (1 - self.alpha) * rc.score
            rescored.append(RetrievedChunk(chunk=rc.chunk, score=blended))

        rescored.sort(key=lambda r: r.score, reverse=True)
        return rescored[:k]


def get_reranker() -> Reranker:
    """Select a reranker from RERANKER_BACKEND (default: identity 'none')."""
    backend = os.getenv("RERANKER_BACKEND", "none").lower()
    if backend == "lexical":
        return LexicalReranker()
    return Reranker()
