"""Retrieval and generation metrics for RAG evaluation.

Pure functions so they are trivial to unit-test and reuse in the validation gate.
"""
from __future__ import annotations

from bems_rag.types import Answer, RetrievedChunk


def hit_at_k(retrieved: list[RetrievedChunk], relevant_ids: set[str], k: int) -> float:
    """1.0 if any relevant chunk is in the top-k, else 0.0."""
    top_ids = {rc.chunk.id for rc in retrieved[:k]}
    return 1.0 if top_ids & relevant_ids else 0.0


def reciprocal_rank(retrieved: list[RetrievedChunk], relevant_ids: set[str]) -> float:
    """1/rank of the first relevant chunk (0 if none found)."""
    for i, rc in enumerate(retrieved, start=1):
        if rc.chunk.id in relevant_ids:
            return 1.0 / i
    return 0.0


def groundedness_rate(answers: list[Answer]) -> float:
    """Fraction of answers that passed the groundedness guard."""
    if not answers:
        return 0.0
    return sum(1 for a in answers if a.grounded) / len(answers)
