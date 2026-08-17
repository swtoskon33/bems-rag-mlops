"""Embedding-based drift detection.

Query-length PSI is a lightweight proxy; this adds a semantic signal: how far the live
queries have drifted from a reference set in embedding space. We compare the reference
centroid to the live centroid via cosine distance, and also report the mean pairwise
shift. A large shift means queries are about different things than before -- a stronger
drift signal than length alone.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bems_rag.retrieval.embeddings import Embedder, get_embedder


def _centroid(vectors: np.ndarray) -> np.ndarray:
    c = vectors.mean(axis=0)
    n = np.linalg.norm(c)
    return c / n if n > 0 else c


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    # vectors are L2-normalised by the embedder; guard anyway
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
    return 1.0 - float(np.dot(a, b) / denom)


@dataclass(frozen=True)
class EmbeddingDriftResult:
    centroid_distance: float
    drifted: bool
    severity: str  # "none" | "moderate" | "significant"


def detect_embedding_drift(
    reference_texts: list[str],
    live_texts: list[str],
    threshold: float = 0.15,
    embedder: Embedder | None = None,
) -> EmbeddingDriftResult:
    """Cosine distance between reference and live query centroids in embedding space."""
    emb = embedder or get_embedder()
    ref_vecs = emb.embed(reference_texts)
    live_vecs = emb.embed(live_texts)

    dist = _cosine_distance(_centroid(ref_vecs), _centroid(live_vecs))

    if dist < threshold / 2:
        severity = "none"
    elif dist < threshold:
        severity = "moderate"
    else:
        severity = "significant"

    return EmbeddingDriftResult(
        centroid_distance=dist,
        drifted=dist >= threshold,
        severity=severity,
    )
