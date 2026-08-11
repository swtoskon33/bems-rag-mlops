"""Embedding backends.

Two backends behind one interface:
  - HashingEmbedder: deterministic, offline, no API key. Used in tests and CI so the
    pipeline is fully reproducible and runs anywhere.
  - OpenAIEmbedder: real embeddings for production (loaded lazily; only needs the
    API key when actually used).

The backend is chosen by the EMBEDDING_BACKEND env var (default: hashing).
"""
from __future__ import annotations

import hashlib
import os
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    """Anything that turns text into a fixed-size vector."""
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:
        ...


class HashingEmbedder:
    """Deterministic hashing embedder — same text always maps to the same vector.

    Not semantically meaningful, but perfect for reproducible tests: the retrieval
    plumbing (indexing, top-k, scoring) can be tested end-to-end with zero external
    dependencies.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            # Hash each token into the vector (a tiny "bag of hashed tokens").
            for token in text.lower().split():
                h = int(hashlib.md5(token.encode()).hexdigest(), 16)
                vectors[i, h % self.dim] += 1.0
            norm = np.linalg.norm(vectors[i])
            if norm > 0:
                vectors[i] /= norm  # L2-normalise -> cosine == dot product
        return vectors


class OpenAIEmbedder:
    """Production embedder using OpenAI. Imported lazily so the dependency and API
    key are only needed when this backend is actually selected."""

    def __init__(self, model: str = "text-embedding-3-small", dim: int = 1536) -> None:
        self.model = model
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        from openai import OpenAI  # lazy import

        client = OpenAI()
        resp = client.embeddings.create(model=self.model, input=texts)
        return np.array([d.embedding for d in resp.data], dtype=np.float32)


def get_embedder() -> Embedder:
    """Factory: pick the backend from EMBEDDING_BACKEND (default: hashing)."""
    backend = os.getenv("EMBEDDING_BACKEND", "hashing").lower()
    if backend == "openai":
        return OpenAIEmbedder()
    return HashingEmbedder()
