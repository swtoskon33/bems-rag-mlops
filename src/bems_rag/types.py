"""Core domain types shared across the RAG pipeline.

Kept deliberately small and framework-agnostic so every layer (ingest, retrieval,
generation, eval) speaks the same language.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class SourceKind(str, Enum):
    """Where a chunk of context came from."""
    TELEMETRY = "telemetry"      # energy readings (solar, wind, EVs, CHP, consumption)
    DOCUMENT = "document"        # building manuals, specs, maintenance notes


@dataclass(frozen=True)
class Chunk:
    """A retrievable unit of context."""
    id: str
    text: str
    kind: SourceKind
    building_id: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Query:
    """An operator question, scoped to a building (tenant)."""
    text: str
    building_id: str


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk plus its similarity score for a given query."""
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class Answer:
    """The generated answer plus the context it was grounded in."""
    text: str
    contexts: list[RetrievedChunk]
    # numbers cited in the answer must be traceable to a context chunk;
    # grounded=False means the guard caught an unsupported numeric claim.
    grounded: bool = True
