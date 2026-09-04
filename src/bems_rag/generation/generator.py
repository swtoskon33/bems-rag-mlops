"""Answer generation with a groundedness guard.

Two backends behind one interface (mirrors the embeddings design):
  - TemplateGenerator: deterministic, offline, no API key. Builds a grounded answer
    straight from the retrieved context. Used in tests/CI.
  - OpenAIGenerator: real LLM generation for production (lazy import).

Both answers pass through a groundedness check: any number that appears in the answer
must also appear in the retrieved context. An unsupported numeric claim flips
grounded=False -- this is the guardrail against a model inventing consumption figures.
"""
from __future__ import annotations

import os
import re
from typing import Protocol

from bems_rag.types import Answer, Query, RetrievedChunk

_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def _numbers(text: str) -> set[str]:
    return set(_NUMBER.findall(text))


def check_groundedness(answer_text: str, contexts: list[RetrievedChunk]) -> bool:
    """True if every number in the answer is traceable to some context chunk."""
    context_numbers: set[str] = set()
    for rc in contexts:
        context_numbers |= _numbers(rc.chunk.text)
    return _numbers(answer_text).issubset(context_numbers)


class Generator(Protocol):
    def generate(self, query: Query, contexts: list[RetrievedChunk]) -> Answer:
        ...



# Chosen from the score distributions on the golden set, not by taste. In-scope
# questions score min 0.21 / median 0.45; out-of-scope ones median 0.23 / max 0.41. The
# ranges overlap, so no threshold separates them cleanly: 0.25 keeps 94% of answerable
# questions while declining 60% of unanswerable ones, and moving it either way trades one
# against the other. Regenerate the distributions with scripts/tune_relevance_floor.py.
_RELEVANCE_FLOOR = 0.25


def _is_relevant(query: Query, contexts: list[RetrievedChunk]) -> bool:
    """Does the best retrieved chunk actually bear on the question?

    Retrieval always returns its top-k, even when the corpus holds nothing relevant:
    asked who the facility manager is, it hands back whatever chunk sits closest, and the
    numeric groundedness check passes because the answer quotes that chunk verbatim.
    Groundedness asks whether the answer follows from the context; it cannot tell whether
    the context answers the question. This does, imperfectly -- see the note on the floor.
    """
    if not contexts:
        return False
    return contexts[0].score >= _RELEVANCE_FLOOR


class TemplateGenerator:
    """Deterministic, offline generator: stitches an answer from the top contexts.

    No model, no API key -- so the whole RAG path (retrieve -> generate -> guard) is
    testable and reproducible in CI.
    """

    def generate(self, query: Query, contexts: list[RetrievedChunk]) -> Answer:
        if not contexts:
            return Answer(
                text="I don't have enough context to answer that for this building.",
                contexts=[],
                grounded=True,
            )
        if not _is_relevant(query, contexts):
            return Answer(
                text="I don't have enough context to answer that for this building.",
                contexts=[],
                grounded=False,
            )
        top = contexts[0].chunk.text
        text = f"Based on the available data: {top}"
        return Answer(
            text=text,
            contexts=contexts,
            grounded=check_groundedness(text, contexts),
        )


class OpenAIGenerator:
    """Production generator using an OpenAI chat model (lazy import)."""

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.model = model

    def generate(self, query: Query, contexts: list[RetrievedChunk]) -> Answer:
        from openai import OpenAI  # lazy import

        context_block = "\n".join(f"- {rc.chunk.text}" for rc in contexts)
        prompt = (
            "Answer the operator's question using ONLY the context below. "
            "Do not invent numbers. If the context is insufficient, say so.\n\n"
            f"Context:\n{context_block}\n\nQuestion: {query.text}"
        )
        client = OpenAI()
        resp = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        text = resp.choices[0].message.content or ""
        return Answer(
            text=text,
            contexts=contexts,
            grounded=check_groundedness(text, contexts),
        )


def get_generator() -> Generator:
    """Factory: pick the backend from GENERATION_BACKEND (default: template)."""
    backend = os.getenv("GENERATION_BACKEND", "template").lower()
    if backend == "openai":
        return OpenAIGenerator()
    return TemplateGenerator()
