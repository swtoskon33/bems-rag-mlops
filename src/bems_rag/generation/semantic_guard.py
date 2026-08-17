"""Semantic groundedness check (beyond the numeric regex guard).

The numeric guard catches invented numbers; this adds a lightweight semantic check:
what fraction of the answer's content words are supported by the retrieved context.
It's a token-overlap proxy for entailment -- cheap, deterministic, and offline. A full
solution would use an NLI model or a second LLM call for true entailment; that adds a
model dependency and cost, so this is a documented trade-off, not a claim of complete
faithfulness checking.
"""
from __future__ import annotations

import re

from bems_rag.types import RetrievedChunk

_WORD = re.compile(r"[a-z]+")
# Very common words carry no grounding signal; ignore them.
_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "of", "in", "on", "at", "to", "and",
    "or", "it", "its", "this", "that", "based", "available", "data", "has", "have",
    "with", "for", "as", "by", "from", "building",
}


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 2}


def semantic_support(answer_text: str, contexts: list[RetrievedChunk]) -> float:
    """Fraction of the answer's content words that appear in the retrieved context.

    1.0 means every meaningful word is supported; low values flag an answer that
    introduces content not present in the context.
    """
    answer_words = _content_words(answer_text)
    if not answer_words:
        return 1.0  # nothing substantive to support

    context_words: set[str] = set()
    for rc in contexts:
        context_words |= _content_words(rc.chunk.text)

    supported = answer_words & context_words
    return len(supported) / len(answer_words)


def is_semantically_grounded(
    answer_text: str,
    contexts: list[RetrievedChunk],
    min_support: float = 0.6,
) -> bool:
    """True if enough of the answer's content is supported by the context."""
    return semantic_support(answer_text, contexts) >= min_support
