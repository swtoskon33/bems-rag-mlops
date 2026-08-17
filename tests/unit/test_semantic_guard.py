"""Unit tests for the semantic groundedness guard."""
import pytest

from bems_rag.generation.semantic_guard import is_semantically_grounded, semantic_support
from bems_rag.types import Chunk, RetrievedChunk, SourceKind

pytestmark = pytest.mark.unit


def _ctx(text: str) -> list[RetrievedChunk]:
    return [RetrievedChunk(Chunk("c1", text, SourceKind.TELEMETRY, "b1"), 0.9)]


def test_fully_supported_answer():
    ctx = _ctx("solar production peaked at noon on clear days")
    assert semantic_support("solar production peaked at noon", ctx) == 1.0
    assert is_semantically_grounded("solar production peaked at noon", ctx)


def test_unsupported_answer_flagged():
    ctx = _ctx("solar production peaked at noon")
    assert semantic_support("the chiller failed causing evacuation", ctx) == 0.0
    assert not is_semantically_grounded("the chiller failed causing evacuation", ctx)


def test_partial_support_below_threshold():
    ctx = _ctx("solar production was high")
    # one supported word (solar) out of several -> below 0.6 default
    assert not is_semantically_grounded("solar panels needed urgent maintenance", ctx)


def test_answer_with_only_stopwords_is_grounded():
    ctx = _ctx("solar production")
    # no substantive claim to support
    assert is_semantically_grounded("it is on the data", ctx)
