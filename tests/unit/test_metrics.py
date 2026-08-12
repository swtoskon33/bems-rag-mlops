"""Unit tests for retrieval/generation metrics."""
import pytest

from bems_rag.eval.metrics import groundedness_rate, hit_at_k, reciprocal_rank
from bems_rag.types import Answer, Chunk, RetrievedChunk, SourceKind


def _rc(cid: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(Chunk(cid, "text", SourceKind.DOCUMENT, "b1"), score)


@pytest.mark.unit
def test_hit_at_k_found():
    ret = [_rc("c2", 0.9), _rc("c1", 0.8)]
    assert hit_at_k(ret, {"c1"}, k=2) == 1.0


@pytest.mark.unit
def test_hit_at_k_not_in_top_k():
    ret = [_rc("c2", 0.9), _rc("c1", 0.8)]
    assert hit_at_k(ret, {"c1"}, k=1) == 0.0


@pytest.mark.unit
def test_reciprocal_rank_second_position():
    ret = [_rc("c2", 0.9), _rc("c1", 0.8)]
    assert reciprocal_rank(ret, {"c1"}) == 0.5


@pytest.mark.unit
def test_reciprocal_rank_none_found():
    ret = [_rc("c2", 0.9)]
    assert reciprocal_rank(ret, {"c1"}) == 0.0


@pytest.mark.unit
def test_groundedness_rate_mixed():
    answers = [
        Answer("ok", [], grounded=True),
        Answer("bad", [], grounded=False),
    ]
    assert groundedness_rate(answers) == 0.5


@pytest.mark.unit
def test_groundedness_rate_empty():
    assert groundedness_rate([]) == 0.0
