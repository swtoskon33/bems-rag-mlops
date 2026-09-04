"""The generator declines when retrieval brings back nothing relevant.

Regression test for a real gap: groundedness checked only that the answer's numbers came
from the retrieved context. Since the template answer quotes that context verbatim, it
always passed, so an out-of-scope question got a confident answer built from whatever
chunk happened to rank first. On the golden set that was 25 out of 25.
"""
import pytest

from bems_rag.generation.generator import _RELEVANCE_FLOOR, TemplateGenerator, _is_relevant
from bems_rag.types import Chunk, Query, RetrievedChunk, SourceKind


def _chunk(text, score):
    return RetrievedChunk(
        chunk=Chunk(id="c1", text=text, kind=SourceKind.DOCUMENT, building_id="B1"),
        score=score,
    )


@pytest.mark.unit
def test_declines_when_nothing_was_retrieved():
    answer = TemplateGenerator().generate(Query(text="anything", building_id="B1"), [])
    assert not answer.contexts


@pytest.mark.unit
def test_declines_when_the_best_match_is_below_the_floor():
    weak = [_chunk("Building B1 has a floor area of 500 square meters.", _RELEVANCE_FLOOR - 0.05)]
    answer = TemplateGenerator().generate(
        Query(text="who is the facility manager?", building_id="B1"), weak)
    assert not answer.grounded
    assert answer.contexts == []
    assert "don't have enough context" in answer.text


@pytest.mark.unit
def test_answers_when_the_match_clears_the_floor():
    strong = [_chunk("Building B1 has a floor area of 500 square meters.", _RELEVANCE_FLOOR + 0.2)]
    answer = TemplateGenerator().generate(
        Query(text="what is the floor area?", building_id="B1"), strong)
    assert answer.contexts
    assert "500" in answer.text


@pytest.mark.unit
def test_relevance_helper_uses_the_top_score():
    assert _is_relevant(Query(text="q", building_id="B1"), [_chunk("x", 0.9)])
    assert not _is_relevant(Query(text="q", building_id="B1"), [_chunk("x", 0.01)])
    assert not _is_relevant(Query(text="q", building_id="B1"), [])
