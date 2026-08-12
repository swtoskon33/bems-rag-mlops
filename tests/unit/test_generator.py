"""Unit tests for the groundedness guard and template generator."""
import pytest

from bems_rag.generation.generator import TemplateGenerator, check_groundedness
from bems_rag.types import Chunk, Query, RetrievedChunk, SourceKind


def _ctx(text: str) -> list[RetrievedChunk]:
    return [RetrievedChunk(Chunk("c1", text, SourceKind.TELEMETRY, "b1"), 0.9)]


@pytest.mark.unit
def test_groundedness_number_in_context():
    assert check_groundedness("output was 320 kWh", _ctx("solar produced 320 kWh")) is True


@pytest.mark.unit
def test_groundedness_invented_number():
    assert check_groundedness("output was 999 kWh", _ctx("solar produced 320 kWh")) is False


@pytest.mark.unit
def test_groundedness_no_numbers_is_grounded():
    assert check_groundedness("solar looked strong", _ctx("solar produced 320 kWh")) is True


@pytest.mark.unit
def test_generator_flags_grounded_answer():
    ans = TemplateGenerator().generate(Query("solar?", "b1"), _ctx("solar produced 320 kWh"))
    assert ans.grounded is True
    assert "320" in ans.text


@pytest.mark.unit
def test_generator_empty_context():
    ans = TemplateGenerator().generate(Query("solar?", "b1"), [])
    assert ans.grounded is True
    assert ans.contexts == []
