"""Integration tests: the full retrieve -> generate -> answer path, including the
multi-tenant isolation guarantee and real-data ingestion.
"""
from pathlib import Path

import pytest

from bems_rag.ingest.bdg2 import load_bdg2_chunks
from bems_rag.pipeline import RagPipeline
from bems_rag.types import Chunk, Query, SourceKind

pytestmark = pytest.mark.integration

METADATA = Path("data/bdg2/metadata.csv")


def _toy_chunks() -> list[Chunk]:
    return [
        Chunk("b1_solar", "solar produced 320 kWh at noon", SourceKind.TELEMETRY, "b1"),
        Chunk("b1_hvac", "HVAC setpoint is 22 degrees", SourceKind.DOCUMENT, "b1"),
        Chunk("b2_wind", "wind output was 45 kWh overnight", SourceKind.TELEMETRY, "b2"),
    ]


def test_end_to_end_grounded_answer():
    p = RagPipeline()
    p.index(_toy_chunks())
    ans = p.answer(Query("how much solar did we produce?", "b1"))
    assert ans.grounded is True
    assert "320" in ans.text
    assert all(rc.chunk.building_id == "b1" for rc in ans.contexts)


def test_tenant_isolation_never_leaks_other_buildings():
    p = RagPipeline()
    p.index(_toy_chunks())
    # Ask b1 a question whose best global match is b2's wind chunk.
    ans = p.answer(Query("what was the wind output?", "b1"))
    # It must never return b2's chunk to tenant b1.
    assert all(rc.chunk.building_id == "b1" for rc in ans.contexts)


def test_unknown_building_returns_no_context():
    p = RagPipeline()
    p.index(_toy_chunks())
    ans = p.answer(Query("anything?", "does_not_exist"))
    assert ans.contexts == []


def test_end_to_end_on_real_bdg2_data():
    if not METADATA.exists():
        pytest.skip("BDG2 metadata not downloaded (see README quickstart)")
    chunks = load_bdg2_chunks(METADATA, limit=200)
    p = RagPipeline()
    p.index(chunks)
    bid = chunks[0].building_id
    ans = p.answer(Query("what is the floor area and when was it built?", bid))
    assert ans.grounded is True
    assert ans.contexts, "expected at least one retrieved chunk for a known building"
    assert all(rc.chunk.building_id == bid for rc in ans.contexts)
