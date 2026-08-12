"""Data-quality tests for the BDG2 metadata and the chunks derived from it.

These are the 'data tests' tier: they assert the data is healthy (schema, ranges,
nulls, tenant integrity) before it is allowed into the pipeline. In CI, broken data
fails the build here -- before it can reach retrieval or training.
"""
import csv
from pathlib import Path

import pytest

from bems_rag.ingest.bdg2 import load_bdg2_chunks
from bems_rag.types import SourceKind

METADATA = Path("data/bdg2/metadata.csv")
REQUIRED_COLUMNS = {"building_id", "primaryspaceusage", "sqm", "yearbuilt"}

pytestmark = pytest.mark.data


@pytest.fixture(scope="module")
def rows():
    if not METADATA.exists():
        pytest.skip("BDG2 metadata not downloaded (see README quickstart)")
    with open(METADATA, newline="") as f:
        return list(csv.DictReader(f))


def test_schema_has_required_columns(rows):
    assert REQUIRED_COLUMNS.issubset(set(rows[0].keys()))


def test_building_ids_unique(rows):
    ids = [r["building_id"] for r in rows if r["building_id"]]
    assert len(ids) == len(set(ids))


def test_building_ids_non_null(rows):
    # Every row must have a tenant key, or tenant isolation is meaningless.
    assert all(r["building_id"].strip() for r in rows)


def test_floor_area_positive_when_present(rows):
    for r in rows:
        sqm = r.get("sqm", "").strip()
        if sqm:
            assert float(sqm) > 0, f"non-positive sqm for {r['building_id']}"


def test_year_built_plausible_when_present(rows):
    for r in rows:
        year = r.get("yearbuilt", "").strip()
        if year:
            assert 1800 <= int(float(year)) <= 2026, f"implausible year for {r['building_id']}"


def test_chunks_have_valid_kind_and_tenant():
    if not METADATA.exists():
        pytest.skip("BDG2 metadata not downloaded")
    chunks = load_bdg2_chunks(METADATA, limit=100)
    assert len(chunks) == 100
    for c in chunks:
        assert c.building_id.strip()
        assert c.kind == SourceKind.DOCUMENT
        assert c.text.strip()
