"""Ingest Building Data Genome Project 2 metadata into RAG chunks.

Each building's metadata row becomes a natural-language 'document' chunk that the RAG
system can retrieve and answer over. Real data: 1,636 non-residential buildings from
the ASHRAE Great Energy Predictor III competition (open dataset).
"""
from __future__ import annotations

import csv
from pathlib import Path

from bems_rag.types import Chunk, SourceKind

# Meter columns: value is "Yes" when the building has that meter.
_METERS = ["electricity", "hotwater", "chilledwater", "steam", "water",
           "irrigation", "solar", "gas"]


def _describe(row: dict[str, str]) -> str:
    """Turn one metadata row into a natural-language building description."""
    parts: list[str] = []
    name = row.get("building_id", "unknown")
    usage = row.get("primaryspaceusage") or "unknown use"
    parts.append(f"Building {name} is a {usage.lower()} facility.")

    sqm = row.get("sqm")
    if sqm:
        parts.append(f"It has a floor area of {sqm} square meters.")

    year = row.get("yearbuilt")
    if year:
        parts.append(f"It was built in {year.split('.')[0]}.")

    occ = row.get("occupants")
    if occ:
        parts.append(f"It has approximately {occ} occupants.")

    eui = row.get("eui")
    if eui:
        parts.append(f"Its energy use intensity (EUI) is {eui}.")

    present = [m for m in _METERS if (row.get(m) or "").strip().lower() == "yes"]
    if present:
        parts.append("Metered energy sources: " + ", ".join(present) + ".")

    return " ".join(parts)


def _facets(row: dict[str, str]) -> list[tuple[str, str]]:
    """Break one metadata row into per-facet (facet_name, sentence) pairs.

    Multiple chunks per building means retrieval has to pick the *right* facet, not
    just the building -- which is what makes the eval discriminative instead of a
    guaranteed hit.
    """
    name = row.get("building_id", "unknown")
    out: list[tuple[str, str]] = []

    usage = row.get("primaryspaceusage")
    if usage:
        out.append(("usage", f"Building {name} is a {usage.lower()} facility."))

    sqm = row.get("sqm")
    if sqm:
        out.append(("area", f"Building {name} has a floor area of {sqm} square meters."))

    year = row.get("yearbuilt")
    if year:
        out.append(("year", f"Building {name} was built in {year.split('.')[0]}."))

    eui = row.get("eui")
    if eui:
        out.append(("eui", f"Building {name} has an energy use intensity (EUI) of {eui}."))

    present = [m for m in _METERS if (row.get(m) or "").strip().lower() == "yes"]
    if present:
        out.append(("energy", f"Building {name} metered energy sources: " + ", ".join(present) + "."))

    return out


def load_bdg2_facet_chunks(metadata_csv: str | Path, limit: int | None = None) -> list[Chunk]:
    """Produce multiple facet chunks per building (area, year, eui, energy, usage).

    Same tenant key (building_id) as the single-chunk loader, so retrieval stays
    scoped per building -- but now there are several chunks to choose between.
    """
    chunks: list[Chunk] = []
    with open(metadata_csv, newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if limit is not None and i >= limit:
                break
            bid = row.get("building_id")
            if not bid:
                continue
            for facet, sentence in _facets(row):
                chunks.append(
                    Chunk(
                        id=f"{bid}_{facet}",
                        text=sentence,
                        kind=SourceKind.DOCUMENT,
                        building_id=bid,
                        metadata={"facet": facet, "usage": row.get("primaryspaceusage", "")},
                    )
                )
    return chunks


def load_bdg2_chunks(metadata_csv: str | Path, limit: int | None = None) -> list[Chunk]:
    """Read metadata.csv and produce one document Chunk per building.

    building_id is used as the tenant key, so retrieval stays scoped per building.
    """
    chunks: list[Chunk] = []
    with open(metadata_csv, newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if limit is not None and i >= limit:
                break
            bid = row.get("building_id")
            if not bid:
                continue
            chunks.append(
                Chunk(
                    id=f"{bid}_meta",
                    text=_describe(row),
                    kind=SourceKind.DOCUMENT,
                    building_id=bid,
                    metadata={
                        "site_id": row.get("site_id", ""),
                        "usage": row.get("primaryspaceusage", ""),
                    },
                )
            )
    return chunks
