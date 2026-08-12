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
