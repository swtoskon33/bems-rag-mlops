"""Generate a larger golden evaluation set from real BDG2 buildings.

For each sampled building we create factual questions whose answer is that building's
own metadata chunk -- so the relevant chunk id is known ground truth. This gives a
golden set grounded in real data instead of a handful of hand-written toy queries.

    python scripts/build_golden.py
"""
from __future__ import annotations

import json
from pathlib import Path

from bems_rag.ingest.bdg2 import load_bdg2_chunks

OUT = Path("data/sample/golden.json")
QUESTION_TEMPLATES = [
    "what is the floor area of this building?",
    "when was this building built?",
    "what energy sources does this building have?",
    "what is the energy use intensity?",
    "what type of facility is this?",
]


def main(n_buildings: int = 25) -> None:
    chunks = load_bdg2_chunks("data/bdg2/metadata.csv", limit=n_buildings)

    golden_chunks = [
        {"id": c.id, "text": c.text, "kind": c.kind.value, "building_id": c.building_id}
        for c in chunks
    ]

    queries = []
    # One factual question per building, cycling through templates for variety.
    for i, c in enumerate(chunks):
        template = QUESTION_TEMPLATES[i % len(QUESTION_TEMPLATES)]
        queries.append({
            "text": template,
            "building_id": c.building_id,
            "relevant_ids": [c.id],
        })

    OUT.write_text(json.dumps({"chunks": golden_chunks, "queries": queries}, indent=2))
    print(f"wrote {OUT}: {len(golden_chunks)} chunks, {len(queries)} queries")


if __name__ == "__main__":
    main()
