"""Generate a larger golden evaluation set from real BDG2 buildings.

For each sampled building we create factual questions whose answer is that building's
own metadata chunk -- so the relevant chunk id is known ground truth. This gives a
golden set grounded in real data instead of a handful of hand-written toy queries.

    python scripts/build_golden.py
"""
from __future__ import annotations

import json
from pathlib import Path

from bems_rag.ingest.bdg2 import load_bdg2_facet_chunks

OUT = Path("data/sample/golden.json")
QUESTION_TEMPLATES = [
    "what is the floor area of this building?",
    "when was this building built?",
    "what energy sources does this building have?",
    "what is the energy use intensity?",
    "what type of facility is this?",
]

# Paraphrased / indirect questions: same intent, different words. These deliberately
# avoid the metadata's exact keywords, so they stress semantic retrieval. Under the
# offline hashing embedder (lexical, not semantic) they surface realistic misses --
# the eval is meant to expose weaknesses, not always score 1.0.
PARAPHRASED_TEMPLATES = [
    "how big is this place in square metres?",
    "how old is the property?",
    "which utilities feed the site?",
    "how energy-intensive is it per unit area?",
    "what is this premises used for?",
]


def main(n_buildings: int = 25) -> None:
    chunks = load_bdg2_facet_chunks("data/bdg2/metadata.csv", limit=n_buildings)

    golden_chunks = [
        {"id": c.id, "text": c.text, "kind": c.kind.value, "building_id": c.building_id}
        for c in chunks
    ]

    # Group chunk ids per building and facet, so each question targets the right facet.
    by_building: dict[str, dict[str, str]] = {}
    for c in chunks:
        by_building.setdefault(c.building_id, {})[c.metadata.get("facet", "")] = c.id

    # (question, facet) pairs: direct wording and paraphrased wording.
    DIRECT = [
        ("what is the floor area of this building?", "area"),
        ("when was this building built?", "year"),
        ("what energy sources does this building have?", "energy"),
        ("what is the energy use intensity?", "eui"),
        ("what type of facility is this?", "usage"),
    ]
    # Dev paraphrases: the reranker's synonym map was tuned against these.
    PARAPHRASED_DEV = [
        ("how big is this place in square metres?", "area"),
        ("how old is the property?", "year"),
        ("which utilities feed the site?", "energy"),
        ("how energy-intensive is it per unit area?", "eui"),
        ("what is this premises used for?", "usage"),
    ]

    # Held-out paraphrases: written after the synonym map was frozen, using wording it
    # has never seen. These are what the reported reranker numbers should be read from --
    # scoring on the dev set alone would measure the hand-written mapping, not reranking.
    PARAPHRASED_HELDOUT = [
        ("what is the footprint of this structure?", "area"),
        ("when did construction finish?", "year"),
        ("what powers this location?", "energy"),
        ("how much power does it draw per square metre?", "eui"),
        ("what goes on inside this place?", "usage"),
    ]

    # Multi-facet: one question whose answer needs two chunks. A retriever that returns
    # only one of them is half right, which hit@1 alone cannot express.
    MULTI_FACET = [
        ("how big is it and when was it built?", ["area", "year"]),
        ("what is its energy intensity and which sources does it meter?", ["eui", "energy"]),
        ("what type of building is it and how large?", ["usage", "area"]),
    ]

    # Out of scope: the corpus holds no answer. The correct behaviour is to retrieve
    # nothing relevant and, downstream, to abstain. Scored separately -- a system that
    # confidently answers these is worse than one that returns nothing.
    OUT_OF_SCOPE = [
        "who is the facility manager?",
        "what is the maintenance budget for next year?",
        "when was the last fire inspection?",
        "how many parking spaces does it have?",
        "what is the lease expiry date?",
    ]

    queries = []
    banks = (
        (DIRECT, "direct"),
        (PARAPHRASED_DEV, "paraphrased_dev"),
        (PARAPHRASED_HELDOUT, "paraphrased_heldout"),
    )
    for i, (bid, facets) in enumerate(by_building.items()):
        for bank, difficulty in banks:
            text, facet = bank[i % len(bank)]
            if facet not in facets:
                continue
            queries.append({
                "text": text,
                "building_id": bid,
                "relevant_ids": [facets[facet]],
                "difficulty": difficulty,
            })

    # multi-facet questions carry several relevant ids
    for i, (bid, facets) in enumerate(by_building.items()):
        text, wanted = MULTI_FACET[i % len(MULTI_FACET)]
        ids = [facets[f] for f in wanted if f in facets]
        if len(ids) == len(wanted):
            queries.append({
                "text": text, "building_id": bid, "relevant_ids": ids,
                "difficulty": "multi_facet",
            })

    # out-of-scope questions have no relevant chunk at all
    for i, (bid, _facets) in enumerate(by_building.items()):
        queries.append({
            "text": OUT_OF_SCOPE[i % len(OUT_OF_SCOPE)],
            "building_id": bid, "relevant_ids": [],
            "difficulty": "out_of_scope",
        })

    OUT.write_text(json.dumps({"chunks": golden_chunks, "queries": queries}, indent=2))
    print(f"wrote {OUT}: {len(golden_chunks)} chunks, {len(queries)} queries")


if __name__ == "__main__":
    main()
