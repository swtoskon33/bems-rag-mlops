"""Report the retrieval score distributions that set the relevance floor.

The floor decides when the system declines to answer. It is a single number separating
two overlapping distributions, so it is worth seeing the overlap rather than trusting the
constant.

    KMP_DUPLICATE_LIB_OK=TRUE EMBEDDING_BACKEND=minilm python scripts/tune_relevance_floor.py
"""
from __future__ import annotations

import json

from bems_rag.pipeline import RagPipeline
from bems_rag.types import Chunk, Query, SourceKind

GOLDEN = "data/sample/golden.json"


def main() -> None:
    with open(GOLDEN) as f:
        golden = json.load(f)
    chunks = [Chunk(id=c["id"], text=c["text"], kind=SourceKind(c["kind"]),
                    building_id=c["building_id"]) for c in golden["chunks"]]
    pipeline = RagPipeline(k=4)
    pipeline.index(chunks)

    answerable, unanswerable = [], []
    for q in golden["queries"]:
        contexts = pipeline.retriever.retrieve(
            Query(text=q["text"], building_id=q["building_id"]), k=1)
        if not contexts:
            continue
        bucket = unanswerable if q["difficulty"] == "out_of_scope" else answerable
        bucket.append(contexts[0].score)

    answerable.sort()
    unanswerable.sort()
    print(f"answerable   n={len(answerable)} min={min(answerable):.3f} "
          f"median={answerable[len(answerable) // 2]:.3f}")
    print(f"unanswerable n={len(unanswerable)} median={unanswerable[len(unanswerable) // 2]:.3f} "
          f"max={max(unanswerable):.3f}")
    print("floor  answers kept  unanswerable declined")
    for floor in (0.10, 0.15, 0.20, 0.25, 0.30, 0.35):
        kept = sum(1 for s in answerable if s >= floor) / len(answerable)
        declined = sum(1 for s in unanswerable if s < floor) / len(unanswerable)
        print(f"{floor:.2f}   {kept:.2f}          {declined:.2f}")


if __name__ == "__main__":
    main()
