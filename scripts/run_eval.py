"""Run offline RAG evaluation on the BDG2 golden set, log to MLflow, and write a
committed markdown report so results are visible directly in the repo.

Usage:
    python scripts/run_eval.py
"""
from __future__ import annotations

import json
from pathlib import Path

from bems_rag.pipeline import RagPipeline
from bems_rag.types import Chunk, Query, SourceKind

GOLDEN = "data/sample/golden.json"
REPORT = Path("docs/eval_report.md")
REPORT_MD = Path("docs/eval_report.md")
REPORT_JSON = Path("docs/eval_report.json")


def main() -> None:
    with open(GOLDEN) as f:
        golden = json.load(f)
    chunks = [Chunk(id=c["id"], text=c["text"], kind=SourceKind(c["kind"]),
                    building_id=c["building_id"]) for c in golden["chunks"]]
    queries = golden["queries"]

    pipeline = RagPipeline(k=4)
    pipeline.index(chunks)

    groups = ["direct", "paraphrased_dev", "paraphrased_heldout", "multi_facet"]
    rows = []
    for group in groups:
        qs = [q for q in queries if q.get("difficulty") == group]
        if not qs:
            continue
        hits = 0
        rr = 0.0
        recall = 0.0
        for q in qs:
            answer = pipeline.answer(Query(text=q["text"], building_id=q["building_id"]))
            got = [rc.chunk.id for rc in answer.contexts]
            rel = set(q["relevant_ids"])
            if any(g in rel for g in got):
                hits += 1
            # multi-facet questions need every relevant chunk, not just one
            recall += len(rel & set(got)) / len(rel) if rel else 0.0
            for rank, g in enumerate(got, start=1):
                if g in rel:
                    rr += 1.0 / rank
                    break
        n = len(qs)
        rows.append({"group": group, "n": n, "hit": hits / n,
                     "mrr": rr / n, "recall": recall / n})

    # out of scope: the corpus has no answer, so the measure is whether the system
    # declines instead of producing a confident wrong one
    oos = [q for q in queries if q.get("difficulty") == "out_of_scope"]
    abstained = 0
    for q in oos:
        answer = pipeline.answer(Query(text=q["text"], building_id=q["building_id"]))
        if not answer.grounded:
            abstained += 1
    oos_rate = abstained / len(oos) if oos else 0.0

    for r in rows:
        print(f"  {r['group']:20} n={r['n']:3}  hit@4={r['hit']:.2f}  "
              f"MRR={r['mrr']:.2f}  recall={r['recall']:.2f}")
    print(f"  {'out_of_scope':20} n={len(oos):3}  abstained={abstained}/{len(oos)} "
          f"= {oos_rate:.2f}")

    intro = (
        f"Golden set: {len(queries)} queries over {len(chunks)} facet chunks from 25 real "
        "BDG2 buildings, in five groups. Direct and paraphrased questions have one "
        "answer; multi-facet questions need two chunks, so recall matters more than "
        "hit@k; out-of-scope questions have no answer in the corpus at all, and the only "
        "correct behaviour is to decline."
    )
    lines = [
        "# Evaluation report", "", intro, "",
        "| Query group | n | hit@4 | MRR | recall |",
        "|-------------|---|-------|-----|--------|",
    ]
    for r in rows:
        lines.append(f"| {r['group']} | {r['n']} | {r['hit']:.2f} | {r['mrr']:.2f} "
                     f"| {r['recall']:.2f} |")
    oos_note = (
        f"**Out of scope:** {abstained}/{len(oos)} declined ({oos_rate:.2f}). These have "
        "no answer in the corpus, so a confident response is a failure and an abstention "
        "is the correct result."
    )
    regen = (
        "Regenerate with `KMP_DUPLICATE_LIB_OK=TRUE EMBEDDING_BACKEND=minilm "
        "python scripts/run_eval.py`."
    )
    lines += ["", oos_note, "", regen, ""]
    REPORT.write_text("\n".join(lines))
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    main()
