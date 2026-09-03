"""Retrieval benchmark: bi-encoder baseline vs two-stage retrieve->rerank.

Reported per query group, and that split is the point. The reranker's synonym map was
written while looking at the dev paraphrases, so scoring on those measures the mapping
rather than the reranking. The held-out paraphrases use wording the map has never seen;
that column is the honest estimate of what reranking buys.

    python scripts/benchmark_retrieval.py
"""
from __future__ import annotations

import json
from pathlib import Path

from bems_rag.ingest.bdg2 import load_bdg2_facet_chunks
from bems_rag.retrieval.reranker import LexicalReranker, Reranker
from bems_rag.retrieval.retriever import Retriever
from bems_rag.types import Query

OUT = Path("docs/retrieval_benchmark.md")
GROUPS = ("direct", "paraphrased_dev", "paraphrased_heldout")


def _load():
    chunks = load_bdg2_facet_chunks("data/bdg2/metadata.csv", limit=25)
    with open("data/sample/golden.json") as f:
        queries = json.load(f)["queries"]
    return chunks, queries


def hit_and_mrr(reranker, chunks, queries, k):
    r = Retriever(reranker=reranker, fetch_k=10)
    r.index(chunks)
    hits = 0
    rr_sum = 0.0
    for q in queries:
        res = r.retrieve(Query(text=q["text"], building_id=q["building_id"]), k=k)
        got = [rc.chunk.id for rc in res]
        rel = set(q["relevant_ids"])
        if any(g in rel for g in got):
            hits += 1
        for rank, g in enumerate(got, start=1):
            if g in rel:
                rr_sum += 1.0 / rank
                break
    n = len(queries) or 1
    return hits / n, rr_sum / n


def main() -> None:
    chunks, all_queries = _load()
    by_group = {g: [q for q in all_queries if q.get("difficulty") == g] for g in GROUPS}

    intro = (
        "Bi-encoder (FAISS) retrieval against two-stage retrieve->rerank, on the golden "
        "set (25 buildings, 75 queries). Results are split by query group because the "
        "reranker's synonym map was written against the dev paraphrases: scoring it "
        "there measures that hand-written mapping, not reranking."
    )
    lines = [
        "# Retrieval benchmark: baseline vs reranker",
        "",
        intro,
        "",
        "| Query group | n | k | hit@k baseline | hit@k reranked | MRR baseline | MRR reranked |",
        "|-------------|---|---|----------------|----------------|--------------|--------------|",
    ]

    summary = {}
    for group in GROUPS:
        qs = by_group[group]
        for k in (1, 3):
            hb, mb = hit_and_mrr(Reranker(), chunks, qs, k)
            hr, mr = hit_and_mrr(LexicalReranker(), chunks, qs, k)
            lines.append(f"| {group} | {len(qs)} | {k} | {hb:.2f} | {hr:.2f} "
                         f"| {mb:.2f} | {mr:.2f} |")
            if k == 1:
                summary[group] = (hb, hr)

    dev_gain = summary["paraphrased_dev"][1] - summary["paraphrased_dev"][0]
    held_gain = summary["paraphrased_heldout"][1] - summary["paraphrased_heldout"][0]

    note = (
        f"On the dev paraphrases the reranker gains {dev_gain:+.2f} hit@1; on the held-out "
        f"paraphrases it gains {held_gain:+.2f}. The held-out number is negative: the "
        "reranker actively hurts retrieval on wording its synonym map has not seen. That "
        "gap is the whole finding. The dev gain measured a hand-written mapping from the "
        "test queries to the corpus vocabulary, not a reranking capability, and once that "
        "mapping does not apply the lexical rescoring reorders candidates worse than the "
        "retriever had them. A synonym table is a lookup, not a model: it cannot "
        "generalise, and here it does not degrade gracefully either. This is the argument "
        "for a cross-encoder, which scores a (query, passage) pair on its own merits "
        "rather than on whether the words happen to match a list."
    )
    caveat = (
        "Both retrieval stages here are lexical: the offline default is a hashing "
        "embedder (bag of hashed tokens), not a semantic model, which is why a lexical "
        "reranker moves the numbers as much as it does. With real embeddings the baseline "
        "would be higher and the reranker's contribution smaller and more semantic. "
        "Reranking is off by default (`RERANKER_BACKEND=none`); these numbers come from "
        "`RERANKER_BACKEND=lexical`."
    )
    lines += ["", note, "", caveat, ""]
    OUT.write_text("\n".join(lines))
    print(f"wrote {OUT}")
    for line in lines[5:12]:
        print(line)


if __name__ == "__main__":
    main()
