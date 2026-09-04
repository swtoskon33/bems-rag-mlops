"""Retrieval benchmark: bi-encoder baseline vs two-stage retrieve->rerank.

Reported per query group, and that split is the point. The reranker's synonym map was
written while looking at the dev paraphrases, so scoring on those measures the mapping
rather than the reranking. The held-out paraphrases use wording the map has never seen;
that column is the honest estimate of what reranking buys.

    python scripts/benchmark_retrieval.py
"""
from __future__ import annotations

import json
import os
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
    os.environ.setdefault("EMBEDDING_BACKEND", "minilm")
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
        f"With semantic embeddings the reranker contributes nothing: {dev_gain:+.2f} hit@1 "
        f"on the dev paraphrases and {held_gain:+.2f} on the held-out ones, which is to say "
        "identical scores either way. That closes the question the earlier hashing-embedder "
        "run raised. There, the reranker appeared to add +0.36 on dev paraphrases and "
        "-0.20 on held-out ones: a synonym map written against the dev wording, "
        "compensating for a retriever that could not match a paraphrase at all. Fix the "
        "retriever and the compensation has nothing left to do. The reranker ships off by "
        "default (`RERANKER_BACKEND=none`); a cross-encoder, which scores a (query, "
        "passage) pair rather than checking words against a list, is the version of this "
        "stage that would still be worth running."
    )
    caveat = (
        "Numbers are from MiniLM embeddings (`EMBEDDING_BACKEND=minilm`). The hashing "
        "embedder remains available as a zero-dependency CI fallback, and its numbers are "
        "in docs/embedding_benchmark.md for comparison."
    )
    lines += ["", note, "", caveat, ""]
    OUT.write_text("\n".join(lines))
    print(f"wrote {OUT}")
    for line in lines[5:12]:
        print(line)


if __name__ == "__main__":
    main()
