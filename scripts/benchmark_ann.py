"""ANN index benchmark: Flat vs HNSW vs IVF vs IVF-PQ.

Builds several FAISS index types over the same building embeddings and measures the
recall / latency / memory trade-off that drives vector-search engineering. Flat
(exact) is the ground truth for recall; the approximate indexes trade a little recall
for lower latency and/or memory.

    python scripts/benchmark_ann.py

Writes docs/ann_benchmark.md. Uses the deterministic offline embedder so results are
reproducible without any model download.
"""
from __future__ import annotations

import time
from pathlib import Path

import faiss
import numpy as np

from bems_rag.ingest.bdg2 import load_bdg2_facet_chunks
from bems_rag.retrieval.embeddings import get_embedder

OUT = Path("docs/ann_benchmark.md")
K = 10
N_QUERIES = 200


def _vectors():
    chunks = load_bdg2_facet_chunks("data/bdg2/metadata.csv", limit=200)
    emb = get_embedder()
    vecs = np.ascontiguousarray(emb.embed([c.text for c in chunks]), dtype=np.float32)
    return vecs, emb


def _recall(approx_ids: np.ndarray, exact_ids: np.ndarray) -> float:
    """Mean fraction of exact top-K neighbours found by the approximate index."""
    hits = 0
    for a, e in zip(approx_ids, exact_ids):
        hits += len(set(a) & set(e)) / len(e)
    return hits / len(approx_ids)


def _bench(index, queries, exact_ids, name, build_time):
    # latency: time per single-query search (p50/p95 over N queries)
    lat = []
    all_ids = []
    for q in queries:
        t0 = time.perf_counter()
        _, ids = index.search(q.reshape(1, -1), K)
        lat.append((time.perf_counter() - t0) * 1000)  # ms
        all_ids.append(ids[0])
    lat = np.array(lat)
    recall = _recall(np.array(all_ids), exact_ids) if exact_ids is not None else 1.0
    # memory: serialized index size
    size_kb = len(faiss.serialize_index(index)) / 1024
    return {
        "name": name, "recall": recall,
        "p50": np.percentile(lat, 50), "p95": np.percentile(lat, 95),
        "mem_kb": size_kb, "build_ms": build_time * 1000,
    }


def main() -> None:
    vecs, _ = _vectors()
    n, d = vecs.shape
    rng = np.random.default_rng(0)
    queries = vecs[rng.choice(n, min(N_QUERIES, n), replace=False)]

    # Ground truth: exact Flat search
    flat = faiss.IndexFlatIP(d)
    flat.add(vecs)
    _, exact_ids = flat.search(queries, K)

    results = []

    # Flat (exact)
    t = time.perf_counter(); fi = faiss.IndexFlatIP(d); fi.add(vecs)
    results.append(_bench(fi, queries, exact_ids, "Flat (exact)", time.perf_counter() - t))

    # HNSW
    t = time.perf_counter()
    hnsw = faiss.IndexHNSWFlat(d, 16)         # M=16
    hnsw.hnsw.efConstruction = 40
    hnsw.add(vecs)
    hnsw.hnsw.efSearch = 64
    results.append(_bench(hnsw, queries, exact_ids, "HNSW (M=16)", time.perf_counter() - t))

    # IVF
    t = time.perf_counter()
    nlist = 16
    quant = faiss.IndexFlatIP(d)
    ivf = faiss.IndexIVFFlat(quant, d, nlist, faiss.METRIC_INNER_PRODUCT)
    ivf.train(vecs); ivf.add(vecs); ivf.nprobe = 8
    results.append(_bench(ivf, queries, exact_ids, "IVF (nlist=16, nprobe=8)", time.perf_counter() - t))

    # IVF-PQ (compressed)
    t = time.perf_counter()
    m = 8                                     # subquantizers (256 / 8 = 32 dims each)
    quant2 = faiss.IndexFlatIP(d)
    ivfpq = faiss.IndexIVFPQ(quant2, d, nlist, m, 4)  # 4 bits -> 16 centroids/subq
    ivfpq.train(vecs); ivfpq.add(vecs); ivfpq.nprobe = 8
    results.append(_bench(ivfpq, queries, exact_ids, "IVF-PQ (m=8)", time.perf_counter() - t))

    # Write report
    intro = (
        f"FAISS index types over {n} building-facet embeddings (dim {d}), "
        f"{len(queries)} queries, K={K}. Flat is exact (recall 1.0 by definition); "
        "approximate indexes trade recall for latency and/or memory. Deterministic "
        "offline embedder, so numbers are reproducible."
    )
    lines = [
        "# ANN index benchmark",
        "",
        intro,
        "",
        "| Index | Recall@10 | p50 (ms) | p95 (ms) | Memory (KB) | Build (ms) |",
        "|-------|-----------|----------|----------|-------------|------------|",
    ]
    for r in results:
        lines.append(
            f"| {r['name']} | {r['recall']:.3f} | {r['p50']:.3f} | {r['p95']:.3f} "
            f"| {r['mem_kb']:.0f} | {r['build_ms']:.1f} |"
        )
    note = (
        "Reading the trade-off: HNSW keeps recall near-exact at low latency but costs "
        "memory; IVF cuts search cost by probing fewer cells (recall depends on "
        "nprobe); IVF-PQ compresses vectors hard (much lower memory) at some recall "
        "cost. At this corpus size Flat is already fast, so the point is the "
        "methodology - the same harness scales to millions of vectors where the "
        "trade-offs bite."
    )
    lines += ["", note]
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}")
    for line in lines[5:11]:
        print(line)


if __name__ == "__main__":
    main()
