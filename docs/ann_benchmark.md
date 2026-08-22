# ANN index benchmark

FAISS index types over 919 building-facet embeddings (dim 256), 200 queries, K=10. Flat is exact (recall 1.0 by definition); approximate indexes trade recall for latency and/or memory. Deterministic offline embedder, so numbers are reproducible.

| Index | Recall@10 | p50 (ms) | p95 (ms) | Memory (KB) | Build (ms) |
|-------|-----------|----------|----------|-------------|------------|
| Flat (exact) | 1.000 | 0.021 | 0.023 | 919 | 0.1 |
| HNSW (M=16) | 0.679 | 0.016 | 0.022 | 1048 | 5.6 |
| IVF (nlist=16, nprobe=8) | 0.973 | 0.016 | 0.020 | 942 | 1.2 |
| IVF-PQ (m=8) | 0.393 | 0.012 | 0.016 | 43 | 15.3 |

Reading the trade-off: HNSW keeps recall near-exact at low latency but costs memory; IVF cuts search cost by probing fewer cells (recall depends on nprobe); IVF-PQ compresses vectors hard (much lower memory) at some recall cost. At this corpus size Flat is already fast, so the point is the methodology — the same harness scales to millions of vectors where the trade-offs bite.
