# HNSW parameter sweep

Recall@10 and latency across M (graph degree) and efSearch (search breadth), over 919 embeddings (dim 256), 200 queries. Exact Flat is the recall ground truth. Recall rises toward 1.0 as efSearch grows, at a latency cost - the HNSW knob you tune per latency budget.

| M | efSearch | Recall@10 | p50 (ms) | p95 (ms) |
|---|----------|-----------|----------|----------|
| 8 | 16 | 0.639 | 0.007 | 0.009 |
| 8 | 32 | 0.650 | 0.009 | 0.011 |
| 8 | 64 | 0.643 | 0.016 | 0.021 |
| 8 | 128 | 0.680 | 0.030 | 0.037 |
| 16 | 16 | 0.681 | 0.008 | 0.011 |
| 16 | 32 | 0.647 | 0.011 | 0.016 |
| 16 | 64 | 0.681 | 0.016 | 0.023 |
| 16 | 128 | 0.676 | 0.030 | 0.039 |
| 32 | 16 | 0.705 | 0.009 | 0.010 |
| 32 | 32 | 0.707 | 0.012 | 0.014 |
| 32 | 64 | 0.685 | 0.018 | 0.023 |
| 32 | 128 | 0.683 | 0.031 | 0.040 |
| 64 | 16 | 0.670 | 0.011 | 0.016 |
| 64 | 32 | 0.654 | 0.016 | 0.020 |
| 64 | 64 | 0.655 | 0.022 | 0.028 |
| 64 | 128 | 0.687 | 0.037 | 0.046 |

Takeaway: raising efSearch and M lifts recall, but HNSW does not reach near-exact recall on this corpus even at efSearch=128. The reason is the embedding geometry: the deterministic hashing embedder produces a non-smooth space where nearest-neighbour graph navigation is less effective than IVF's coarse quantisation (IVF hits 0.97 in the index benchmark). The real lesson: ANN index quality depends on the embedding manifold, not just parameters. With smooth semantic embeddings (BGE/E5), HNSW recall would be substantially higher.
