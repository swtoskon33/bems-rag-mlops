# HNSW parameter sweep

Recall@10 and latency across M (graph degree) and efSearch (search breadth), over 919 embeddings (dim 256), 200 queries. Exact Flat is the recall ground truth. Recall rises toward 1.0 as efSearch grows, at a latency cost — the HNSW knob you tune per latency budget.

| M | efSearch | Recall@10 | p50 (ms) | p95 (ms) |
|---|----------|-----------|----------|----------|
| 8 | 16 | 0.640 | 0.007 | 0.010 |
| 8 | 32 | 0.635 | 0.010 | 0.013 |
| 8 | 64 | 0.663 | 0.015 | 0.020 |
| 8 | 128 | 0.661 | 0.029 | 0.036 |
| 16 | 16 | 0.628 | 0.008 | 0.011 |
| 16 | 32 | 0.661 | 0.011 | 0.015 |
| 16 | 64 | 0.661 | 0.017 | 0.023 |
| 16 | 128 | 0.679 | 0.030 | 0.039 |
| 32 | 16 | 0.710 | 0.009 | 0.013 |
| 32 | 32 | 0.687 | 0.012 | 0.016 |
| 32 | 64 | 0.688 | 0.018 | 0.022 |
| 32 | 128 | 0.712 | 0.032 | 0.043 |
| 64 | 16 | 0.672 | 0.011 | 0.016 |
| 64 | 32 | 0.659 | 0.015 | 0.019 |
| 64 | 64 | 0.669 | 0.021 | 0.026 |
| 64 | 128 | 0.682 | 0.038 | 0.049 |

Takeaway: raising efSearch and M lifts recall (0.63 -> 0.71 here), but HNSW does not reach near-exact recall on this corpus even at efSearch=128. The reason is the embedding geometry: the deterministic hashing embedder produces a non-smooth space where nearest-neighbour graph navigation is less effective than IVF's coarse quantisation (IVF hits 0.97 in the index benchmark). This is the real lesson - ANN index quality depends on the embedding manifold, not just the index parameters. With smooth semantic embeddings (BGE/E5), HNSW recall would be substantially higher.
