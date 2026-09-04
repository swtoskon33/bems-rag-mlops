# HNSW parameter sweep

Recall@10 and latency across M (graph degree) and efSearch (search breadth), over 919 embeddings (dim 384), 200 queries. Exact Flat is the recall ground truth. Recall rises toward 1.0 as efSearch grows, at a latency cost - the HNSW knob you tune per latency budget.

| M | efSearch | Recall@10 | p50 (ms) | p95 (ms) |
|---|----------|-----------|----------|----------|
| 8 | 16 | 0.996 | 0.009 | 0.013 |
| 8 | 32 | 0.999 | 0.014 | 0.018 |
| 8 | 64 | 0.998 | 0.024 | 0.031 |
| 8 | 128 | 0.998 | 0.045 | 0.053 |
| 16 | 16 | 0.998 | 0.010 | 0.012 |
| 16 | 32 | 0.998 | 0.015 | 0.018 |
| 16 | 64 | 0.999 | 0.026 | 0.032 |
| 16 | 128 | 0.999 | 0.047 | 0.053 |
| 32 | 16 | 0.998 | 0.011 | 0.014 |
| 32 | 32 | 0.998 | 0.015 | 0.018 |
| 32 | 64 | 0.999 | 0.026 | 0.031 |
| 32 | 128 | 0.998 | 0.048 | 0.057 |
| 64 | 16 | 0.999 | 0.015 | 0.018 |
| 64 | 32 | 0.999 | 0.022 | 0.031 |
| 64 | 64 | 0.998 | 0.033 | 0.042 |
| 64 | 128 | 0.998 | 0.057 | 0.078 |

Takeaway: recall is 0.996 to 0.999 across every M and efSearch tried. That is the answer to the earlier run on hashing embeddings, where the same sweep plateaued around 0.71: the limit was never the index parameters, it was the embedding geometry. Graph navigation needs a space where near neighbours are semantically near, and a bag of hashed tokens is not one. With MiniLM the manifold is smooth and HNSW reaches near-exact recall at the cheapest settings on offer.
