# Retrieval benchmark: baseline vs reranker

Bi-encoder (FAISS) retrieval vs two-stage retrieve->rerank on the golden set (50 queries, 25 buildings). The reranker rescores (query, chunk) pairs by domain-normalised token overlap.

| k | hit@k baseline | hit@k reranked | MRR baseline | MRR reranked |
|---|----------------|----------------|--------------|--------------|
| 1 | 0.72 | 0.90 | 0.72 | 0.90 |
| 2 | 0.72 | 1.00 | 0.72 | 0.95 |
| 3 | 0.90 | 1.00 | 0.78 | 0.95 |

The reranker lifts hit@1 and hit@2 substantially: the bi-encoder retrieves the right building's chunks, and the reranker reorders them so the correct facet surfaces first. A production system would swap the lexical scorer for a cross-encoder behind the same interface.
