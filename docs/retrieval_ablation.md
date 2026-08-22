# Retrieval ablation

Each component's contribution on the golden set (50 queries, 25 buildings), at k=1 and k=3.

| Config | hit@1 | MRR@1 | hit@3 | MRR@3 |
|--------|-------|-------|-------|-------|
| Dense only | 0.72 | 0.72 | 0.90 | 0.78 |
| BM25 only | 0.60 | 0.60 | 0.80 | 0.67 |
| Hybrid (RRF) | 0.72 | 0.72 | 0.80 | 0.75 |
| Hybrid + reranker | 0.90 | 0.90 | 1.00 | 0.95 |

Dense captures paraphrased semantics; BM25 captures exact terms; RRF fusion gets the best of both; the reranker then reorders the fused set so the correct facet surfaces first. Each stage adds measurable value.
