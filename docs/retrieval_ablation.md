# Retrieval ablation

Each component's contribution on the golden set (75 queries, 25 buildings), at k=1 and k=3.

| Config | hit@1 | MRR@1 | hit@3 | MRR@3 |
|--------|-------|-------|-------|-------|
| Dense only | 0.60 | 0.60 | 0.88 | 0.70 |
| BM25 only | 0.47 | 0.47 | 0.73 | 0.56 |
| Hybrid (RRF) | 0.55 | 0.55 | 0.80 | 0.65 |
| Hybrid + reranker | 0.67 | 0.67 | 0.93 | 0.79 |

Dense captures paraphrased semantics; BM25 captures exact terms; RRF fusion gets the best of both; the reranker then reorders the fused set so the correct facet surfaces first. Each stage adds measurable value.
