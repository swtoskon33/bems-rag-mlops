# Retrieval ablation

Each component's contribution on the golden set (75 queries, 25 buildings), at k=1 and k=3.

| Config | hit@1 | MRR@1 | hit@3 | MRR@3 |
|--------|-------|-------|-------|-------|
| Dense only | 0.88 | 0.88 | 1.00 | 0.93 |
| BM25 only | 0.47 | 0.47 | 0.73 | 0.56 |
| Hybrid (RRF) | 0.69 | 0.69 | 0.91 | 0.80 |
| Hybrid + reranker | 0.80 | 0.80 | 0.96 | 0.87 |

With semantic embeddings the ordering inverts from the earlier hashing-embedder run. Dense retrieval alone is now the strongest configuration; adding BM25 through RRF makes it worse (0.88 to 0.69 hit@1), because fusing a weak lexical ranker into a strong semantic one dilutes it. The reranker recovers part of that loss but does not reach dense alone. Hybrid retrieval earns its place when both arms are comparably good, which was true of two lexical arms and is not true here -- the honest conclusion is to serve dense and keep BM25 for exact-term queries rather than fusing it in by default.
