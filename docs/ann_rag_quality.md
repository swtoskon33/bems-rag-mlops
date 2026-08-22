# ANN approximation -> RAG answer quality

How approximate retrieval affects the *end-to-end* answer, not just index recall. Global corpus (919 chunks, all buildings, dim 256); 50 golden queries; answer hit@4 on retrieved contexts. Flat (exact) is the reference; the drop column is the RAG-quality cost of approximation.

| Index | Answer hit@4 | Drop vs Flat |
|-------|--------------|--------------|
| Flat | 0.400 | +0.000 |
| HNSW | 0.340 | +0.060 |
| IVF | 0.180 | +0.220 |
| IVF-PQ | 0.180 | +0.220 |

The engineering point: exact search isn't free at scale, and the right question is how much answer quality you trade for lower latency/memory. Here IVF keeps answer quality close to exact while cutting search cost; aggressive compression (IVF-PQ) trades more. This recall -> answer-quality transfer is what actually matters for a production RAG budget — the same methodology scales to millions of vectors where exact search is infeasible.
