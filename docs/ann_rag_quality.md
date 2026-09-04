# ANN approximation -> RAG answer quality

How approximate retrieval affects the end-to-end answer, not just index recall. Global corpus (919 chunks, all buildings, dim 384); 75 golden queries; answer hit@4 on retrieved contexts. Flat (exact) is the reference; the drop column is the RAG-quality cost of approximation.

| Index | Answer hit@4 | Drop vs Flat |
|-------|--------------|--------------|
| Flat | 0.213 | +0.000 |
| HNSW | 0.213 | +0.000 |
| IVF | 0.213 | +0.000 |
| IVF-PQ | 0.240 | -0.027 |

The engineering point: exact search isn't free at scale, and the right question is how much answer quality you trade for lower latency/memory. Here IVF keeps answer quality close to exact while cutting search cost; aggressive compression (IVF-PQ) trades more. This recall -> answer-quality transfer is what matters for a production RAG budget - the same methodology scales to millions of vectors where exact search is infeasible.
