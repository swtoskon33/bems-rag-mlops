# Retrieval benchmark: baseline vs reranker

Bi-encoder (FAISS) retrieval against two-stage retrieve->rerank, on the golden set (25 buildings, 75 queries). Results are split by query group because the reranker's synonym map was written against the dev paraphrases: scoring it there measures that hand-written mapping, not reranking.

| Query group | n | k | hit@k baseline | hit@k reranked | MRR baseline | MRR reranked |
|-------------|---|---|----------------|----------------|--------------|--------------|
| direct | 25 | 1 | 1.00 | 1.00 | 1.00 | 1.00 |
| direct | 25 | 3 | 1.00 | 1.00 | 1.00 | 1.00 |
| paraphrased_dev | 25 | 1 | 1.00 | 1.00 | 1.00 | 1.00 |
| paraphrased_dev | 25 | 3 | 1.00 | 1.00 | 1.00 | 1.00 |
| paraphrased_heldout | 25 | 1 | 0.64 | 0.64 | 0.64 | 0.64 |
| paraphrased_heldout | 25 | 3 | 1.00 | 1.00 | 0.79 | 0.79 |

With semantic embeddings the reranker contributes nothing: +0.00 hit@1 on the dev paraphrases and +0.00 on the held-out ones, which is to say identical scores either way. That closes the question the earlier hashing-embedder run raised. There, the reranker appeared to add +0.36 on dev paraphrases and -0.20 on held-out ones: a synonym map written against the dev wording, compensating for a retriever that could not match a paraphrase at all. Fix the retriever and the compensation has nothing left to do. The reranker ships off by default (`RERANKER_BACKEND=none`); a cross-encoder, which scores a (query, passage) pair rather than checking words against a list, is the version of this stage that would still be worth running.

Numbers are from MiniLM embeddings (`EMBEDDING_BACKEND=minilm`). The hashing embedder remains available as a zero-dependency CI fallback, and its numbers are in docs/embedding_benchmark.md for comparison.
