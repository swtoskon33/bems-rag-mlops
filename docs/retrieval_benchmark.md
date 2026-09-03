# Retrieval benchmark: baseline vs reranker

Bi-encoder (FAISS) retrieval against two-stage retrieve->rerank, on the golden set (25 buildings, 75 queries). Results are split by query group because the reranker's synonym map was written against the dev paraphrases: scoring it there measures that hand-written mapping, not reranking.

| Query group | n | k | hit@k baseline | hit@k reranked | MRR baseline | MRR reranked |
|-------------|---|---|----------------|----------------|--------------|--------------|
| direct | 25 | 1 | 1.00 | 1.00 | 1.00 | 1.00 |
| direct | 25 | 3 | 1.00 | 1.00 | 1.00 | 1.00 |
| paraphrased_dev | 25 | 1 | 0.44 | 0.80 | 0.44 | 0.80 |
| paraphrased_dev | 25 | 3 | 0.80 | 1.00 | 0.56 | 0.90 |
| paraphrased_heldout | 25 | 1 | 0.36 | 0.16 | 0.36 | 0.16 |
| paraphrased_heldout | 25 | 3 | 0.84 | 0.84 | 0.53 | 0.43 |

On the dev paraphrases the reranker gains +0.36 hit@1; on the held-out paraphrases it gains -0.20. The held-out number is negative: the reranker actively hurts retrieval on wording its synonym map has not seen. That gap is the whole finding. The dev gain measured a hand-written mapping from the test queries to the corpus vocabulary, not a reranking capability, and once that mapping does not apply the lexical rescoring reorders candidates worse than the retriever had them. A synonym table is a lookup, not a model: it cannot generalise, and here it does not degrade gracefully either. This is the argument for a cross-encoder, which scores a (query, passage) pair on its own merits rather than on whether the words happen to match a list.

Both retrieval stages here are lexical: the offline default is a hashing embedder (bag of hashed tokens), not a semantic model, which is why a lexical reranker moves the numbers as much as it does. With real embeddings the baseline would be higher and the reranker's contribution smaller and more semantic. Reranking is off by default (`RERANKER_BACKEND=none`); these numbers come from `RERANKER_BACKEND=lexical`.
