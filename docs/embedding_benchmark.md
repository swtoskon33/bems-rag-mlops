# Embedding benchmark: hashing vs sentence-transformer

Retrieval hit@1 by embedder and reranker, on the same golden set (25 buildings, 75 queries). The held-out column is the one that matters: it is worded after the reranker's synonym map was frozen, so nothing in the system was tuned against it.

| Embedder | Reranker | direct | paraphrased (dev) | paraphrased (held-out) | index (s) |
|----------|----------|--------|-------------------|------------------------|-----------|
| hashing | none | 1.00 | 0.44 | 0.36 | 0.0 |
| hashing | lexical | 1.00 | 0.80 | 0.16 | 0.0 |
| minilm | none | 1.00 | 1.00 | 0.64 | 10.47 |
| minilm | lexical | 1.00 | 1.00 | 0.64 | 4.6 |

Two things fall out of this. A real embedder nearly doubles held-out paraphrase retrieval (0.36 to 0.64) because it places rewordings near each other in vector space, which is what the hashing embedder structurally cannot do: it shares no dimensions between two ways of saying the same thing. And once retrieval is semantic, the lexical reranker adds exactly nothing (0.64 either way) -- its apparent value earlier was a synonym table compensating for a weak retriever, not reranking. The right fix for paraphrases was a better embedder, not a hand-written mapping.

The cost is the dependency: the model is a download, so hashing stays the CI default and the semantic path is an optional extra (`pip install -e ".[semantic]"`, `EMBEDDING_BACKEND=minilm`). Indexing is also slower, which the table shows.
