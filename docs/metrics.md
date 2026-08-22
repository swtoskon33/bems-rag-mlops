# Evaluation metrics

The eval harness (`scripts/run_eval.py`) scores retrieval and generation on the golden
set and logs the results to MLflow and `docs/eval_report.md`.

## Retrieval

**hit@k** — fraction of queries where at least one relevant chunk appears in the top-k
retrieved. Measures whether the right context is found at all.

    hit@k = (# queries with a relevant chunk in top-k) / (# queries)

**MRR (Mean Reciprocal Rank)** — averages 1/rank of the first relevant chunk. Rewards
ranking the right chunk higher, not just including it somewhere in top-k.

    MRR = mean(1 / rank_of_first_relevant_chunk)

A high hit@k but lower MRR means the right chunk is usually retrieved, but not always
ranked first — exactly the signal you want when tuning k or the embedder.

## Generation

**Groundedness** — fraction of answers where every numeric figure traces back to a
retrieved chunk. Guards against invented numbers, which matter in an energy-reporting
domain. Two independent checks:

- **Numeric guard** — every number in the answer must appear in the retrieved context.
- **Semantic overlap** — the answer's content must overlap the retrieved context above
  a threshold, catching non-numeric fabrication.

## Why the scores aren't 1.00

The golden set mixes direct questions ("what is the floor area?") with paraphrased ones
("how big is this place in square metres?") over multiple facet chunks per building. The
offline hashing embedder is lexical, so paraphrased queries sometimes retrieve the wrong
facet — producing realistic hit@k ≈ 0.90, MRR ≈ 0.72. The eval is designed to expose
these misses, not to score a perfect (and meaningless) 1.00.
