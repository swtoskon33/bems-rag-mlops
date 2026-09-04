# Evaluation report

Golden set: 125 queries over 125 facet chunks from 25 real BDG2 buildings, in five groups. Direct and paraphrased questions have one answer; multi-facet questions need two chunks, so recall matters more than hit@k; out-of-scope questions have no answer in the corpus at all, and the only correct behaviour is to decline.

| Query group | n | hit@4 | MRR | recall |
|-------------|---|-------|-----|--------|
| direct | 25 | 1.00 | 1.00 | 1.00 |
| paraphrased_dev | 25 | 0.80 | 0.80 | 0.80 |
| paraphrased_heldout | 25 | 0.96 | 0.75 | 0.96 |
| multi_facet | 25 | 1.00 | 1.00 | 1.00 |

**Out of scope:** 15/25 declined (0.60). These have no answer in the corpus, so a confident response is a failure and an abstention is the correct result.

Regenerate with `KMP_DUPLICATE_LIB_OK=TRUE EMBEDDING_BACKEND=minilm python scripts/run_eval.py`.
