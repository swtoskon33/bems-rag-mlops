# RAG Evaluation Report

_Last run: 2026-09-03 21:47 UTC_

Offline evaluation of the retrieval + generation pipeline on the golden query set.
Regenerate with `python scripts/run_eval.py`.

| Metric | Value | What it measures |
|---|---|---|
| hit@k | 0.880 | fraction of queries whose relevant chunk is in the top-k |
| MRR | 0.698 | mean reciprocal rank of the first relevant chunk |
| groundedness | 1.000 | fraction of answers with every number traceable to context |
| queries | 75 | size of the golden evaluation set |

Metrics are also logged to MLflow (`mlflow ui` to browse run history and compare
champion vs challenger).
