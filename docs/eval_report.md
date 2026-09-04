# RAG Evaluation Report

_Last run: 2026-09-04 05:14 UTC_

Offline evaluation of the retrieval + generation pipeline on the golden query set.
Regenerate with `python scripts/run_eval.py`.

| Metric | Value | What it measures |
|---|---|---|
| hit@k | 1.000 | fraction of queries whose relevant chunk is in the top-k |
| MRR | 0.931 | mean reciprocal rank of the first relevant chunk |
| groundedness | 1.000 | fraction of answers with every number traceable to context |
| queries | 75 | size of the golden evaluation set |

Metrics are also logged to MLflow (`mlflow ui` to browse run history and compare
champion vs challenger).
