"""Run offline RAG evaluation on the BDG2 golden set, log to MLflow, and write a
committed markdown report so results are visible directly in the repo.

Usage:
    python scripts/run_eval.py
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from bems_rag.eval.harness import evaluate, log_to_mlflow
from bems_rag.pipeline import RagPipeline

GOLDEN = "data/sample/golden.json"
REPORT_MD = Path("docs/eval_report.md")
REPORT_JSON = Path("docs/eval_report.json")


def main() -> None:
    report = evaluate(RagPipeline(), GOLDEN, k=4)
    metrics = report.as_dict()

    # 1. Track in MLflow (history / comparison across runs).
    log_to_mlflow(report, run_name="offline-eval")

    # 2. Write a machine-readable artifact.
    REPORT_JSON.write_text(json.dumps(metrics, indent=2) + "\n")

    # 3. Write a human-readable, committed markdown report.
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    md = f"""# RAG Evaluation Report

_Last run: {ts}_

Offline evaluation of the retrieval + generation pipeline on the golden query set.
Regenerate with `python scripts/run_eval.py`.

| Metric | Value | What it measures |
|---|---|---|
| hit@k | {metrics['hit_at_k']:.3f} | fraction of queries whose relevant chunk is in the top-k |
| MRR | {metrics['mrr']:.3f} | mean reciprocal rank of the first relevant chunk |
| groundedness | {metrics['groundedness']:.3f} | fraction of answers with every number traceable to context |
| queries | {int(metrics['n_queries'])} | size of the golden evaluation set |

Metrics are also logged to MLflow (`mlflow ui` to browse run history and compare
champion vs challenger).
"""
    REPORT_MD.write_text(md)
    print("wrote", REPORT_MD, "and", REPORT_JSON)
    print(metrics)


if __name__ == "__main__":
    main()
