"""Dagster orchestration of the RAG retraining/promotion loop.

Models the pipeline as a DAG of assets so the eval -> gate -> promote flow is a real
orchestrated graph (not just a script). Run the Dagster UI with:

    dagster dev -f src/bems_rag/orchestration/dagster_pipeline.py

then materialise the assets from the UI, or from the CLI.
"""
from __future__ import annotations

from dagster import Definitions, ScheduleDefinition, asset, define_asset_job

from bems_rag.eval.harness import evaluate
from bems_rag.eval.validation_gate import CandidateMetrics, evaluate_gate
from bems_rag.pipeline import RagPipeline

GOLDEN = "data/sample/golden.json"


@asset
def eval_report() -> dict:
    """Evaluate the current RAG pipeline on the golden set."""
    report = evaluate(RagPipeline(), GOLDEN, k=4)
    return {"hit_at_k": report.hit_at_k, "mrr": report.mrr,
            "groundedness": report.groundedness}


@asset
def gate_decision(eval_report: dict) -> dict:
    """Compare the challenger to a champion baseline via the validation gate."""
    champion = CandidateMetrics(hit_at_k=0.80, mrr=0.75, groundedness=1.0)
    challenger = CandidateMetrics(
        hit_at_k=eval_report["hit_at_k"],
        mrr=eval_report["mrr"],
        groundedness=eval_report["groundedness"],
    )
    decision = evaluate_gate(champion, challenger)
    return {"passed": decision.passed, "reasons": decision.reasons}


@asset
def promotion(gate_decision: dict) -> str:
    """Promote the challenger if the gate passed; otherwise keep the champion."""
    if gate_decision["passed"]:
        return "promoted: challenger -> champion"
    reasons = gate_decision["reasons"]
    return f"blocked: {reasons}"


# A job that materialises the whole loop, and a daily schedule for retraining.
retrain_job = define_asset_job("retrain_and_promote", selection="*")
daily_retrain = ScheduleDefinition(job=retrain_job, cron_schedule="0 2 * * *")

defs = Definitions(
    assets=[eval_report, gate_decision, promotion],
    jobs=[retrain_job],
    schedules=[daily_retrain],
)
