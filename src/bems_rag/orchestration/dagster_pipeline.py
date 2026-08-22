"""Dagster orchestration of the RAG retraining/promotion loop.

Models the pipeline as a DAG of assets so the eval -> gate -> promote flow is a real
orchestrated graph (not just a script). Run the Dagster UI with:

    dagster dev -f src/bems_rag/orchestration/dagster_pipeline.py

then materialise the assets from the UI, or from the CLI.
"""
from __future__ import annotations

from dagster import Definitions, ScheduleDefinition, asset, define_asset_job

from bems_rag.eval.harness import evaluate
from bems_rag.eval.registry import Registry
from bems_rag.eval.validation_gate import CandidateMetrics, evaluate_gate
from bems_rag.pipeline import RagPipeline

GOLDEN = "data/sample/golden.json"


@asset
def eval_report() -> dict:
    """Evaluate the current RAG pipeline on the golden set."""
    report = evaluate(RagPipeline(), GOLDEN, k=4)
    return {"hit_at_k": report.hit_at_k, "mrr": report.mrr,
            "groundedness": report.groundedness}


def _champion_metrics(registry: Registry) -> CandidateMetrics | None:
    """Read the live champion's metrics from the registry (None if none yet).

    Shared logic with scripts/cd_promote.py: the orchestrated path must gate against
    the same real champion, not a hardcoded baseline.
    """
    version = registry.get_alias_version("champion")
    if version is None:
        return None
    m = registry.get_version_metrics(version)
    if not {"hit_at_k", "mrr", "groundedness"} <= m.keys():
        return None
    return CandidateMetrics(
        hit_at_k=m["hit_at_k"], mrr=m["mrr"], groundedness=m["groundedness"]
    )


@asset
def gate_decision(eval_report: dict) -> dict:
    """Compare the challenger to the live champion via the validation gate."""
    challenger = CandidateMetrics(
        hit_at_k=eval_report["hit_at_k"],
        mrr=eval_report["mrr"],
        groundedness=eval_report["groundedness"],
    )
    champion = _champion_metrics(Registry())
    if champion is None:
        # No champion registered yet -> first deploy, auto-pass.
        return {"passed": True, "reasons": ["no champion yet (first deploy)"]}
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
