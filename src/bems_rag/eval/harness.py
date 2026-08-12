"""Offline RAG evaluation harness with MLflow tracking.

Runs a golden set of queries through the pipeline, computes aggregate retrieval and
generation metrics, and logs them to MLflow so every evaluation is tracked, comparable,
and reproducible.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from bems_rag.eval.metrics import groundedness_rate, hit_at_k, reciprocal_rank
from bems_rag.pipeline import RagPipeline
from bems_rag.types import Chunk, Query, SourceKind


@dataclass(frozen=True)
class EvalReport:
    hit_at_k: float
    mrr: float
    groundedness: float
    n_queries: int

    def as_dict(self) -> dict[str, float]:
        return {
            "hit_at_k": self.hit_at_k,
            "mrr": self.mrr,
            "groundedness": self.groundedness,
            "n_queries": float(self.n_queries),
        }


def load_golden(path: str | Path) -> tuple[list[Chunk], list[tuple[Query, set[str]]]]:
    data = json.loads(Path(path).read_text())
    chunks = [
        Chunk(c["id"], c["text"], SourceKind(c["kind"]), c["building_id"])
        for c in data["chunks"]
    ]
    queries = [
        (Query(q["text"], q["building_id"]), set(q["relevant_ids"]))
        for q in data["queries"]
    ]
    return chunks, queries


def evaluate(pipeline: RagPipeline, golden_path: str | Path, k: int = 4) -> EvalReport:
    chunks, queries = load_golden(golden_path)
    pipeline.index(chunks)

    hits, rrs, answers = [], [], []
    for query, relevant in queries:
        retrieved = pipeline.retriever.retrieve(query, k=k)
        hits.append(hit_at_k(retrieved, relevant, k))
        rrs.append(reciprocal_rank(retrieved, relevant))
        answers.append(pipeline.generator.generate(query, retrieved))

    n = len(queries)
    return EvalReport(
        hit_at_k=sum(hits) / n,
        mrr=sum(rrs) / n,
        groundedness=groundedness_rate(answers),
        n_queries=n,
    )


def log_to_mlflow(report: EvalReport, run_name: str = "offline-eval") -> None:
    """Log the eval report to MLflow (best-effort; no-op if MLflow unavailable)."""
    import mlflow

    with mlflow.start_run(run_name=run_name):
        mlflow.log_metrics({k: v for k, v in report.as_dict().items()})
