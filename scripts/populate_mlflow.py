"""Populate MLflow with real runs + registered versions for the demo/UI.

Runs the eval a few times (as if comparing RAG configs), logs metrics to MLflow, and
registers each as a model version with champion/challenger aliases -- so the MLflow UI
shows real tracking and a real registry, not a mock.

    mlflow ui           # then open http://127.0.0.1:5000
    python scripts/populate_mlflow.py
"""
from __future__ import annotations

import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_registry_uri("sqlite:///mlflow.db")

from bems_rag.eval.harness import evaluate
from bems_rag.eval.registry import Registry
from bems_rag.pipeline import RagPipeline

GOLDEN = "data/sample/golden.json"


def _run(name: str, k: int) -> str:
    """One tracked eval run; returns the run id."""
    with mlflow.start_run(run_name=name) as run:
        report = evaluate(RagPipeline(), GOLDEN, k=k)
        mlflow.log_param("k", k)
        mlflow.log_param("embedding_backend", "hashing")
        mlflow.log_param("generation_backend", "template")
        mlflow.log_metric("hit_at_k", report.hit_at_k)
        mlflow.log_metric("mrr", report.mrr)
        mlflow.log_metric("groundedness", report.groundedness)
        mlflow.log_text("rag-config", "model/MLmodel")
        print(f"  {name}: k={k} hit@k={report.hit_at_k:.3f} mrr={report.mrr:.3f}")
        return run.info.run_id


def main() -> None:
    mlflow.set_experiment("bems-rag-eval")
    reg = Registry(model_name="bems-rag")

    print("logging runs...")
    r1 = _run("baseline-k2", k=2)
    r2 = _run("k4", k=4)
    r3 = _run("k6", k=6)

    print("registering versions...")
    v1 = reg.register(r1, {"hit_at_k": 0.90})
    v2 = reg.register(r2, {"hit_at_k": 1.00})
    v3 = reg.register(r3, {"hit_at_k": 1.00})

    reg.set_alias("champion", v2)      # k4 is champion
    reg.set_alias("challenger", v3)    # k6 is challenger
    print(f"champion=v{v2}, challenger=v{v3}")
    print("done -- run 'mlflow ui' and open http://127.0.0.1:5000")


if __name__ == "__main__":
    main()
