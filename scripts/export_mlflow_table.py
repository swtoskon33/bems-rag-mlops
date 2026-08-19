"""Export MLflow runs + registry aliases to a Markdown snippet for the README."""
from __future__ import annotations

from pathlib import Path

import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_registry_uri("sqlite:///mlflow.db")
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

OUT = Path("docs/mlflow_runs.md")


def main() -> None:
    client = MlflowClient()
    exp = client.get_experiment_by_name("bems-rag-eval")

    lines = [
        "## MLflow tracking",
        "",
        ("Real runs logged to the MLflow tracking server and registry "
        "(regenerate with `python scripts/populate_mlflow.py`)."),
        "",
        "### Eval runs",
        "",
        "| Run | k | hit@k | MRR | groundedness |",
        "|-----|---|-------|-----|--------------|",
    ]
    if exp is not None:
        runs = client.search_runs([exp.experiment_id], order_by=["start_time ASC"])
        for r in runs:
            name = r.data.tags.get("mlflow.runName", r.info.run_id[:8])
            k = r.data.params.get("k", "-")
            hit = r.data.metrics.get("hit_at_k", 0.0)
            mrr = r.data.metrics.get("mrr", 0.0)
            g = r.data.metrics.get("groundedness", 0.0)
            lines.append(f"| {name} | {k} | {hit:.2f} | {mrr:.2f} | {g:.2f} |")

    lines += ["", "### Model registry (bems-rag)", "",
              "| Alias | Version |", "|-------|---------|"]
    for alias in ("champion", "challenger"):
        mv = None
        try:
            mv = client.get_model_version_by_alias("bems-rag", alias)
        except MlflowException:
            continue
        lines.append(f"| {alias} | v{mv.version} |")

    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
