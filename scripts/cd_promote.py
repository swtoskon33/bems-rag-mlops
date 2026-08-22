"""CD promotion: evaluate the challenger, gate it against the live champion, promote.

Unlike a stub that always auto-promotes, this reads the current champion's metrics from
the registry and only promotes the challenger if the validation gate passes. On the
first run (no champion) the challenger is registered and promoted to seed the registry.
"""
from __future__ import annotations

import sys

import mlflow

from bems_rag.eval.harness import evaluate
from bems_rag.eval.registry import Registry
from bems_rag.eval.validation_gate import CandidateMetrics, evaluate_gate
from bems_rag.pipeline import RagPipeline

GOLDEN = "data/sample/golden.json"


def _load_champion_metrics(registry: Registry) -> CandidateMetrics | None:
    """Read the live champion's metrics from the registry (None if no champion yet)."""
    version = registry.get_alias_version("champion")
    if version is None:
        return None
    m = registry.get_version_metrics(version)
    if not {"hit_at_k", "mrr", "groundedness"} <= m.keys():
        return None
    return CandidateMetrics(
        hit_at_k=m["hit_at_k"], mrr=m["mrr"], groundedness=m["groundedness"]
    )


def _register_challenger(registry: Registry, metrics: dict[str, float]) -> str:
    """Log a run and register the challenger config as a new version."""
    with mlflow.start_run() as run:
        mlflow.log_metrics(metrics)
        version = registry.register(run.info.run_id, metrics)
    return version


def main() -> int:
    registry = Registry()

    report = evaluate(RagPipeline(), GOLDEN, k=4)
    metrics = {
        "hit_at_k": report.hit_at_k,
        "mrr": report.mrr,
        "groundedness": report.groundedness,
    }
    challenger = CandidateMetrics(**metrics)
    print(f"challenger: hit@k={challenger.hit_at_k:.3f} "
          f"mrr={challenger.mrr:.3f} groundedness={challenger.groundedness:.3f}")

    champion = _load_champion_metrics(registry)
    if champion is None:
        version = _register_challenger(registry, metrics)
        registry.promote(version)
        print(f"no champion yet -> registered v{version} and promoted as first champion")
        return 0

    decision = evaluate_gate(champion, challenger)
    if not decision.passed:
        print("gate FAILED -> keeping current champion. reasons:")
        for r in decision.reasons:
            print(f"  - {r}")
        return 1

    version = _register_challenger(registry, metrics)
    demoted = registry.promote(version)
    print(f"gate PASSED -> promoted v{version} to champion "
          f"(demoted v{demoted} to previous_champion)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
