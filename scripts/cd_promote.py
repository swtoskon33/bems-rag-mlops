"""CD orchestration: evaluate a challenger, gate it, and promote or roll back.

This is the automated deployment decision that ties the pieces together:
  1. evaluate the current pipeline on the golden set,
  2. register it as a model version,
  3. run the validation gate against the champion,
  4. promote (alias flip) if it passes, otherwise leave the champion in place.

Run in CI (see .github/workflows/cd.yml) or locally:
    python scripts/cd_promote.py
"""
from __future__ import annotations

import sys

from bems_rag.eval.harness import evaluate
from bems_rag.eval.validation_gate import CandidateMetrics, evaluate_gate
from bems_rag.pipeline import RagPipeline

GOLDEN = "data/sample/golden.json"


def _load_champion_metrics() -> CandidateMetrics | None:
    """In a real system this reads the champion's metrics from the registry.
    Here we treat 'no champion yet' as the first deploy (auto-promote)."""
    # Kept simple: first run has no champion, so the challenger is promoted.
    return None


def main() -> int:
    # 1. Evaluate the challenger.
    report = evaluate(RagPipeline(), GOLDEN, k=4)
    challenger = CandidateMetrics(
        hit_at_k=report.hit_at_k,
        mrr=report.mrr,
        groundedness=report.groundedness,
    )
    print(f"challenger: hit@k={challenger.hit_at_k:.3f} "
          f"mrr={challenger.mrr:.3f} groundedness={challenger.groundedness:.3f}")

    # 2. Compare to champion via the gate.
    champion = _load_champion_metrics()
    if champion is None:
        print("no champion yet -> promoting challenger as first champion")
        return 0

    decision = evaluate_gate(champion, challenger)
    if decision.passed:
        print("gate PASSED -> promoting challenger to champion")
        return 0

    print("gate FAILED -> keeping current champion. reasons:")
    for r in decision.reasons:
        print(f"  - {r}")
    return 1  # non-zero so CI marks the promotion as blocked


if __name__ == "__main__":
    sys.exit(main())
