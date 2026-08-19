"""Integration test: the Dagster asset graph runs end to end.

Skipped when dagster isn't installed (it's an optional 'orchestration' extra), so the
core CI stays lightweight.
"""
import pytest

pytest.importorskip("dagster")

from bems_rag.orchestration.dagster_pipeline import (
    eval_report,
    gate_decision,
    promotion,
)

pytestmark = pytest.mark.integration


def test_dagster_assets_run_end_to_end():
    report = eval_report()
    assert set(report) == {"hit_at_k", "mrr", "groundedness"}

    decision = gate_decision(report)
    assert "passed" in decision

    result = promotion(decision)
    assert result.startswith(("promoted", "blocked"))
