"""Integration tests for the MLflow-backed model registry and promotion flow.

Uses a temporary local MLflow store so the test is hermetic and reproducible.
"""
import mlflow
import pytest

from bems_rag.eval.registry import Registry

pytestmark = pytest.mark.integration


@pytest.fixture()
def registry(tmp_path):
    # Point MLflow at a temp store so nothing leaks between tests.
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    mlflow.set_registry_uri(f"sqlite:///{tmp_path}/mlflow.db")
    return Registry(model_name="bems-rag-test")


def _register(reg, metrics):
    with mlflow.start_run() as run:
        mlflow.log_metrics(metrics)
        # Log a trivial artifact so the model version has a source.
        mlflow.log_text("rag-config", "model/MLmodel")
        return reg.register(run.info.run_id, metrics)


def test_register_creates_versions(registry):
    v1 = _register(registry, {"hit_at_k": 0.80})
    v2 = _register(registry, {"hit_at_k": 0.90})
    assert str(v1) == "1"
    assert str(v2) == "2"


def test_promote_sets_champion_and_keeps_previous(registry):
    v1 = _register(registry, {"hit_at_k": 0.80})
    v2 = _register(registry, {"hit_at_k": 0.90})
    registry.set_alias("champion", v1)

    demoted = registry.promote(v2)
    assert demoted == v1
    assert registry.get_alias_version("champion") == v2
    assert registry.get_alias_version("previous_champion") == v1


def test_rollback_restores_previous_champion(registry):
    v1 = _register(registry, {"hit_at_k": 0.90})
    v2 = _register(registry, {"hit_at_k": 0.70})  # a bad challenger
    registry.set_alias("champion", v1)
    registry.promote(v2)                 # v2 now champion, v1 previous

    restored = registry.rollback()       # SLO breach -> roll back
    assert restored == v1
    assert registry.get_alias_version("champion") == v1


def test_metrics_stored_as_version_tags(registry):
    v1 = _register(registry, {"hit_at_k": 0.83, "mrr": 0.77})
    mv = registry.client.get_model_version(registry.model_name, v1)
    assert mv.tags["hit_at_k"] == "0.83"
    assert mv.tags["mrr"] == "0.77"
