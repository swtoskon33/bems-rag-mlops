"""Integration tests for the FastAPI serving app and champion/challenger routing."""
import pytest
from fastapi.testclient import TestClient

from bems_rag.pipeline import RagPipeline
from bems_rag.serving.app import ServingState, create_app
from bems_rag.serving.router import RoutingConfig, Stage
from bems_rag.types import Chunk, SourceKind

pytestmark = pytest.mark.integration


def _pipeline(tag: str) -> RagPipeline:
    # Two pipelines with distinguishable answers via different chunk text.
    p = RagPipeline()
    p.index([
        Chunk(f"{tag}_c", f"{tag}: solar produced 320 kWh", SourceKind.TELEMETRY, "b1"),
    ])
    return p


def test_shadow_serves_champion_but_runs_challenger():
    state = ServingState(
        champion=_pipeline("champion"),
        challenger=_pipeline("challenger"),
        config=RoutingConfig(stage=Stage.SHADOW),
    )
    client = TestClient(create_app(state))
    r = client.post("/answer", json={"text": "solar?", "building_id": "b1"})
    assert r.status_code == 200
    body = r.json()
    assert body["served_by"] == "champion"        # user sees champion
    assert len(state.shadow_log) == 1             # challenger ran and was logged


def test_full_serves_challenger():
    state = ServingState(
        champion=_pipeline("champion"),
        challenger=_pipeline("challenger"),
        config=RoutingConfig(stage=Stage.FULL),
    )
    client = TestClient(create_app(state))
    r = client.post("/answer", json={"text": "solar?", "building_id": "b1"})
    assert r.json()["served_by"] == "challenger"


def test_no_challenger_falls_back_to_champion():
    state = ServingState(champion=_pipeline("champion"))
    client = TestClient(create_app(state))
    r = client.post("/answer", json={"text": "solar?", "building_id": "b1"})
    assert r.json()["served_by"] == "champion"
    assert state.shadow_log == []


def test_health_reports_stage():
    state = ServingState(champion=_pipeline("champion"),
                         config=RoutingConfig(stage=Stage.CANARY, rollout_pct=5))
    client = TestClient(create_app(state))
    assert client.get("/health").json()["stage"] == "canary"
