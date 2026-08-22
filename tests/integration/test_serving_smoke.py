"""Smoke test for the production ASGI entrypoint (guards the Docker CMD).

The Dockerfile serves bems_rag.serving.main:app. This test imports that module-level
app and hits it, so a broken entrypoint (e.g. a factory needing an argument) fails CI
instead of only surfacing as a 500 in the running container.
"""
from fastapi.testclient import TestClient


def test_serving_main_app_answers():
    from bems_rag.serving.main import app

    client = TestClient(app)
    r = client.post(
        "/answer",
        json={"building_id": "Panther_lodging_Dean", "text": "what is the floor area?"},
    )
    assert r.status_code == 200
    body = r.json()
    assert "grounded" in body
    assert body.get("served_by") == "champion"
