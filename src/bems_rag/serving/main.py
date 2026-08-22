"""Production ASGI entrypoint: a module-level `app` uvicorn can import directly.

The Dockerfile runs `uvicorn bems_rag.serving.main:app`. Unlike create_app (which needs
a ServingState argument), this builds the state once at import time from the bundled
BDG2 data, so the container serves real answers with no extra wiring.
"""
from __future__ import annotations

from bems_rag.ingest.bdg2 import load_bdg2_chunks
from bems_rag.pipeline import RagPipeline
from bems_rag.serving.app import ServingState, create_app


def _build_app():
    chunks = load_bdg2_chunks("data/bdg2/metadata.csv", limit=200)
    champion = RagPipeline()
    champion.index(chunks)
    return create_app(ServingState(champion=champion))


app = _build_app()
