from __future__ import annotations

import uvicorn

from bems_rag.ingest.bdg2 import load_bdg2_chunks
from bems_rag.pipeline import RagPipeline
from bems_rag.serving.app import ServingState, create_app


def build_app():
    chunks = load_bdg2_chunks("data/bdg2/metadata.csv", limit=200)
    champion = RagPipeline()
    champion.index(chunks)
    return create_app(ServingState(champion=champion)), chunks


app, _chunks = build_app()

if __name__ == "__main__":
    print("Loaded", len(_chunks), "buildings. Example:", _chunks[0].building_id)
    uvicorn.run(app, host="127.0.0.1", port=8000)
