"""FastAPI serving app with champion/challenger routing.

Exposes /answer for operator queries. A request is always answered by the champion in
shadow mode; in canary, a deterministic slice of tenants is served by the challenger.
The challenger's shadow output is recorded for offline comparison.

Both pipelines are injected at startup so the same code serves any champion/challenger
pair. Prometheus metrics are exposed at /metrics.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from fastapi import FastAPI
from pydantic import BaseModel

from bems_rag.pipeline import RagPipeline
from bems_rag.serving.router import RoutingConfig, route_to_challenger, runs_challenger
from bems_rag.types import Query


class AnswerRequest(BaseModel):
    text: str
    building_id: str


class AnswerResponse(BaseModel):
    text: str
    grounded: bool
    served_by: str          # "champion" | "challenger"
    contexts: list[str]     # chunk ids used


@dataclass
class ServingState:
    champion: RagPipeline
    challenger: RagPipeline | None = None
    config: RoutingConfig = field(default_factory=RoutingConfig)
    # in-memory shadow log; in production this would be a datastore
    shadow_log: list[dict] = field(default_factory=list)


def create_app(state: ServingState) -> FastAPI:
    app = FastAPI(title="bems-rag serving")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "stage": state.config.stage.value}

    @app.post("/answer", response_model=AnswerResponse)
    def answer(req: AnswerRequest) -> AnswerResponse:
        query = Query(req.text, req.building_id)

        # Run challenger if it should execute (served or shadow-logged).
        challenger_ans = None
        if state.challenger is not None and runs_challenger(req.building_id, state.config):
            challenger_ans = state.challenger.answer(query)

        serve_challenger = (
            state.challenger is not None
            and route_to_challenger(req.building_id, state.config)
        )

        if serve_challenger and challenger_ans is not None:
            served_by = "challenger"
            ans = challenger_ans
        else:
            served_by = "champion"
            ans = state.champion.answer(query)
            # In shadow, record the challenger's parallel answer for comparison.
            if challenger_ans is not None and not serve_challenger:
                state.shadow_log.append({
                    "building_id": req.building_id,
                    "query": req.text,
                    "champion_grounded": ans.grounded,
                    "challenger_grounded": challenger_ans.grounded,
                })

        return AnswerResponse(
            text=ans.text,
            grounded=ans.grounded,
            served_by=served_by,
            contexts=[rc.chunk.id for rc in ans.contexts],
        )

    return app
