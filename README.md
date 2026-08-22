# bems-rag-mlops

[![CI](https://github.com/swtoskon33/bems-rag-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/swtoskon33/bems-rag-mlops/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)


Production-style RAG for building energy management: operators ask natural-language
questions about their buildings and get grounded answers over real building metadata.
Built as a reference for how a RAG system is evaluated, tracked, and operated.

Data: the open Building Data Genome Project 2 (1,636 real non-residential buildings,
ASHRAE GEPIII competition): https://github.com/buds-lab/building-data-genome-project-2

> A sanitised, public reference inspired by a private building-energy RAG system.
> The modelling is deliberately simple; the focus is the MLOps scaffolding around it.


```
                         RAG request path
  Operator query ──> Retrieve+Rerank ──> Generate ──> Groundedness ──> Answer
  building_id +       per-tenant    template/       numeric +        grounded=T/F
  question            FAISS         OpenAI          semantic

  Data:  BDG2 ingest (1,636 real buildings)  ──>  one FAISS index per tenant

  +-- MLOps scaffolding ------------------------------------------------+
  |                                                                     |
  |  Eval harness       Validation gate    MLflow registry    Serving   |
  |  hit@k MRR grnd      champion vs        versions+aliases   FastAPI   |
  |  50 queries     challenger; no     promote/rollback   shadow/   |
  |                      per-bldg regress                      canary    |
  |                                                                     |
  |  Drift              Monitoring         CI/CD + Dagster   Docker      |
  |  PSI + embedding    Prometheus         tests, eval-gate  one image:  |
  |  centroid           /metrics           -promote; DAG     all stages  |
  |                                                                     |
  +---------------------------------------------------------------------+
```

## Overview

A RAG system for building-energy questions, with the surrounding MLOps: evaluation,
deployment, monitoring, and retraining. The parts included:

- Grounded generation: every number in an answer must trace back to retrieved context,
  or the answer is flagged (grounded=False). Guards against invented figures.
- Offline evaluation: retrieval (hit@k, MRR) and generation (groundedness) on a golden
  set, logged to MLflow, written to a committed report.
- Multi-tenant retrieval: each building searches only its own chunks.

Two-stage retrieval: a bi-encoder (FAISS) fetches candidates, then a cross-encoder-style reranker rescores them. Lifts hit@1 from 0.72 to 0.90 on the golden set (see docs/retrieval_benchmark.md).
- Versioned promotion: each RAG config is a registered MLflow model version with
  champion/challenger aliases; promotion is an alias flip, with rollback.
- Automated CD: a workflow runs eval -> validation gate -> promote after CI passes.
- Drift: query-length PSI plus embedding-centroid distance (a semantic signal).
- Monitoring: Prometheus metrics with a ready Grafana dashboard (monitoring/grafana_dashboard.json) — latency p99, ungrounded ratio, drift, canary %.
- Groundedness: a numeric guard plus a semantic content-overlap check.
- Frontend: a React + Vite query playground (frontend/) — pick a building, ask a
  question, see the grounded answer, served-by model, and round-trip latency.
- Orchestration: the eval -> gate -> promote loop is a Dagster asset graph with a
  daily schedule (see docs/orchestration.md).
- Reproducible & offline-first: deterministic embeddings and a template generator mean
  the whole pipeline and its tests run with no API key.

## Architecture

    Query (building) -> RagPipeline -> grounded answer
                        Retriever (per-tenant FAISS)
                          -> Generator (LLM + groundedness guard)
                          -> Answer (+ contexts)

    Eval harness (golden set) -> hit@k, MRR, groundedness
                              -> MLflow (run history, comparison)
                              -> docs/eval_report.md (committed)

Pluggable backends (env var), so the same code runs offline in CI and with real models
in production:

| Component  | Offline (default)            | Production                              |
|------------|------------------------------|-----------------------------------------|
| Embeddings | HashingEmbedder (det.)       | OpenAIEmbedder (EMBEDDING_BACKEND=openai)|
| Generation | TemplateGenerator            | OpenAIGenerator (GENERATION_BACKEND=openai)|

## Example: querying the API

More examples (Python client, tenant isolation, switching to real models) in [docs/usage.md](docs/usage.md).


```bash
curl -X POST localhost:8000/answer \
  -H "Content-Type: application/json" \
  -d '{"building_id": "Panther_lodging_Dean", "question": "what is the floor area?"}'
```

```json
{
  "building_id": "Panther_lodging_Dean",
  "answer": "Panther_lodging_Dean has a floor area of 508.8 m2, built in 1989.",
  "grounded": true,
  "served_by": "champion"
}
```

The answer is tenant-scoped (only this building's data) and `grounded` is `false` if any
figure can't be traced to retrieved context.

## Frontend

A React + Vite UI (`frontend/`) sits on top of the serving API — building selector,
question box, grounded/ungrounded badge, which model served the answer, and per-query
latency.

```bash
python scripts/serve.py          # backend on :8000
cd frontend && npm install && npm run dev   # UI on :5173
```

## Design decisions & trade-offs

Choices made deliberately, and what each one trades off:

- **Offline-first model (hashing embeddings + template generator).** Keeps the whole
  pipeline and its tests reproducible and runnable in CI with no API key. Trade-off:
  weaker semantic quality than real embeddings. Both backends are pluggable via
  `EMBEDDING_BACKEND` / `GENERATION_BACKEND`, so swapping in OpenAI is one env var.
- **Deliberately simple modelling.** The focus is the MLOps scaffolding, not the model.
  Trade-off: this repo doesn't showcase modelling depth (that lives in my publications) —
  but it shows how I operate ML in production.
- **Champion/challenger compares RAG configs, not trained networks.** The versioned
  artifact is a configuration (embeddings, k, chunking) plus its eval metrics. Trade-off:
  no gradient training loop; the point is the promotion/rollback machinery.
- **Tenant-first retrieval (one index per building).** Guarantees isolation and avoids a
  bug where global top-k then filter returned zero results for small tenants. Trade-off:
  more indexes to hold; fine at this scale, would revisit with a shared ANN index at very
  large tenant counts.
- **Dual groundedness (numeric regex + semantic overlap).** Cheap, deterministic, offline.
  Trade-off: not a full NLI entailment check — a documented approximation.
- **Dual drift (query-length PSI + embedding centroid).** PSI is a light proxy; the
  embedding signal catches topic shift. Trade-off: with hashing embeddings only the
  relative ordering is meaningful (documented in the drift module).

## Results

See docs/eval_report.md (regenerate: `python scripts/run_eval.py`).
Current golden-set scores (50 queries over 125 facet chunks, 25 buildings): hit@k 0.90, MRR 0.72, groundedness 1.00. Scores are deliberately not perfect — the golden set includes paraphrased questions that stress semantic retrieval, so the offline lexical embedder misses some. The eval is meant to expose weaknesses, not flatter the system.

## Retrieval benchmark

Two-stage retrieval (bi-encoder -> reranker) vs the bi-encoder baseline, on the golden
set (regenerate: `python scripts/benchmark_retrieval.py`, full table in
docs/retrieval_benchmark.md):

| k | hit@k baseline | hit@k reranked | MRR baseline | MRR reranked |
|---|----------------|----------------|--------------|--------------|
| 1 | 0.72 | 0.90 | 0.72 | 0.90 |
| 2 | 0.72 | 1.00 | 0.72 | 0.95 |
| 3 | 0.90 | 1.00 | 0.78 | 0.95 |

The bi-encoder finds the right building's chunks; the reranker reorders them so the
correct *facet* surfaces first. The lexical scorer is pluggable (`RERANKER_BACKEND`) —
a production system swaps in a cross-encoder behind the same interface.

## MLflow tracking

Real runs logged to the MLflow tracking server and registry (regenerate with `python scripts/populate_mlflow.py`).

### Eval runs

| Run | k | hit@k | MRR | groundedness |
|-----|---|-------|-----|--------------|
| baseline-k2 | 2 | 1.00 | 1.00 | 1.00 |
| k4 | 4 | 1.00 | 1.00 | 1.00 |
| k6 | 6 | 1.00 | 1.00 | 1.00 |

### Model registry (bems-rag)

| Alias | Version |
|-------|---------|
| champion | v2 |
| challenger | v3 |

## Quickstart

    python3.11 -m venv .venv && source .venv/bin/activate
    pip install -e ".[dev]"

    mkdir -p data/bdg2
    curl -sL "https://media.githubusercontent.com/media/buds-lab/building-data-genome-project-2/master/data/metadata/metadata.csv" -o data/bdg2/metadata.csv

    python scripts/run_eval.py

## Repo layout

    src/bems_rag/
      types.py         shared domain types (Chunk, Query, Answer, ...)
      ingest/          BDG2 metadata -> document chunks
      retrieval/       pluggable embeddings + per-tenant FAISS retriever
      generation/      generator + groundedness guard
      eval/            metrics + harness + MLflow model registry (promotion/rollback)
    tests/             unit, data, integration (64 tests)
    scripts/run_eval.py  offline eval -> MLflow + committed report
    docs/              eval report + CI/CD and drift runbooks

## Roadmap

- [x] Core RAG: ingest, retrieval, generation, groundedness guard
- [x] Offline eval harness + MLflow tracking + committed report
- [x] Test tiers: unit / data / integration (64 tests)
- [x] Champion/challenger validation gate (no per-building regression)
- [x] MLflow model registry: versioned promotion (alias flip) + rollback
- [x] CI (GitHub Actions: lint + all tiers) and CD (shadow -> canary -> rollback)
- [x] Drift detection: query-length PSI + embedding-centroid distance
- [x] Monitoring (Prometheus metrics)
- [x] Automated CD workflow: eval -> gate -> promote (with rollback primitives)
- [x] Groundedness: numeric guard + semantic overlap check
- [x] Dockerfile + CI/CD and drift runbooks

## Stack

Python 3.11, FAISS, MLflow, FastAPI, pytest, Docker, GitHub Actions, Prometheus.

## License

MIT - see LICENSE.
