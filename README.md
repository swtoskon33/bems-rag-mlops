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

## Why this exists

RAG demos are easy; knowing whether a RAG system is good is the hard part. This repo
shows the parts that matter:

- Grounded generation: every number in an answer must trace back to retrieved context,
  or the answer is flagged (grounded=False). Guards against invented figures.
- Offline evaluation: retrieval (hit@k, MRR) and generation (groundedness) on a golden
  set, logged to MLflow, written to a committed report.
- Multi-tenant retrieval: each building searches only its own chunks.
- Versioned promotion: each RAG config is a registered MLflow model version with
  champion/challenger aliases; promotion is an alias flip, with rollback.
- Automated CD: a workflow runs eval -> validation gate -> promote after CI passes.
- Drift: query-length PSI plus embedding-centroid distance (a semantic signal).
- Groundedness: a numeric guard plus a semantic content-overlap check.
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

## Results

![Evaluation dashboard](docs/eval_dashboard.png)

See docs/eval_report.md (regenerate: `python scripts/run_eval.py`).
Current golden-set scores (25 real-building queries): hit@k 1.00, MRR 1.00, groundedness 1.00.

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
    tests/             unit, data, integration (60 tests)
    scripts/run_eval.py  offline eval -> MLflow + committed report
    docs/              eval report + CI/CD and drift runbooks

## Roadmap

- [x] Core RAG: ingest, retrieval, generation, groundedness guard
- [x] Offline eval harness + MLflow tracking + committed report
- [x] Test tiers: unit / data / integration (60 tests)
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
