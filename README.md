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
  Operator query ──> Hybrid+Rerank ──> Generate ──> Groundedness ──> Answer
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

Hybrid retrieval: sparse BM25 + dense FAISS fused with Reciprocal Rank Fusion, then reranked — full ablation in docs/retrieval_ablation.md.

ANN benchmark: Flat vs HNSW vs IVF vs IVF-PQ with Recall@10 / latency / memory (see docs/ann_benchmark.md) — the vector-search trade-off, connected to my ANN research.

Two-stage retrieval: a bi-encoder fetches candidates, then a reranker rescores them. Measured on held-out paraphrases the lexical reranker *hurts* retrieval (hit@1 0.36 -> 0.16); it only helps on the wording its synonym map was written against. Documented in docs/retrieval_benchmark.md as the case for a cross-encoder.
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
Current golden-set scores (75 queries over 125 facet chunks, 25 buildings, MiniLM embeddings): hit@k 1.00, MRR 0.93, groundedness 1.00. The golden set covers five groups: direct questions, paraphrases the reranker was tuned against, held-out paraphrases, multi-facet questions needing two chunks, and out-of-scope questions the system should decline. That last group is what caught the abstention gap: before the relevance floor, the system answered 25 out of 25 unanswerable questions with grounded=True.

## Embedding benchmark

Every other number here comes from the offline hashing embedder. This is what changes
with a real semantic model (`KMP_DUPLICATE_LIB_OK=TRUE python scripts/benchmark_embeddings.py`,
full table in docs/embedding_benchmark.md):

| Embedder | Reranker | direct | paraphrased (dev) | paraphrased (held-out) |
|----------|----------|--------|-------------------|------------------------|
| hashing (CI fallback) | none | 1.00 | 0.44 | 0.36 |
| hashing | lexical | 1.00 | 0.80 | 0.16 |
| MiniLM (default for published numbers) | none | 1.00 | 1.00 | 0.64 |
| MiniLM | lexical | 1.00 | 1.00 | 0.64 |

A real embedder nearly doubles held-out paraphrase retrieval (0.36 to 0.64): it places
rewordings near each other in vector space, which a bag of hashed tokens structurally
cannot do. And once retrieval is semantic the lexical reranker adds nothing at all — its
earlier apparent value was a synonym table compensating for a weak retriever. The right
fix for paraphrases was a better embedder, not a hand-written mapping.

Hashing stays the CI default because the model is a download; the semantic path is an
optional extra (`pip install -e ".[semantic]"`, `EMBEDDING_BACKEND=minilm`).


**All benchmark numbers in this README come from MiniLM embeddings.** The hashing embedder remains the zero-dependency fallback so the test suite runs in CI without a model download, but it is not the configuration the results describe. Regenerate with `KMP_DUPLICATE_LIB_OK=TRUE python scripts/benchmark_*.py`.


**Reproducing the numbers.** CI runs the hashing fallback so it needs no model download; the published figures come from MiniLM. A nightly workflow (`.github/workflows/nightly.yml`) reruns the evaluation and benchmarks with the `[semantic]` extra and fails if the committed reports have drifted.

## Retrieval benchmark

Baseline retrieval against two-stage retrieve->rerank, by query group, on MiniLM
embeddings (regenerate: `KMP_DUPLICATE_LIB_OK=TRUE python scripts/benchmark_retrieval.py`,
full table in docs/retrieval_benchmark.md):

| Query group | hit@1 baseline | hit@1 reranked | hit@3 baseline |
|-------------|----------------|----------------|----------------|
| direct | 1.00 | 1.00 | 1.00 |
| paraphrased, dev | 1.00 | 1.00 | 1.00 |
| paraphrased, held-out | 0.64 | 0.64 | 1.00 |

The reranker adds nothing, and that is the finding. On the earlier hashing embedder it
appeared to add +0.36 on dev paraphrases and -0.20 on held-out ones, because its synonym
map had been written against the dev wording to compensate for a retriever that could not
match a paraphrase at all. With a semantic retriever there is nothing left to compensate
for. It ships off by default; a cross-encoder, scoring a (query, passage) pair rather than
matching words against a list, is the version of this stage worth running.

## Retrieval ablation

Two-stage retrieval pipeline:

```
Query ──> BM25 (sparse) ──┐
                          ├──> RRF fusion ──> Reranker ──> Top-K
Query ──> Dense (FAISS) ──┘
```


Each retrieval component's contribution on the golden set (regenerate:
`python scripts/benchmark_ablation.py`, full table in docs/retrieval_ablation.md):

| Config | hit@1 | hit@3 | note |
|--------|-------|-------|------|
| Dense only | 0.88 | 1.00 | |
| BM25 only | 0.47 | 0.73 | |
| Hybrid (RRF) | 0.69 | 0.91 | |
| Hybrid + reranker | 0.80 | 0.96 | (dev paraphrases; see the benchmark above) |

Dense captures paraphrased semantics; BM25 captures exact terms (equipment ids, units);
RRF fusion combines both; the reranker reorders the fused set so the right facet surfaces
first. Each stage's value is measured, not assumed.

## ANN index benchmark

FAISS index types over the building embeddings, showing the recall / latency / memory
trade-off that governs vector search at scale (regenerate:
`python scripts/benchmark_ann.py`, full table in docs/ann_benchmark.md):

| Index | Recall@10 | p50 (ms) | Memory (KB) |
|-------|-----------|----------|-------------|
| Flat (exact) | 1.000 | 0.030 | 1379 |
| HNSW (M=16) | 0.998 | 0.023 | 1508 |
| IVF (nlist=16, nprobe=8) | 0.998 | 0.019 | 1410 |
| IVF-PQ (m=8) | 0.621 | 0.013 | 59 |

Flat is exact; the approximate indexes trade recall for latency and/or memory. IVF
keeps near-exact recall by probing a subset of cells; IVF-PQ compresses vectors ~21x
(919 -> 43 KB) at a recall cost. At this corpus size Flat is already fast, so the value
is the methodology — the same harness scales to millions of vectors where these
trade-offs decide the design.

### HNSW parameter sweep

Recall climbs with efSearch/M but plateaus below exact on these hashing embeddings —
the honest lesson is that ANN index quality depends on the embedding manifold, not just
parameters (full sweep + analysis in docs/hnsw_sweep.md). IVF's coarse quantisation
beats HNSW's graph navigation here; with smooth semantic embeddings HNSW would recover.

### ANN approximation -> RAG answer quality

The question that actually matters: how much does approximate retrieval cost the *final
answer*? Measured end-to-end on the golden set (docs/ann_rag_quality.md):

| Index | Answer hit@4 | Drop vs exact |
|-------|--------------|---------------|
| Flat (exact) | 0.400 | +0.000 |
| HNSW | 0.340 | -0.060 |
| IVF | 0.180 | -0.220 |
| IVF-PQ | 0.180 | -0.220 |

This recall -> answer-quality transfer is the real production question: how much answer
quality you trade for lower latency/memory. The methodology scales to millions of
vectors where exact search is infeasible.

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
    tests/             unit, data, integration (76 tests)
    scripts/run_eval.py  offline eval -> MLflow + committed report
    docs/              eval report + CI/CD and drift runbooks

## Roadmap

- [x] Core RAG: ingest, retrieval, generation, groundedness guard
- [x] Offline eval harness + MLflow tracking + committed report
- [x] Test tiers: unit / data / integration (76 tests)
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
