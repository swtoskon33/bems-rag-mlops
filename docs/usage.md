# Usage examples

Practical ways to query and operate the system. Start the API first:

    python scripts/serve.py        # http://localhost:8000

## 1. Ask a question (curl)

    curl -X POST localhost:8000/answer \
      -H "Content-Type: application/json" \
      -d '{"building_id": "Panther_lodging_Dean", "text": "what is the floor area?"}'

Response:

    {
      "text": "Building Panther_lodging_Dean has a floor area of 508.8 square meters.",
      "grounded": true,
      "served_by": "champion",
      "contexts": ["Panther_lodging_Dean_area"]
    }

## 2. Python client

    import requests

    def ask(building_id: str, question: str, url: str = "http://localhost:8000"):
        r = requests.post(f"{url}/answer",
                          json={"building_id": building_id, "text": question})
        r.raise_for_status()
        return r.json()

    ans = ask("Panther_lodging_Dean", "when was it built?")
    print(ans["text"], "| grounded:", ans["grounded"])

## 3. Ungrounded answers are flagged

A question whose answer isn't in the retrieved context returns `grounded: false`,
so downstream code can withhold or caveat it:

    ans = ask("Panther_lodging_Dean", "who is the facility manager?")
    if not ans["grounded"]:
        print("No grounded answer — not in the building's data.")

## 4. Tenant isolation

Each building only sees its own chunks. Asking building A about building B's data
returns an ungrounded answer, not B's figures — a core multi-tenant guarantee.

## 5. Run the evaluation

    python scripts/run_eval.py        # writes docs/eval_report.md + logs to MLflow

Metrics are defined in [metrics.md](metrics.md).

## 6. Switch to real models (production path)

Backends are pluggable via environment variables — no code change:

    export EMBEDDING_BACKEND=openai
    export GENERATION_BACKEND=openai
    export OPENAI_API_KEY=sk-...
    python scripts/serve.py

Offline (default) uses a deterministic hashing embedder and a template generator, so
tests and CI run with no API key.
