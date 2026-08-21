# RAG Query Playground (frontend)

A small React + Vite UI for the bems-rag-mlops service: pick a building, ask a
question, and see the grounded, tenant-scoped answer returned by the FastAPI backend.

## Run

Start the backend (from the repo root):

    python scripts/serve.py        # serves on http://localhost:8000

Then the frontend:

    cd frontend
    npm install
    npm run dev                    # http://localhost:5173

Point the UI at a different backend with `VITE_API_URL`:

    VITE_API_URL=http://localhost:8000 npm run dev

## What it shows

- Building selector (real BDG2 building ids) and a free-text question box.
- The grounded answer, with a `grounded` / `ungrounded` badge and which model
  served it (champion vs challenger).
- The retrieved context chunk ids backing the answer.
