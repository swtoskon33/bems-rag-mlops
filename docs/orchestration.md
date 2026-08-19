# Orchestration (Dagster)

The retraining/promotion loop is modelled as a Dagster asset graph, so the flow is a
real orchestrated DAG rather than a script.

## Assets

    eval_report  ->  gate_decision  ->  promotion

- **eval_report** - evaluate the current RAG pipeline on the golden set (hit@k, MRR,
  groundedness).
- **gate_decision** - compare the challenger to the champion baseline via the
  validation gate (blocks on regression).
- **promotion** - promote the challenger to champion if the gate passed, otherwise
  keep the champion.

A `retrain_and_promote` job materialises the whole graph, and a daily schedule
(`0 2 * * *`) runs it automatically - the drift/retrain loop expressed as orchestration.

## Run it

    # UI
    dagster dev -f src/bems_rag/orchestration/dagster_pipeline.py

    # or headless, materialise all assets
    dagster asset materialize --select "*" \
      -f src/bems_rag/orchestration/dagster_pipeline.py
