# Drift Runbook

The drift loop is automated: **detect -> alert -> retrain -> gate -> canary -> promote**,
with a human only supervising dashboards.

## Detection
- Continuously compare production query distribution against a training/baseline window
  using PSI (`monitoring/drift.py`).
- PSI bands: `< 0.1` none, `0.1-0.2` moderate (watch), `> 0.2` significant (act).
- Crossing the threshold raises an automatic alert -- no human staring at charts.

## Until labels arrive
Ground truth is delayed, so watch **proxy metrics** that indirectly show model health
(e.g. actual consumption vs forecast; user-reported bad answers).

## Reaction
- The alert triggers a retraining pipeline on fresh data.
- The new challenger must pass the **validation gate** (`eval/validation_gate.py`) --
  it only proceeds if it beats the champion, with no per-building regression.

## Back to production
- On winning, the challenger goes live via **canary** (shadow -> 5% -> staged -> full).
- On any SLO breach, **auto-rollback** via alias flip.
- The full cycle runs with no manual step; the operator supervises Grafana.

## Diagnose before retraining
Not every shift needs a retrain:
- A broken data source (e.g. a stuck sensor) = fix the source, not the model.
- A genuine distribution shift (e.g. seasonal change in what operators ask) = retrain,
  optionally add the new failure cases as hard negatives.

## Prevent
Known seasonal drift -> scheduled retraining at season boundaries, ahead of the shift.
