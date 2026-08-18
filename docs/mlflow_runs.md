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
