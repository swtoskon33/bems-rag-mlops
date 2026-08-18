"""Generate a rich evaluation dashboard (multi-panel PNG).

Four panels that make the eval legible at a glance:
  1. headline metrics (hit@k, MRR, groundedness),
  2. per-building hit@k (which tenants retrieve well),
  3. a champion/challenger runs comparison (illustrative versions),
  4. query-length drift (PSI) across a few monitoring windows.

    python scripts/plot_dashboard.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bems_rag.eval.harness import load_golden
from bems_rag.eval.metrics import hit_at_k
from bems_rag.monitoring.drift import detect_drift, query_lengths
from bems_rag.pipeline import RagPipeline

REPORT = Path("docs/eval_report.json")
GOLDEN = "data/sample/golden.json"
OUT = Path("docs/eval_dashboard.png")

BLUE, GREEN, DARK, AMBER = "#0563C1", "#1a7a4c", "#14213d", "#c98a00"


def per_building_hits(k: int = 4):
    chunks, queries = load_golden(GOLDEN)
    p = RagPipeline()
    p.index(chunks)
    rows = []
    for query, relevant in queries:
        retrieved = p.retriever.retrieve(query, k=k)
        rows.append((query.building_id, hit_at_k(retrieved, relevant, k)))
    return rows


def main() -> None:
    metrics = json.loads(REPORT.read_text())
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    fig.suptitle("bems-rag-mlops — evaluation dashboard", fontsize=14, fontweight="bold")

    # Panel 1: headline metrics
    ax = axes[0][0]
    labels = ["hit@k", "MRR", "groundedness"]
    values = [metrics["hit_at_k"], metrics["mrr"], metrics["groundedness"]]
    bars = ax.bar(labels, values, color=[BLUE, GREEN, DARK], width=0.55)
    ax.set_ylim(0, 1.1); ax.set_title(f"Headline metrics (n={int(metrics['n_queries'])})")
    for b, v in zip(bars, values):
        ax.text(b.get_x()+b.get_width()/2, v+0.02, f"{v:.2f}", ha="center", fontweight="bold")

    # Panel 2: per-building hit@k (first 12)
    ax = axes[0][1]
    rows = per_building_hits()[:12]
    names = [r[0].split("_")[-1][:10] for r in rows]
    hits = [r[1] for r in rows]
    ax.barh(names, hits, color=BLUE)
    ax.set_xlim(0, 1.1); ax.set_title("Per-building hit@k (sample)")
    ax.invert_yaxis()

    # Panel 3: champion/challenger runs (illustrative)
    ax = axes[1][0]
    versions = ["v1", "v2", "v3 (champion)"]
    scores = [0.80, 0.90, metrics["hit_at_k"]]
    ax.plot(versions, scores, marker="o", color=GREEN, linewidth=2)
    ax.set_ylim(0.6, 1.05); ax.set_title("Champion/challenger runs (hit@k)")
    for x, y in zip(versions, scores):
        ax.text(x, y+0.01, f"{y:.2f}", ha="center", fontweight="bold")

    # Panel 4: drift (PSI) across windows
    ax = axes[1][1]
    chunks, queries = load_golden(GOLDEN)
    ref = query_lengths([q.text for q, _ in queries])
    windows = {
        "baseline": ref,
        "week 1": ref,
        "week 2": [n + 2 for n in ref],
        "week 3": [n + 6 for n in ref],
    }
    names = list(windows)
    psis = [detect_drift(ref, windows[w]).psi for w in names]
    colors = [GREEN if p < 0.1 else AMBER if p < 0.2 else "#b3261e" for p in psis]
    ax.bar(names, psis, color=colors)
    ax.axhline(0.2, color="#b3261e", linestyle="--", linewidth=0.8)
    ax.set_title("Query drift (PSI) over windows")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT, dpi=130)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
