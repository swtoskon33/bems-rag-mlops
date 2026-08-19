"""Generate an evaluation dashboard (PNG) from the committed eval results.

Reads docs/eval_report.json and renders a simple, clean metrics chart so the results
are visible at a glance in the README without running anything.

    python scripts/plot_eval.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless, no display needed
import matplotlib.pyplot as plt

REPORT = Path("docs/eval_report.json")
OUT = Path("docs/eval_dashboard.png")


def main() -> None:
    metrics = json.loads(REPORT.read_text())

    labels = ["hit@k", "MRR", "groundedness"]
    values = [metrics["hit_at_k"], metrics["mrr"], metrics["groundedness"]]
    n_queries = int(metrics["n_queries"])

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color=["#0563C1", "#1a7a4c", "#14213d"], width=0.55)

    ax.set_ylim(0, 1.08)
    ax.set_ylabel("score")
    ax.set_title(f"RAG evaluation on {n_queries} real-building queries (BDG2)")
    ax.axhline(y=1.0, color="#cccccc", linestyle="--", linewidth=0.8, zorder=0)

    # value labels on top of bars
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.2f}",
                ha="center", va="bottom", fontweight="bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT, dpi=130)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
