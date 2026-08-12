"""Drift detection for production query distribution.

Compares a live window of queries against a reference (training/baseline) distribution
using Population Stability Index (PSI) over a simple feature -- here, query length
buckets as a lightweight proxy. When PSI crosses a threshold, drift is flagged and (in
the full pipeline) triggers retraining.

PSI is the standard, interpretable drift metric:
  < 0.1  no significant shift
  0.1-0.2 moderate shift (watch)
  > 0.2  significant shift (act)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# Query-length buckets (word count) used as the drift feature.
_BUCKETS = [(0, 4), (4, 8), (8, 12), (12, 1_000)]


def _bucketize(lengths: list[int]) -> list[float]:
    """Return the fraction of samples in each bucket (smoothed to avoid zeros)."""
    counts = [0] * len(_BUCKETS)
    for n in lengths:
        for i, (lo, hi) in enumerate(_BUCKETS):
            if lo <= n < hi:
                counts[i] += 1
                break
    total = len(lengths)
    # Laplace smoothing so empty buckets don't blow up the log ratio.
    return [(c + 1) / (total + len(_BUCKETS)) for c in counts]


def population_stability_index(reference: list[int], live: list[int]) -> float:
    """PSI between a reference and a live set of query lengths."""
    ref = _bucketize(reference)
    cur = _bucketize(live)
    return sum((c - r) * math.log(c / r) for r, c in zip(ref, cur))


@dataclass(frozen=True)
class DriftResult:
    psi: float
    drifted: bool
    severity: str   # "none" | "moderate" | "significant"


def detect_drift(reference: list[int], live: list[int], threshold: float = 0.2) -> DriftResult:
    """Compute PSI on query lengths and classify severity."""
    psi = population_stability_index(reference, live)
    if psi < 0.1:
        severity = "none"
    elif psi < 0.2:
        severity = "moderate"
    else:
        severity = "significant"
    return DriftResult(psi=psi, drifted=psi >= threshold, severity=severity)


def query_lengths(texts: list[str]) -> list[int]:
    """Word-count feature for a batch of query texts."""
    return [len(t.split()) for t in texts]
