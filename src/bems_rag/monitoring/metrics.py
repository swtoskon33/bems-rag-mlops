"""Prometheus metrics for the serving layer.

Exposes the counters/histograms a production dashboard needs: request volume by which
model served, answer latency, groundedness outcomes, and the latest drift score. These
are scraped by Prometheus and visualised in Grafana; SLO breaches (e.g. p99 latency or
groundedness rate) drive alerts.
"""
from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# Requests split by which model served them -- lets you watch canary volume ramp.
REQUESTS = Counter(
    "bems_rag_requests_total",
    "Total answer requests",
    ["served_by"],
)

# Answer latency; histogram so Prometheus can compute p50/p95/p99.
LATENCY = Histogram(
    "bems_rag_answer_latency_seconds",
    "End-to-end answer latency",
    ["served_by"],
)

# Grounded vs ungrounded answers -- a groundedness SLO alerts on this.
GROUNDED = Counter(
    "bems_rag_answers_grounded_total",
    "Answers by groundedness outcome",
    ["grounded"],
)

# Latest query-drift PSI, updated by the drift job.
DRIFT_PSI = Gauge(
    "bems_rag_query_drift_psi",
    "Latest population stability index for query length",
)


def record_request(served_by: str, latency_seconds: float, grounded: bool) -> None:
    """Record one served answer across all relevant metrics."""
    REQUESTS.labels(served_by=served_by).inc()
    LATENCY.labels(served_by=served_by).observe(latency_seconds)
    GROUNDED.labels(grounded=str(grounded).lower()).inc()


def set_drift(psi: float) -> None:
    """Publish the latest drift score for dashboards/alerts."""
    DRIFT_PSI.set(psi)
