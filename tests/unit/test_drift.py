"""Unit tests for PSI-based query drift detection."""
import pytest

from bems_rag.monitoring.drift import (
    detect_drift,
    population_stability_index,
    query_lengths,
)

pytestmark = pytest.mark.unit


def test_no_drift_for_identical_distribution():
    ref = [3, 5, 7, 9, 11] * 20
    live = [3, 5, 7, 9, 11] * 20
    result = detect_drift(ref, live)
    assert not result.drifted
    assert result.severity == "none"
    assert result.psi < 0.1


def test_significant_drift_when_distribution_shifts():
    # Reference: short queries. Live: all long queries.
    ref = [2, 3, 2, 3, 2] * 20
    live = [20, 25, 30, 22, 28] * 20
    result = detect_drift(ref, live)
    assert result.drifted
    assert result.severity == "significant"
    assert result.psi >= 0.2


def test_psi_is_zero_ish_for_same_input():
    ref = [5, 5, 5, 10, 10] * 10
    assert population_stability_index(ref, ref) == pytest.approx(0.0, abs=1e-9)


def test_query_lengths_feature():
    assert query_lengths(["how much solar", "what is the hvac setpoint here"]) == [3, 6]


def test_threshold_is_configurable():
    # A mild shift: mostly overlapping, a few samples move buckets.
    ref = [3, 5, 5, 5, 7] * 20
    live = [3, 5, 5, 7, 7] * 20
    psi = detect_drift(ref, live).psi
    # Same PSI, different thresholds -> different verdicts.
    strict = detect_drift(ref, live, threshold=psi - 0.001)
    lenient = detect_drift(ref, live, threshold=psi + 0.001)
    assert strict.drifted
    assert not lenient.drifted
