"""Unit tests for embedding-based drift detection.

Note: with the deterministic hashing embedder, semantic closeness is weak (different
wordings hash apart). We therefore assert the RELATIVE ordering -- unrelated topics
drift more than reworded-but-related ones -- which is the property that matters and is
embedder-independent. With a real semantic embedder the absolute thresholds separate
cleanly too.
"""
import pytest

from bems_rag.monitoring.embedding_drift import detect_embedding_drift

pytestmark = pytest.mark.unit

REF = ["what is the floor area", "when was it built", "what energy sources"]


def test_identical_queries_have_near_zero_drift():
    result = detect_embedding_drift(REF, REF)
    assert result.centroid_distance < 1e-6
    assert not result.drifted


def test_unrelated_topics_drift_more_than_related():
    related = ["what is the building size", "what year constructed", "which meters"]
    unrelated = ["stock prices today", "football scores", "pizza recipe"]
    d_related = detect_embedding_drift(REF, related).centroid_distance
    d_unrelated = detect_embedding_drift(REF, unrelated).centroid_distance
    assert d_unrelated > d_related


def test_threshold_controls_flag():
    unrelated = ["stock prices", "football", "pizza"]
    dist = detect_embedding_drift(REF, unrelated).centroid_distance
    # same distance, thresholds on either side -> opposite verdicts
    strict = detect_embedding_drift(REF, unrelated, threshold=dist - 0.01)
    lenient = detect_embedding_drift(REF, unrelated, threshold=dist + 0.01)
    assert strict.drifted
    assert not lenient.drifted
