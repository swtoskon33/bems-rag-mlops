"""Integration test for the CD promotion decision (evaluate -> gate)."""
import pytest

from bems_rag.eval.validation_gate import CandidateMetrics, evaluate_gate

pytestmark = pytest.mark.integration


def test_cd_promotes_when_challenger_beats_champion():
    champion = CandidateMetrics(hit_at_k=0.80, mrr=0.75, groundedness=1.0)
    challenger = CandidateMetrics(hit_at_k=0.90, mrr=0.85, groundedness=1.0)
    decision = evaluate_gate(champion, challenger)
    assert decision.passed


def test_cd_blocks_when_challenger_regresses():
    champion = CandidateMetrics(hit_at_k=0.90, mrr=0.85, groundedness=1.0)
    challenger = CandidateMetrics(hit_at_k=0.70, mrr=0.60, groundedness=1.0)
    decision = evaluate_gate(champion, challenger)
    assert not decision.passed
    assert decision.reasons  # explains why it was blocked


def test_cd_blocks_on_groundedness_floor():
    champion = CandidateMetrics(hit_at_k=0.90, mrr=0.85, groundedness=1.0)
    challenger = CandidateMetrics(hit_at_k=0.95, mrr=0.90, groundedness=0.80)
    decision = evaluate_gate(champion, challenger)
    assert not decision.passed
