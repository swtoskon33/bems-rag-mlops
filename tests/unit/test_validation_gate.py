"""Unit tests for the champion/challenger validation gate."""
import pytest

from bems_rag.eval.validation_gate import (
    CandidateMetrics,
    GateThresholds,
    evaluate_gate,
)

pytestmark = pytest.mark.unit


def _m(hit=1.0, mrr=0.9, grounded=1.0, seg=None):
    return CandidateMetrics(hit, mrr, grounded, seg or {})


def test_challenger_wins_when_better_or_equal():
    champ = _m(hit=0.90, mrr=0.85, grounded=1.0)
    chall = _m(hit=0.95, mrr=0.88, grounded=1.0)
    decision = evaluate_gate(champ, chall)
    assert decision.passed
    assert decision.reasons == []


def test_blocks_on_hit_at_k_drop():
    champ = _m(hit=0.90)
    chall = _m(hit=0.80)
    decision = evaluate_gate(champ, chall)
    assert not decision.passed
    assert any("hit@k" in r for r in decision.reasons)


def test_blocks_on_mrr_regression():
    champ = _m(mrr=0.90)
    chall = _m(mrr=0.70)
    decision = evaluate_gate(champ, chall)
    assert not decision.passed
    assert any("MRR" in r for r in decision.reasons)


def test_blocks_on_groundedness_below_floor():
    champ = _m(grounded=1.0)
    chall = _m(grounded=0.8)
    decision = evaluate_gate(champ, chall)
    assert not decision.passed
    assert any("groundedness" in r for r in decision.reasons)


def test_blocks_when_a_single_building_regresses():
    # Challenger improves the average but one building gets worse.
    champ = _m(hit=0.90, seg={"b1": 1.0, "b2": 0.8})
    chall = _m(hit=0.95, seg={"b1": 0.5, "b2": 1.0})
    decision = evaluate_gate(champ, chall)
    assert not decision.passed
    assert any("b1" in r for r in decision.reasons)


def test_custom_thresholds_require_strict_improvement():
    champ = _m(hit=0.90)
    chall = _m(hit=0.90)  # equal
    strict = GateThresholds(min_hit_at_k_vs_champion=0.05)
    decision = evaluate_gate(champ, chall, strict)
    assert not decision.passed
