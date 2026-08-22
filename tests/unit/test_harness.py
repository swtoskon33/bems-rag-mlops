"""Unit tests for the eval harness: evaluate() computes hit@k, MRR, groundedness."""
from bems_rag.eval.harness import evaluate
from bems_rag.pipeline import RagPipeline


def test_evaluate_returns_bounded_metrics():
    report = evaluate(RagPipeline(), "data/sample/golden.json", k=4)
    # every metric is a fraction in [0, 1]
    for value in (report.hit_at_k, report.mrr, report.groundedness):
        assert 0.0 <= value <= 1.0


def test_evaluate_is_deterministic():
    # offline hashing embedder + template generator -> identical runs
    r1 = evaluate(RagPipeline(), "data/sample/golden.json", k=4)
    r2 = evaluate(RagPipeline(), "data/sample/golden.json", k=4)
    assert r1.hit_at_k == r2.hit_at_k
    assert r1.mrr == r2.mrr
    assert r1.groundedness == r2.groundedness


def test_evaluate_hit_at_k_monotonic_in_k():
    # larger k can only find at least as many relevant chunks
    low = evaluate(RagPipeline(), "data/sample/golden.json", k=1)
    high = evaluate(RagPipeline(), "data/sample/golden.json", k=5)
    assert high.hit_at_k >= low.hit_at_k
