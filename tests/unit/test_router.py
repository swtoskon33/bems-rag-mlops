"""Unit tests for the champion/challenger deployment router."""
import pytest

from bems_rag.serving.router import (
    RoutingConfig,
    Stage,
    route_to_challenger,
    runs_challenger,
)

pytestmark = pytest.mark.unit

# A spread of tenant ids to check routing proportions.
TENANTS = [f"building_{i:03d}" for i in range(100)]


def test_shadow_never_serves_challenger():
    cfg = RoutingConfig(stage=Stage.SHADOW)
    assert all(not route_to_challenger(t, cfg) for t in TENANTS)


def test_shadow_still_runs_challenger_for_logging():
    cfg = RoutingConfig(stage=Stage.SHADOW)
    assert all(runs_challenger(t, cfg) for t in TENANTS)


def test_full_serves_challenger_to_everyone():
    cfg = RoutingConfig(stage=Stage.FULL)
    assert all(route_to_challenger(t, cfg) for t in TENANTS)


def test_canary_serves_roughly_the_rollout_fraction():
    cfg = RoutingConfig(stage=Stage.CANARY, rollout_pct=25)
    served = sum(route_to_challenger(t, cfg) for t in TENANTS)
    # deterministic hashing won't be exactly 25, but should be in a sane band
    assert 15 <= served <= 35


def test_canary_zero_pct_serves_nobody():
    cfg = RoutingConfig(stage=Stage.CANARY, rollout_pct=0)
    assert not any(route_to_challenger(t, cfg) for t in TENANTS)


def test_canary_routing_is_deterministic():
    cfg = RoutingConfig(stage=Stage.CANARY, rollout_pct=50)
    first = [route_to_challenger(t, cfg) for t in TENANTS]
    second = [route_to_challenger(t, cfg) for t in TENANTS]
    assert first == second


def test_canary_rollout_is_monotonic():
    # A tenant served at 25% must also be served at any higher percentage.
    t = "building_042"
    served_at = [
        route_to_challenger(t, RoutingConfig(stage=Stage.CANARY, rollout_pct=p))
        for p in (0, 25, 50, 75, 100)
    ]
    # once True, stays True as pct increases
    assert served_at == sorted(served_at)
