"""Deployment router: champion/challenger traffic splitting for shadow and canary.

Implements the routing decisions behind staged rollout:
  - SHADOW: every user request is served by the champion; the challenger also runs but
    its output is only logged (0% user-facing). Zero-risk real-traffic evaluation.
  - CANARY: a fixed fraction of tenants are served by the challenger for real; the rest
    stay on champion. Rollout is raised gradually (5 -> 25 -> 50 -> 100%).

Tenant routing is deterministic (hash of building_id), so a given building has a stable
experience within a rollout percentage instead of flipping per request.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum


class Stage(str, Enum):
    SHADOW = "shadow"      # 0% user-facing; challenger logged only
    CANARY = "canary"      # rollout_pct of tenants served by challenger
    FULL = "full"          # 100% challenger (post-promotion)


@dataclass(frozen=True)
class RoutingConfig:
    stage: Stage = Stage.SHADOW
    rollout_pct: int = 0   # only used in CANARY; 0-100


def _tenant_bucket(building_id: str) -> int:
    """Stable 0-99 bucket for a tenant, from a hash of its id."""
    h = int(hashlib.md5(building_id.encode()).hexdigest(), 16)
    return h % 100


def route_to_challenger(building_id: str, config: RoutingConfig) -> bool:
    """Return True if this tenant's user-facing answer should come from the challenger."""
    if config.stage is Stage.FULL:
        return True
    if config.stage is Stage.SHADOW:
        return False  # never user-facing in shadow
    # CANARY: deterministic slice of tenants by bucket.
    return _tenant_bucket(building_id) < config.rollout_pct


def runs_challenger(building_id: str, config: RoutingConfig) -> bool:
    """Whether the challenger executes at all (for logging/metrics).

    In shadow it always runs (logged, not served); otherwise it runs when it is the
    one being served.
    """
    if config.stage is Stage.SHADOW:
        return True
    return route_to_challenger(building_id, config)
