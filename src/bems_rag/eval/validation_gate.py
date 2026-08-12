"""Champion/challenger validation gate.

A challenger (newly evaluated RAG config) may only be promoted if it beats the current
champion on criteria fixed in advance -- evaluated on the same golden set. Losing on any
criterion stops promotion. This is the automated check that guards production: no model
ships just because its average looks good.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GateThresholds:
    """Promotion criteria, defined up front (never tuned post-hoc)."""
    min_hit_at_k_vs_champion: float = 0.0   # challenger.hit >= champion.hit + this
    max_mrr_regression: float = 0.0         # challenger.mrr >= champion.mrr - this
    min_groundedness: float = 1.0           # absolute floor: no invented numbers


# Module-level default so it is not constructed in the function signature.
DEFAULT_THRESHOLDS = GateThresholds()


@dataclass(frozen=True)
class CandidateMetrics:
    hit_at_k: float
    mrr: float
    groundedness: float
    # per-building hit@k, to catch a candidate that improves the average while
    # regressing specific tenants
    segment_hit: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class GateDecision:
    passed: bool
    reasons: list[str]


def evaluate_gate(
    champion: CandidateMetrics,
    challenger: CandidateMetrics,
    thresholds: GateThresholds | None = None,
) -> GateDecision:
    """Decide whether the challenger may be promoted, with human-readable reasons."""
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS
    reasons: list[str] = []

    # 1. Overall retrieval quality must not drop.
    if challenger.hit_at_k < champion.hit_at_k + thresholds.min_hit_at_k_vs_champion:
        reasons.append(
            f"hit@k {challenger.hit_at_k:.3f} < champion {champion.hit_at_k:.3f} "
            f"(+{thresholds.min_hit_at_k_vs_champion})"
        )

    # 2. Ranking quality must not regress beyond tolerance.
    if challenger.mrr < champion.mrr - thresholds.max_mrr_regression:
        reasons.append(
            f"MRR {challenger.mrr:.3f} regressed from champion {champion.mrr:.3f} "
            f"(max drop {thresholds.max_mrr_regression})"
        )

    # 3. Groundedness has an absolute floor -- safety, not just relative.
    if challenger.groundedness < thresholds.min_groundedness:
        reasons.append(
            f"groundedness {challenger.groundedness:.3f} < floor {thresholds.min_groundedness}"
        )

    # 4. No individual building may regress vs champion.
    for bid, champ_hit in champion.segment_hit.items():
        chall_hit = challenger.segment_hit.get(bid)
        if chall_hit is None:
            reasons.append(f"segment '{bid}' missing from challenger metrics")
        elif chall_hit < champ_hit:
            reasons.append(
                f"segment '{bid}' regressed ({champ_hit:.3f} -> {chall_hit:.3f})"
            )

    return GateDecision(passed=len(reasons) == 0, reasons=reasons)
