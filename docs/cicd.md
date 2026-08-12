# CI/CD Walkthrough

How a change reaches production in this repo, and what protects each step.

## CI (on every pull request)

1. **PR + review** — nothing merges to `main` without review and green CI.
2. **Tests, three tiers**
   - unit: code correctness (metrics, groundedness guard, routing, drift)
   - data: BDG2 quality gates (schema, unique/non-null ids, ranges) -- broken data
     fails here, before it can reach retrieval
   - integration: end-to-end answer, tenant isolation, serving routing
3. **Lint** (ruff) and **coverage** on the full suite.
4. **Validation gate** (offline): a challenger config is compared to the champion on the
   golden set -- no hit@k drop, no MRR regression, groundedness floor, no per-building
   regression. Losing stops promotion.

## CD (after the gate passes)

Staged rollout, controlled by the serving `RoutingConfig`:

1. **Shadow** -- challenger runs on real traffic but its answers are only logged; users
   still see the champion. Zero-risk comparison.
2. **Canary** -- a deterministic slice of tenants (e.g. 5% -> ~2 of 31 buildings) is
   served by the challenger. Watch latency, errors, groundedness, business metrics.
3. **Staged rollout** -- raise 5 -> 25 -> 50 -> 100%; routing is monotonic so a tenant
   already on the challenger stays on it. On full, the challenger takes the `champion`
   alias in the registry.
4. **Auto-rollback** -- if an SLO breaks at any stage, flip the alias back to the
   previous champion. Minutes, no code redeploy; if it broke at 5%, 95% never saw it.

## Reproducibility

The same Docker image (tagged by commit SHA) is used for eval and serve, so what was
evaluated is byte-for-byte what runs.
