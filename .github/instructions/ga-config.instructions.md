---
applyTo: "genetic_algorithm/config/**/*.yaml"
description: "Canonical GA experiment config constraints and validation rules."
---

# GA Config Contract

Every config must pass the project resolver:

```bash
python -m genetic_algorithm config validate <config.yaml>
```

The validator enforces only mechanical invariants:

- `population_size` is between 2 and the implementation guard of 10,000.
- `0 <= elite_size < population_size`.
- `1 <= tournament_size <= population_size`.
- `random_immigrants` fits in the non-elite slots.
- probabilities are finite and in `[0, 1]`.
- runtime dispatch values name implemented modes, selectors and crossovers.
- pair lists are non-empty, unique and contain non-empty strings.
- explicit worker counts are positive.
- classic `island_model` cannot request walk-forward because that engine disables it.

Do not present population size, island size, pair count, elite ratio, tournament size, mutation
rate or generation count as proven profitable ranges. They are experimental factors. Compare them
with paired seeds, identical panels and budgets through the V2 planner/analyzer. Historical
single-wave observations belong in hypotheses, not startup gates.

Automatic waves currently require the enforced `safe_v2` profile. Unsupported features remain
disabled there until their paired out-of-sample evidence passes the promotion contract.
