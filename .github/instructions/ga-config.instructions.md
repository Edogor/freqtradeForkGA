---
applyTo: "genetic_algorithm/config/**/*.yaml"
description: "GA experiment config constraints and validation rules. Use when: editing or reviewing YAML configs for GA experiments."
---

# GA Config Constraints

When editing experiment configs, enforce these verified constraints:

**Population sizing:**
- Standard GA: `population_size` 10-15 (>15 → 59-65% overfitting)
- Island model: ≥ 60 per island (`population_size` × `num_islands` ≥ 60 per island)

**Selection pressure:**
- `tournament_size` must be ≥ 3 (1 = random, >6 = premature convergence)
- `elite_size` ≈ 10% of `population_size`

**Incompatible combinations (never allow):**
- Island model + walk-forward together (data partitioning conflict)
- Fitness sharing + NSGA-II mode (distorts Pareto front)

**Required settings:**
- `enable_cache: true` unless debugging cache issues
- `auto_download: false` only with verified data presence

See [GA_CONFIG_CHEATSHEET.md](../../GA_CONFIG_CHEATSHEET.md) for full parameter reference.
