# Config Validator & Anti-Pattern Reference

> The GA config validator (`utils/config_validator.py`) runs **before** evolution starts. It catches misconfigurations that would waste hours of compute time or silently produce bad results. Validated against 150+ experiments.

---

## Table of Contents

1. [How It Works](#1-how-it-works)
2. [Errors vs Warnings](#2-errors-vs-warnings)
3. [Anti-Pattern Catalogue (AP-1 to AP-20)](#3-anti-pattern-catalogue-ap-1-to-ap-20)
4. [Preflight Checks](#4-preflight-checks)
5. [Pair-Split Validation](#5-pair-split-validation)
6. [Data Availability Checks](#6-data-availability-checks)
7. [Integrating the Validator](#7-integrating-the-validator)

---

## 1. How It Works

`validate_ga_config(config)` accepts the full configuration dictionary and returns `(errors, warnings)`:

```python
from genetic_algorithm.utils.config_validator import validate_ga_config

errors, warnings = validate_ga_config(config)
for w in warnings:
    print(f"[WARN] {w}")
if errors:
    for e in errors:
        print(f"[ERROR] {e}")
    sys.exit(1)
```

`run_ga.py` calls `validate_config()` before any GA object is created, so the process exits cleanly with an actionable error rather than crashing mid-run.

---

## 2. Errors vs Warnings

| Level | Meaning | Behaviour |
|-------|---------|-----------|
| **ERROR** | Fatal misconfiguration — evolution **will** crash or produce meaningless results | Blocks execution |
| **WARNING** | Suboptimal configuration — evolution runs but may produce poor results or overfitting | Logged; execution continues |

---

## 3. Anti-Pattern Catalogue (AP-1 to AP-20)

Each anti-pattern code is printed in warning/error messages so you can look it up here.

| Code | Trigger | Impact | Fix |
|------|---------|--------|-----|
| **AP-2** | `population_size > 15` in standard GA | 59–65% holdout degradation (E19) | Keep ≤ 15 for standard GA |
| **AP-3** | `crossover_method: component` on island model | 33–53% holdout degradation (E18) | Use `uniform` or `single_point` |
| **AP-4** | `walk_forward.train_days > 150` | Walk-forward windows become too large, train/val gap shrinks | Use 90–120 days |
| **AP-5** | LLM guidance + `mutation_rate > 0.20` | Over-mutates LLM-seeded genes, wastes LLM quality | Cap mutation at 0.18 with LLM |
| **AP-6** | `population_per_island > 6` | 62–100% holdout degradation on island (E24) | Strict ≤ 6 per island |
| **AP-7** (NSGA-II) | `mode: nsga2` | Near-zero fitness (0.0008) in current implementation | Do not use until fixed |
| **AP-8** | NSGA-II + `fitness_sharing` | NSGA-II has its own diversity mechanism; sharing double-penalises | Disable sharing with nsga2 |
| **AP-9** | Island model + walk-forward enabled | Walk-forward silently disabled inside island mode | Do not set WF in island configs |
| **AP-10** | Island model + Monte Carlo/CPCV | Extra overhead, incompatible at island level | Disable MC/CPCV in island configs |
| **AP-11** | `monte_carlo.num_permutations > 20` | Premature convergence, excessive compute | Use 10–15 permutations |
| **AP-12** | `> 2 pairs` in standard GA (no pair-split) | Generalisation failure (E32: 0 SAFE, 5 WARNING) | 1–2 pairs for standard GA |
| **AP-13** | `population_size > 15` (alias of AP-2) | Same as AP-2 | Keep ≤ 15 |
| **AP-14** | `convergence_patience ≤ 4` + `elite_size = 2` | Premature stopping; elites not preserved long enough | Use patience ≥ 6 or elite ≥ 3 |
| **AP-15** | LLM guidance + `selection_method: rank` | Negative synergy; rank kills LLM diversity contributions | Use `tournament` with LLM |
| **AP-17** | `convergence_patience ≥ 8` + `elite_size < 3` | Holdout degradation from under-preserved elites | Use `elite_size ≥ 3` when patience ≥ 8 |
| **AP-20** | `mutation_rate ≥ 0.18` + tournament + `elite_size ≤ 2` | Early stop + holdout degradation | Increase elite_size to ≥ 3 |
| **AP-10** (short) | Short selling + `population_size < 25` | Search space doubles; insufficient exploration | Disable short or increase pop to ≥ 25 |

> AP numbers may have gaps — the numbering reflects experiment history, not a sequential checklist.

---

## 4. Preflight Checks

Before anti-pattern checks, the validator runs structural validity checks:

- `genetic_algorithm` section present and valid
- `backtesting.pairs` non-empty
- `backtesting.timerange` in `YYYYMMDD-YYYYMMDD` format
- `fitness_weights` keys are valid canonical names (with alias deprecation warnings)
- `elite_size < population_size`
- `tournament_size <= population_size`

---

## 5. Pair-Split Validation

When `island_model.pair_split: true` is set, the validator checks:

- Enough pairs exist for the number of islands (warns if `len(pairs) < num_islands`)
- Validates each island would receive at least one pair
- Checks for common mismatches between `num_islands` and `len(pairs)`

The pair-split test suite lives in `tests/test_pair_split_validation.py` (437 test cases).

---

## 6. Data Availability Checks

The validator warns (not errors) when:

- `backtesting.datadir` is unset (uses FreqTrade default)
- `backtesting.timerange` extends further back than available data files (if `datadir` is set)
- Multi-timeframe indicators reference timeframes that have no data files

---

## 7. Integrating the Validator

The validator is automatically called by `run_ga.py` before evolution starts. For manual use:

```python
from genetic_algorithm.utils.config_validator import validate_ga_config
import yaml

with open('my_config.yaml') as f:
    config = yaml.safe_load(f)

errors, warnings = validate_ga_config(config)

for w in warnings:
    print(f"⚠️  {w}")
for e in errors:
    print(f"❌ {e}")
```

You can also call the specialised sub-validators directly:

```python
from genetic_algorithm.utils.config_validator import (
    validate_anti_patterns,   # AP-1 to AP-20 checks
    validate_pair_split,      # Island pair-split consistency
    validate_data_availability # Data file existence checks
)
```

---

*Last updated: April 2026*
