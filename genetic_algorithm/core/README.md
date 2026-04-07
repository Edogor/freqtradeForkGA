# core/ — Backward Compatibility Layer

This directory contains **shim modules** that re-export symbols from their
new locations in `engine/`, `genome/`, `evaluation/`, and `advanced/`.

## Why do shims exist?

The GA codebase was restructured from a flat `core/` directory into a
modular architecture. To avoid breaking existing imports, each moved file
left behind a shim that re-exports everything from the new location:

```python
# Example: core/crossover.py (shim)
"""Backward-compatibility shim — real code moved to engine.operators.crossover."""
from genetic_algorithm.engine.operators.crossover import *
```

## File classification

### Shim files (re-exports only)
| File | Real location |
|------|--------------|
| `strategy_gene.py` | `genome/gene.py` |
| `individual.py` | `genome/individual.py` |
| `seed_strategies.py` | `genome/seeds.py` |
| `crossover.py` | `engine/operators/crossover.py` |
| `mutation.py` | `engine/operators/mutation.py` |
| `selection.py` | `engine/operators/selection.py` |
| `culling.py` | `engine/operators/culling.py` |
| `population.py` | `engine/population.py` |
| `nsga2.py` | `engine/nsga2.py` |
| `operator_selection.py` | `engine/operators/adaptive.py` |
| `warm_start.py` | `engine/warm_start.py` |
| `pareto_archive.py` | `engine/pareto_archive.py` |
| `experiment_tracker.py` | `engine/experiment_tracker.py` |
| `feature_importance.py` | `engine/feature_importance.py` |
| `hall_of_fame.py` | `engine/hall_of_fame.py` |
| `coevolution.py` | `advanced/coevolution.py` |
| `ensemble_evolution.py` | `advanced/ensemble_evolution.py` |
| `incremental_evolution.py` | `advanced/incremental.py` |
| `map_elites.py` | `advanced/map_elites.py` |
| `parsimony.py` | `advanced/parsimony.py` |
| `lifecycle_manager.py` | `advanced/lifecycle_manager.py` |

### Real implementation files
| File | Purpose | Delegates to |
|------|---------|-------------|
| `evolution.py` | Main `GeneticAlgorithm` class | `engine/runner.py`, `engine/generation.py`, `engine/checkpoint.py`, `engine/adaptive.py` |
| `island_model.py` | Regime-based island model | — |
| `generic_island_model.py` | Generic island model | — |

## When to import from `core/` vs new locations

- **New code**: Import from the new location (e.g. `from genetic_algorithm.engine.population import Population`)
- **Existing code**: Imports from `core/` still work via shims
- **Deprecation plan**: Shims will be removed in a future version after all internal imports are updated

## Do NOT add new files here

All new code should go in the appropriate module:
- `engine/` — evolution engine components
- `genome/` — strategy representation
- `evaluation/` — fitness scoring
- `advanced/` — optional features
- `orchestration/` — experiment management
