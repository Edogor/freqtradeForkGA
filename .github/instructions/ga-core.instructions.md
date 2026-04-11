---
applyTo: "genetic_algorithm/{engine,advanced,evaluation,genome,core}/**"
description: "GA core engine conventions. Use when: editing evolution loop, backtesting, fitness evaluation, strategy generation, validation, or advanced evolution modes."
---

# GA Core Engine Conventions

**Module layout:** `core/` files are backward-compat shims. Real code lives in:
- `engine/` — runner, generation, checkpoint, adaptive mutation, hall_of_fame, islands, migration, population, warm_start
- `advanced/` — coevolution, ensemble_evolution, incremental, MAP-Elites, lifecycle_manager, parsimony
- `evaluation/` — fitness, direct_backtester, parallel, cache, surrogate, regime_aware
- `evaluation/validation/` — monte_carlo, cpcv, deflated_sharpe, param_sensitivity
- `genome/` — gene definitions (`gene.py`), code generation (`codegen.py`), indicators, seeds

**Evaluation chain:** `FitnessEvaluator` → `DirectBacktester` (Python API, never CLI)
- Cache backtest results by strategy hash (`evaluation/cache.py`)
- Parallel evaluation via `ProcessPoolExecutor` in `evaluation/parallel.py`
- Walk-forward and Monte Carlo are post-hoc on top-5 only (never in-loop)

**Strategy lifecycle:** `StrategyGene` (`genome/gene.py`) → codegen (`genome/codegen.py`) → `DirectBacktester` (results)
- Genes are dicts with indicator params, buy/sell logic, risk management
- Codegen produces valid FreqTrade `IStrategy` subclass code

**Evolution loop** (`engine/runner.py`):
- Checkpoint every 5 generations (`engine/checkpoint.py`)
- SIS hooks at: initialization (seeds), post-evaluation (immigrants, indicator bias), convergence check, weight adaptation
- Early stopping after `patience` generations with no improvement
- Holdout monitoring for overfitting detection (configurable interval + penalty)
- Adaptive mutation escalation on stagnation (`engine/adaptive.py`)

**Testing:** `pytest genetic_algorithm/tests/ -q` (50 test files, 1400+ tests)
