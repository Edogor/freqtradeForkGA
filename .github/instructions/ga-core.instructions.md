---
applyTo: "genetic_algorithm/{evolution,direct_backtester,parallel,walk_forward,monte_carlo,strategy_gene,generator,fitness,surrogate}*"
description: "GA core engine conventions. Use when: editing evolution loop, backtesting, fitness evaluation, strategy generation, or validation code."
---

# GA Core Engine Conventions

**Evaluation chain:** `FitnessEvaluator` → `DirectBacktester` (Python API, never CLI)
- Cache backtest results by strategy hash
- Parallel evaluation via `ProcessPoolExecutor` in `parallel.py`
- Walk-forward and Monte Carlo are post-hoc on top-5 only (never in-loop)

**Strategy lifecycle:** `StrategyGene` (genome) → `StrategyGenerator` (Python code) → `DirectBacktester` (results)
- Genes are dicts with indicator params, buy/sell logic, risk management
- Generator produces valid FreqTrade `IStrategy` subclass code

**Evolution loop** (`evolution.py`):
- Checkpoint every 5 generations
- SIS hooks at: initialization (seeds), post-evaluation (immigrants, indicator bias), convergence check, weight adaptation
- Early stopping after `patience` generations with no improvement

**Testing:** `pytest genetic_algorithm/tests/ -q`
