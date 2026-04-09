# Changelog

All notable changes to the Freqtrade Genetic Algorithm fork.

## [SIS v3.1] - 2026-04-09

### Fixed
- **Regression normalization bug** (`predictors.py`): single-row prediction always
  returned 0 due to per-batch min-max normalization. Fixed by storing absolute p5/p95
  percentile anchors at training time and using them at predict time.
- **`clf_overfit_risk` excluded from quality score**: classifier was near-random
  (AUC ≈ 0.50) and contributed noise rather than signal. Now permanently excluded.
- **Stagnation immigrant multiplier reduced** 3x → 2x: original 3x was not capped,
  causing cold-restart-style population replacement under stagnation.

### Added
- **`configure_for_run_length(max_generations)`** (`sis_integrator.py`): auto-scales
  the evidence trust ramp and online-retrain interval for short runs (≤20 gens).
  Trust initial raised to 0.4 and retrain fires ~3× per run so weights have actual
  effect within the experiment window.
- **Immigrant population cap** (`max_immigrants_fraction`, default 25%): prevents
  unbounded immigrant injection under stagnation; wire-through from evolution.py.
- **Low-fitness archetype filter**: archetypes below 70% of global mean fitness are
  excluded from immigrant generation; survivors are sorted quality-first.
- **Indicator blending** (`_generate_immigrant`): corpus-rank vs live indicator weights
  are blended by current `evidence_trust`, so SIS gradually shifts toward
  what actually worked in the live run.
- **`_refresh_archetype_stats`**: online retrain now re-classifies live data and
  updates `archetype_stats` fitness means (70% history + 30% live). Immigrants
  in later generations use fresh priors instead of frozen gen-0 corpus stats.
- **LRU memory cache** (`direct_backtester.py`): `BacktestCache` now uses an
  `OrderedDict` with configurable `max_memory_entries` (default 300). Prevents
  unbounded RAM growth in long runs while keeping hot entries in memory.
- **`ga_monitor_v2.sh` `--live / -l` mode**: shows only running experiments plus
  a filtered log tail — useful when monitoring many concurrent waves.
- **Hall-of-Fame block in monitor**: when a `--wave` filter is given, completed
  experiments display `fitness | T.profit | V.profit | T/V gap | pair_gen ratio`
  parsed directly from `hall_of_fame.json` via an embedded Python snippet.
- **`launch_sis_production.sh`**: orchestration script for the 4-experiment SIS
  production A/B test (2 control + 2 SIS, performance pair + robustness pair).

### Changed
- `quality_gate_threshold` default raised **0.10 → 0.25**: requires stronger evidence
  before an archetype is considered reliable for immigrant generation.
- `BacktestCache.__init__` signature: added `max_memory_entries` keyword arg.

## [Unreleased] - 2025-02-22

### Added

#### NSGA-II Multi-Objective Optimization
- Complete implementation of NSGA-II (Non-dominated Sorting Genetic Algorithm II)
- Multi-objective fitness evaluation with configurable objectives:
  - Profit maximization
  - Sharpe ratio optimization
  - Win rate improvement
  - Drawdown minimization
- Pareto front tracking across generations
- Non-dominated sorting with crowding distance for diversity preservation
- Full integration with existing evolution framework

#### Parallel Strategy Evaluation
- Multi-process parallel backtesting using `ProcessPoolExecutor`
- Configurable worker count (auto-detects optimal based on CPU cores)
- **Benchmark Results:**
  - 6 strategies, 4 workers: 22.73s → 7.41s (**3.07x speedup**)
  - 12 strategies, 6 workers: 33.02s → 9.11s (**3.62x speedup**)
- Automatic fallback to sequential evaluation on errors
- Worker process isolation for stability
- Graceful shutdown and resource cleanup

#### Configuration Options
New `parallel_evaluation` section in `ga_config.yaml`:
```yaml
parallel_evaluation:
  enabled: true
  num_workers: null  # Auto-detect optimal workers
  worker_log_level: "WARNING"
```

### Documentation
- `PARALLEL_EVALUATION_GUIDE.md` - Complete parallel evaluation documentation
- Updated `TODO_ga_improvements.md` with completed features
- Benchmark script at `genetic_algorithm/benchmark_parallel.py`

### Technical Details
- New module: `genetic_algorithm/evaluation/parallel.py`
- Dependencies: Uses standard library `concurrent.futures` (no new deps)
- Test suite: `tests/test_parallel_evaluation.py`

---

## [Previous] - 2025-02-19

### Features (Pre-existing)
- Walk-Forward Analysis integration
- Dynamic max_open_trades optimization
- Genetic algorithm-based strategy parameter optimization
- Multi-criteria fitness evaluation
- Configuration-driven evolution parameters
