# Changelog

All notable changes to the Freqtrade Genetic Algorithm fork.

## [0.4.0] - 2026-04-12

### Strategy Intelligence System (SIS)
- **SIS v3** — Full ML-driven feedback loop for evolution guidance
  - Corpus builder: 65-feature extraction from strategy genomes + backtest results
  - Multi-target predictor (LightGBM): predicts profit, Sharpe, drawdown, win rate
  - Archetype classifier (HDBSCAN): discovers strategy families in gene space
  - Temporal analyzer: tracks archetype performance drift over time
  - Pattern miner: identifies indicator combinations with highest lift (CCI 26.5×, STOCH 12.5×)
  - **5 integrator hooks** wired into evolution loop:
    1. Seed filtering — remove predicted-bad initial strategies
    2. Smart immigrants — inject archetype-diverse strategies
    3. Indicator bias — weight proven indicators during mutation
    4. Convergence intervention — detect stagnation, diversify
    5. Adaptive weights — shift fitness emphasis based on population state
  - A/B evaluation framework for measuring SIS impact
  - Full test suite (1400+ tests passing)
- **SIS v3.1** (merged from `fix/sis-v3`)
  - Fixed regression normalization bug in `predictors.py` (single-row prediction always returned 0)
  - Excluded `clf_overfit_risk` from quality score (AUC ≈ 0.50 = noise)
  - Reduced stagnation immigrant multiplier 3x → 2x with cap
  - Added `configure_for_run_length()` for short-run auto-scaling
  - Immigrant population cap (`max_immigrants_fraction`, default 25%)
  - Low-fitness archetype filter (below 70% of global mean)
  - Indicator blending by evidence trust (corpus-rank vs live weights)
  - Online archetype retrain with 70% history + 30% live blending
  - LRU memory cache in `BacktestCache` (max 300 entries, prevents RAM growth)
  - `quality_gate_threshold` raised 0.10 → 0.25

### Infrastructure Redesign (Phase 1–5)
- **Phase 1**: Extracted `CheckpointManager`, `AdaptiveController`, `GenerationStep` from monolithic `GeneticAlgorithm`
- **Phase 2**: Island model unification — `IslandCoordinator` facade + migration utilities
- **Phase 3**: Extracted `RunEngine` orchestrator from `evolution.py` (651 LOC)
- **Phase 4**: Enhanced orchestration — scheduler, monitor, lifecycle management
- **Phase 5**: Architecture documentation + domain-based module READMEs
- Extracted `BacktestCache` to `evaluation/cache.py`
- Domain-based module reorganization: `core/`, `engine/`, `evaluation/`, `orchestration/`, `strategies/`, `intelligence/`
- Backward-compatible shims with private name re-exports for all moved modules
- Config schema with defaults and validation (`config/schema.py`)
- Data registry system (`data/registry.json`)

### CLI & Automation
- Unified CLI (`genetic_algorithm/cli.py`) with subcommands: `run`, `resume`, `validate`, `benchmark`
- `Makefile` with targets: `test`, `smoke`, `run`, `benchmark`, `lint`
- Shell scripts migrated to use `python -m genetic_algorithm.cli`
- Queue daemon (`ga_auto_queue_v2.sh`) with persistent monitoring and wave management
- Rich live dashboards: `ga_current.sh`, `ga_monitor_v2.sh`, `wave_dual_monitor.sh`
- `launch_sis_production.sh` for SIS A/B production test orchestration

### Copilot Agent Customization
- Project-level `copilot-instructions.md` with architecture overview and conventions
- 6 domain-specific instruction files: `ga-core`, `ga-config`, `genome`, `dashboard`, `sis-intelligence`, `orchestration`
- `ga-operator` agent for hands-off server operations
- `ga-analyst` agent for post-experiment analysis and diagnosis
- 5 reusable prompt templates for results analysis, config generation, and wave planning
- `sis-corpus-rebuild` and `launch-wave` skills with step-by-step workflows
- GA config validation hook (`validate_ga_config.py`) + PostToolUse YAML format advisory

### Testing
- 61 new tests for infrastructure modules (`test_cli.py`, `test_config_schema.py`, `test_registry.py`)
- `test_engine_decomposition.py` — CheckpointManager + AdaptiveController tests
- `test_generation_step.py` — GenerationStep extraction tests (549 lines)
- `test_runner.py` — RunEngine orchestrator tests (463 lines)
- `test_island_coordinator.py` — Island model unification tests (334 lines)
- `test_orchestration.py` — Scheduler, monitor, lifecycle tests (409 lines)
- `test_phase12_fixes.py` — 23 tests for Phase 1-2 bug fixes
- SIS smoke tests + full strategy intelligence test suite

### Bug Fixes
- CheckpointManager backward compat for v1 checkpoint format
- `cli.py` wrong kwargs passing + `_load_config` schema defaults application
- Schema defaults for `indicators` and `timeframes` preventing KeyError crashes
- Private name re-exports in 6 backward-compatibility shims fixing import errors
- 11 Phase 1-2 bug fixes: convergence_patience typo, adaptive mutation condition, crowding distance, mutable default, cpu_count None, empty predictions, numpy scalar, missing cleanup

### Documentation
- GA Infrastructure Guide (963 lines) — tutorial + reference
- `genetic_algorithm/ARCHITECTURE.md` — component diagrams
- `genetic_algorithm/core/README.md` — domain module guide
- `GA_CONFIG_CHEATSHEET.md`, `KNOWN_ISSUES.md`, `CONFIG_RANKING.md`

---

## [0.3.0] - 2025-03-19

### Experiment Campaign (Waves 24–32)
- 30+ experiments with systematic parameter exploration
- Verified critical constraints: pop size 10–15, tournament ≥ 3, island pop ≥ 60
- Identified and documented overfitting patterns and anti-patterns
- `EXPLORATION_LOG.md` and `CONFIG_RANKING.md` with ranked experiment results

---

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
