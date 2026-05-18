# Architecture Guide

## Module Structure

```
genetic_algorithm/
├── __main__.py              # Entry point: python -m genetic_algorithm
├── cli.py                   # CLI commands (run, monitor, queue, data, serve)
│
├── engine/                  # Evolution engine (decomposed)
│   ├── runner.py            # RunEngine: single-run lifecycle
│   ├── generation.py        # GenerationStep: one generation cycle
│   ├── checkpoint.py        # CheckpointManager: save/restore state
│   ├── adaptive.py          # AdaptiveController: rates + convergence
│   ├── migration.py         # Shared migration utilities
│   ├── islands.py           # IslandCoordinator facade
│   ├── population.py        # Population + diversity tracking
│   ├── nsga2.py             # NSGA-II multi-objective
│   ├── pareto_archive.py    # Pareto front archive
│   ├── hall_of_fame.py      # Best strategies archive
│   ├── experiment_tracker.py # Per-generation snapshots for web
│   ├── feature_importance.py # Indicator usage analytics
│   ├── warm_start.py        # Resume from prior results
│   └── operators/
│       ├── crossover.py     # Crossover operators
│       ├── mutation.py      # Mutation operators
│       ├── selection.py     # Selection strategies
│       ├── culling.py       # Population culling
│       └── adaptive.py      # Adaptive operator selection
│
├── genome/                  # Strategy representation
│   ├── gene.py              # IndicatorGene, ConditionGene, StrategyGene
│   ├── individual.py        # Individual (fitness + metadata wrapper)
│   ├── codegen.py           # Gene → FreqTrade Python code
│   ├── indicators.py        # Indicator catalog + valid operators
│   ├── seeds.py             # Seed strategy library
│   └── ensemble.py          # Ensemble strategy support
│
├── evaluation/              # Fitness scoring
│   ├── direct_backtester.py # DirectBacktester (FreqTrade in-process)
│   ├── fitness.py           # FitnessEvaluator (weighted metrics)
│   ├── parallel.py          # ParallelEvaluator (process pool)
│   ├── cache.py             # BacktestCache (memory + disk LRU)
│   ├── surrogate.py         # SurrogateModel (ML pre-filter)
│   └── validation/
│       ├── cpcv.py          # Combinatorial purged cross-validation
│       ├── deflated_sharpe.py
│       ├── monte_carlo.py
│       └── param_sensitivity.py
│
├── intelligence/            # SIS (Strategy Intelligence System)
│   ├── sis_engine.py
│   ├── corpus_builder.py
│   ├── predictors.py
│   ├── archetypes.py
│   └── ...
│
├── orchestration/           # Experiment management
│   ├── registry.py          # ExperimentRegistry (JSON, single source of truth)
│   ├── scheduler.py         # RunScheduler (queue + concurrency)
│   ├── monitor.py           # ExperimentMonitor (terminal dashboard)
│   └── lifecycle.py         # DataLifecycle (retention + cleanup)
│
├── advanced/                # Optional features
│   ├── coevolution.py
│   ├── ensemble_evolution.py
│   ├── incremental.py
│   ├── map_elites.py
│   ├── parsimony.py
│   └── lifecycle_manager.py
│
├── market/                  # Market data & regime
│   ├── regime_aware.py
│   └── shared_memory.py
│
├── config/
│   ├── schema.py            # Config validation + defaults
│   └── presets/             # Ready-to-use configs
│       ├── quick_test.yaml
│       ├── standard.yaml
│       ├── island.yaml
│       ├── production.yaml
│       └── nsga2.yaml
│
├── core/                    # LEGACY — backward-compat shims
│   ├── evolution.py         # Main GeneticAlgorithm (delegates to engine/)
│   ├── island_model.py      # Regime-based island model
│   ├── generic_island_model.py
│   └── [20+ shim files]    # Re-exports to new locations
│
├── web/                     # Web dashboard
│   ├── server.py
│   └── ...
│
└── llm/                     # LLM integration
    ├── designer.py
    └── ...
```

## Key Design Decisions

### D1: Decomposed Engine
The original `evolution.py` (~3,300 lines) was decomposed into:
- **RunEngine** (`engine/runner.py`): Owns the evolution loop lifecycle — signal handling, timing, web events, checkpoint scheduling, resource checks, teardown.
- **GenerationStep** (`engine/generation.py`): One generation cycle — evaluate, select, reproduce, inject immigrants.
- **CheckpointManager** (`engine/checkpoint.py`): Save/restore evolution state.
- **AdaptiveController** (`engine/adaptive.py`): Mutation/crossover rate adaptation, convergence detection.

The `GeneticAlgorithm` class in `core/evolution.py` still exists but now delegates to these components.

### D2: Island Model Unification
Both island implementations share migration utilities via `engine/migration.py`:
- `get_top_individuals()`, `inject_migrants()` — shared helpers
- 4 migration topologies: ring, fully_connected, tournament, hierarchical
- `IslandCoordinator` facade in `engine/islands.py` provides a unified entry point

### D3: Python CLI
Single entry point replaces 20+ shell scripts:
```bash
python -m genetic_algorithm run config.yaml
python -m genetic_algorithm monitor
python -m genetic_algorithm queue start --max-concurrent 5
python -m genetic_algorithm data report
python -m genetic_algorithm data cleanup --dry-run
```

### D4: Experiment Registry
`orchestration/registry.py` — single JSON file as source of truth for all experiment state. Updated atomically with file locking. All components (scheduler, monitor, runner) read/write through the registry instead of scanning the filesystem.

### D5: Orchestration
- **Scheduler**: Queue-based daemon with concurrent run management, memory checks, process recovery
- **Monitor**: Log-parsing dashboard with generation/fitness/profit/ETA extraction
- **Lifecycle**: Policy-driven data archival, cleanup, and disk usage reporting

## Import Policy

- **New code**: Import from the modular location (e.g. `from genetic_algorithm.engine.population import Population`)
- **`engine/__init__.py` MUST remain empty**: No eager imports — `core/` shims import from `engine/`, creating potential circular imports if `__init__.py` eagerly imports submodules
- **`core/` shims**: Will be removed in a future release after all internal imports are updated
