# Project Guidelines — FreqTrade Genetic Algorithm System

Autonomous evolution of cryptocurrency trading strategies using genetic algorithms, integrated with FreqTrade's backtesting engine. The primary workflow is **automated**: generate configs → launch experiments → evolve populations → evaluate via backtest → feed results into SIS → generate better configs → repeat.

## Architecture

```
run_ga.py / cli.py (entry points)
  ├─ core/        → backward-compat shims (real code in engine/ and advanced/)
  ├─ engine/      → runner, generation, checkpoint, adaptive, hall_of_fame, islands, migration, population
  ├─ advanced/    → coevolution, ensemble_evolution, incremental, MAP-Elites, lifecycle_manager
  ├─ genome/      → gene definitions, codegen (gene→strategy code), indicators, seeds
  ├─ evaluation/  → fitness, direct_backtester, parallel, cache, surrogate, regime_aware
  │   └─ validation/ → monte_carlo, cpcv, deflated_sharpe, param_sensitivity
  ├─ intelligence/ → SIS: corpus, predictors, archetypes, temporal, patterns, integrator, ab_framework
  ├─ orchestration/ → ExperimentRegistry, DataLifecycle, Scheduler, Monitor
  ├─ market/      → regime detection, shared memory
  ├─ ml/          → regime detector/trainer (LightGBM)
  ├─ llm/         → LLM strategy designer, injector, diagnostics
  ├─ monitor/     → terminal_monitor, log_capture, key_listener
  ├─ web/         → FastAPI + WebSocket + React dashboard
  │   ├─ routers/ → runs, generations, strategies, config, data, sis, backtest, dry_run, ws
  │   └─ services/ → data_service
  └─ config/      → YAML experiments (queue/, done/{wave}/, benchmark_v2/, presets/)
```

**Key directories:**
- `genetic_algorithm/engine/` — Core evolution loop, checkpoint, adaptive mutation, HoF
- `genetic_algorithm/evaluation/` — Backtesting, fitness, parallel workers, validation
- `genetic_algorithm/genome/` — Strategy gene definitions and code generation
- `genetic_algorithm/intelligence/` — SIS: corpus, predictors, archetypes, A/B framework
- `genetic_algorithm/orchestration/` — Experiment lifecycle (registry, data cleanup, scheduling)
- `genetic_algorithm/web/` — FastAPI backend + React/TypeScript frontend dashboard
- `genetic_algorithm/config/` — YAML experiment configs (`queue/`, `done/`, `benchmark_v2/`)
- `genetic_algorithm/data/` — Results, hall of fame, checkpoints, corpus

See [GA_SYSTEM_SUMMARY.md](../GA_SYSTEM_SUMMARY.md) for full architecture.
See [genetic_algorithm/ARCHITECTURE.md](../genetic_algorithm/ARCHITECTURE.md) for component diagrams.

## Build and Test

```bash
# Install
pip install -e '.[dev]'                          # Editable install with dev deps

# Test
make test                                        # Full suite (1400+ tests)
make smoke                                       # Quick 2-min validation
pytest genetic_algorithm/tests/ -q               # GA-only tests
pytest tests/test_strategy_intelligence.py -v    # SIS tests

# Run
python genetic_algorithm/run_ga.py --config genetic_algorithm/config/<name>.yaml

# CLI (preferred interface)
python -m genetic_algorithm run --config <name>.yaml
python -m genetic_algorithm monitor              # Live experiment monitor
python -m genetic_algorithm queue start           # Start queue daemon
python -m genetic_algorithm experiment list       # List registered experiments
python -m genetic_algorithm data cleanup          # Lifecycle management

# Automated queue (daemon, persistent monitoring)
./ga_auto_queue_v2.sh --wave <wave> --max <N> --persistent

# Monitor
./ga_current.sh                                  # Rich live dashboard
./ga_monitor_v2.sh --live                       # Alternative live monitor
```

## Automated Evolution Workflow

The core loop that runs autonomously on the server:

1. **Config generation** — YAML files in `genetic_algorithm/config/queue/` define experiments
2. **Queue daemon** — `ga_auto_queue_v2.sh --persistent` picks configs, launches `run_ga.py`
3. **Evolution** — Population initializes (random + HoF injection + SIS seeds), then loops:
   evaluate → select (tournament) → crossover → mutate → replace (elitism) → checkpoint
4. **SIS feedback** — When enabled (`sis.enabled: true`), 5 hooks improve evolution:
   - Seed filtering (remove predicted-bad initial strategies)
   - Smart immigrants (inject archetype-diverse strategies)
   - Indicator bias (weight proven indicators: CCI 26.5x, STOCH 12.5x lift)
   - Convergence intervention (detect stagnation, diversify)
   - Adaptive weights (shift fitness emphasis based on population state)
5. **Validation** — Top-5 strategies get walk-forward + Monte Carlo post-hoc validation
6. **Archive** — Results written to `data/`, HoF updated, config moved to `done/{wave}/`
7. **Learn** — SIS corpus rebuilt from accumulated HoF strategies; next generation benefits

## Configuration and evidence conventions

- The executable contract is `config/schema.py` plus the versioned mechanical invariants in
  `config/invariants.py`.
- Population, island size, pair count, generations, mutation, elitism and tournament pressure are
  experimental factors; no fixed value is a universal overfitting boundary.
- Compare tuning changes with paired seeds, the same panel and budget, then decide from the
  hash-verified V2 analysis.
- `tournament_size: 1` is mechanically valid and means uniform-random tournament selection.
- Classic `island_model` plus walk-forward is rejected because that engine disables the requested
  walk-forward setting. Generic islands have a separate explicit contract.
- Automatic waves use `safe_v2`; optional features remain experimental until promotion evidence
  supports enabling them.
- Data availability and exact period coverage are proven by the data manifest, never assumed from
  `auto_download` or a mutable path.

## Config YAML Key Sections

```yaml
genetic_algorithm:
  population_size: 30          # Experimental factor; pair against a control
  generations: 25
  mutation_rate: 0.20          # Adapts up on convergence
  crossover_rate: 0.75
  tournament_size: 3
  elite_size: 2

backtesting:
  timerange: "20240301-20260228"
  pairs: ["BTC/USDT", "ETH/USDT"]
  stake_amount: 0.15
  max_open_trades: 3
  enable_cache: true           # 2-5x speedup

parallel_evaluation:
  enabled: true                # 3.5-4.7x on 4-8 workers

walk_forward:
  enabled: true                # 5x slower but prevents overfitting

sis:
  enabled: true                # SIS feedback hooks
```

## Code Conventions

- **Python 3.11+**, type hints on public APIs
- GA system in `genetic_algorithm/`, FreqTrade core in `freqtrade/`
- `core/` files are backward-compat shims — real code lives in `engine/`, `advanced/`, `evaluation/`, `genome/`
- Strategy genes in `genome/gene.py`, code generation in `genome/codegen.py`
- All backtest evaluation goes through `FitnessEvaluator` → `DirectBacktester` (never shell out to freqtrade CLI)
- Configs are YAML (not JSON) for GA experiments
- SIS integration points live in `engine/runner.py` (search for `sis_integrator`)
- Checkpoint every 5 generations; resume with `--resume`
- Experiment lifecycle managed by `ExperimentRegistry` (register → start → complete/fail/cancel)
- Prefer `python -m genetic_algorithm` CLI over raw shell scripts
- Terminal monitor (`monitor/terminal_monitor.py`) provides rich live feedback during evolution

## Documentation Index

| Topic | File |
|-------|------|
| Full architecture | [GA_SYSTEM_SUMMARY.md](../GA_SYSTEM_SUMMARY.md) |
| All config parameters | [GA_CONFIG_CHEATSHEET.md](../GA_CONFIG_CHEATSHEET.md) |
| Infrastructure roadmap | [INFRASTRUCTURE_REDESIGN.md](../INFRASTRUCTURE_REDESIGN.md) |
| Known issues & fixes | [KNOWN_ISSUES.md](../KNOWN_ISSUES.md), [GA_FIXES_AND_IMPROVEMENTS.md](../GA_FIXES_AND_IMPROVEMENTS.md) |
| Config reference | [genetic_algorithm/docs/CONFIG_REFERENCE.md](../genetic_algorithm/docs/CONFIG_REFERENCE.md) |
| Parallel evaluation | [genetic_algorithm/docs/PARALLEL_EVALUATION_GUIDE.md](../genetic_algorithm/docs/PARALLEL_EVALUATION_GUIDE.md) |
| Robustness features | [genetic_algorithm/docs/TIER3_ROBUSTNESS_FEATURES.md](../genetic_algorithm/docs/TIER3_ROBUSTNESS_FEATURES.md) |
| SIS details | [genetic_algorithm/intelligence/README.md](../genetic_algorithm/intelligence/README.md) |
| Experiment results | [EXPLORATION_LOG.md](../EXPLORATION_LOG.md), [CONFIG_RANKING.md](../CONFIG_RANKING.md) |
| Contributing | [CONTRIBUTING.md](../CONTRIBUTING.md) |
