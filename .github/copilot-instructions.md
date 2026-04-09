# Project Guidelines — FreqTrade Genetic Algorithm System

Autonomous evolution of cryptocurrency trading strategies using genetic algorithms, integrated with FreqTrade's backtesting engine. The primary workflow is **automated**: generate configs → launch experiments → evolve populations → evaluate via backtest → feed results into SIS → generate better configs → repeat.

## Architecture

```
run_ga.py (entry)
  ├─ GeneticAlgorithm | IslandModelEvolution   (evolution loop)
  ├─ StrategyGenerator                          (gene → Python strategy code)
  ├─ FitnessEvaluator → DirectBacktester        (FreqTrade Python API, cached)
  ├─ ParallelEvaluator                          (ProcessPoolExecutor)
  ├─ SISIntegrator                              (5 hooks: seed, immigrants, indicator bias, convergence, adaptive)
  │   ├─ CorpusBuilder (65 features) → MultiTargetPredictor (LightGBM)
  │   ├─ ArchetypeClassifier (HDBSCAN) → TemporalAnalyzer
  │   └─ PatternMiner (indicator enrichment)
  ├─ WalkForwardValidator / MonteCarloValidator  (robustness)
  ├─ HallOfFame                                 (persistent JSON archive)
  └─ Dashboard                                  (FastAPI + WebSocket)
```

**Key directories:**
- `genetic_algorithm/` — Core GA engine, intelligence (SIS), dashboard, tests
- `genetic_algorithm/config/` — YAML experiment configs and queue/
- `genetic_algorithm/data/` — Results, hall of fame, checkpoints, corpus
- `genetic_algorithm/intelligence/` — SIS: corpus, predictors, archetypes, temporal, patterns, integrator

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

## Critical Conventions (Verified from Experiments)

| Rule | Why |
|------|-----|
| Population size 10-15 for standard GA | >15 causes 59-65% overfitting (E19, E30) |
| Island population ≥ 60 per island | <6 per island → 62-100% overfitting (E24) |
| Never combine island model + walk-forward | Data partitioning conflict causes failures |
| `tournament_size` ≥ 3 | =1 is random search, >6 premature convergence |
| Set `auto_download: false` only with verified data | Silent 0-trade runs if data missing |
| Disable fitness sharing with NSGA-II | Distorts Pareto front |
| Walk-forward applied post-hoc on top-5 | In-loop would cause N×W worker explosion |
| `elite_size` ≈ 10% of population | Balances exploitation vs exploration |

See [GA_CONFIG_CHEATSHEET.md](../GA_CONFIG_CHEATSHEET.md) for all parameter ranges.
See [KNOWN_ISSUES.md](../KNOWN_ISSUES.md) for active issues and anti-patterns.

## Config YAML Key Sections

```yaml
genetic_algorithm:
  population_size: 12          # 10-15 for standard, 60+ per island
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
- GA core in `genetic_algorithm/`, FreqTrade core in `freqtrade/`
- Strategy genes defined in `strategy_gene.py`, converted to code by `generator.py`
- All backtest evaluation goes through `FitnessEvaluator` → `DirectBacktester` (never shell out to freqtrade CLI)
- Configs are YAML (not JSON) for GA experiments
- SIS integration points live in `evolution.py` (search for `sis_integrator`)
- Checkpoint every 5 generations; resume with `--checkpoint`

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
