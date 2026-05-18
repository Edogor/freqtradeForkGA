# GA Infrastructure Guide
## Strategy Exploration & Production with the Redesigned Architecture

> **Audience:** Developers and quant traders using this fork.  
> **Branch:** `feature/ga-infrastructure-redesign`  
> **Entry point:** `python -m genetic_algorithm <command>`

---

## Table of Contents

1. [What Changed — The Redesign at a Glance](#1-what-changed)
2. [New Capabilities](#2-new-capabilities)
3. [Quick Start — Your First Run in 2 Minutes](#3-quick-start)
4. [Understanding Configs](#4-understanding-configs)
5. [Strategy Exploration Workflows](#5-strategy-exploration-workflows)
6. [Going to Production](#6-going-to-production)
7. [Batch Experiments & Queuing](#7-batch-experiments--queuing)
8. [Monitoring](#8-monitoring)
9. [Resuming and Checkpoints](#9-resuming-and-checkpoints)
10. [Advanced: Island Model](#10-advanced-island-model)
11. [Advanced: NSGA-II Multi-Objective](#11-advanced-nsga-ii-multi-objective)
12. [Advanced: Strategy Intelligence System (SIS)](#12-advanced-strategy-intelligence-system-sis)
13. [Advanced: Walk-Forward + Anti-Overfit Stack](#13-advanced-walk-forward--anti-overfit-stack)
14. [Makefile Shortcuts](#14-makefile-shortcuts)
15. [Reference — Full CLI Command Tree](#15-reference--full-cli-command-tree)
16. [Config Key Reference](#16-config-key-reference)

---

## 1. What Changed

The original codebase had a single 3,300-line `evolution.py` god-class and 37+ shell scripts that launched it in various ways.  
The redesign replaced all of that with a **layered, composable architecture**.

### Before vs. After

| Before | After |
|--------|-------|
| `python genetic_algorithm/run_ga.py --config X` | `python -m genetic_algorithm run X` |
| 37 shell scripts with duplicated logic | 1 unified CLI, 1 Makefile |
| No experiment tracking — scan filesystem | Atomic JSON registry (`orchestration/registry.py`) |
| Evolution loop mixed with lifecycle, signals, reporting | Separated into `RunEngine`, `GenerationStep`, `CheckpointManager`, `AdaptiveController` |
| Island model: 2 separate implementations, no shared code | `IslandCoordinator` facade + shared `migration.py` utilities |
| Configs: raw dicts, no defaults, no validation | `schema.py` with full `DEFAULTS`, preset system, `config validate` command |
| Checkpoints: best-effort JSON | Atomic write + SHA-256 checksum, v1/v2 backward compat |

### Module Map (what was extracted from `evolution.py`)

```
engine/
├── runner.py          ← RunEngine: loop structure, signals, timing, teardown
├── generation.py      ← GenerationStep: one generation cycle
├── checkpoint.py      ← CheckpointManager: save/restore state (atomic + checksummed)
├── adaptive.py        ← AdaptiveController: mutation rate adaptation + convergence
├── islands.py         ← IslandCoordinator: unified island-model facade
└── migration.py       ← Shared migration utilities (4 topologies)

orchestration/
├── registry.py        ← ExperimentRegistry: JSON source of truth (file-locked)
├── scheduler.py       ← RunScheduler: queue daemon + concurrency management
├── monitor.py         ← ExperimentMonitor: log-parsing terminal dashboard
└── lifecycle.py       ← DataLifecycle: retention policies + disk reporting
```

Backward compatibility shims in `core/` ensure all existing code continues to import from its old paths.

---

## 2. New Capabilities

### 2.1 Unified CLI

Any workflow that previously required a shell script is now a single command:

```bash
python -m genetic_algorithm run production           # by preset name
python -m genetic_algorithm run my_config.yaml       # by file path
python -m genetic_algorithm config validate run3.yaml
python -m genetic_algorithm queue add --dir config/benchmark/ --tag benchmarks
python -m genetic_algorithm experiment list --status completed
python -m genetic_algorithm serve                     # web dashboard
```

### 2.2 Preset System

Five ready-to-use presets. Override any key in your own YAML:

| Preset | Purpose | Runtime |
|--------|---------|---------|
| `quick_test` | Smoke test — 2 gen × 5 pop | ~2 min |
| `standard` | Single-pop discovery — 30 pop × 12 gen | ~30–60 min |
| `island` | 6-island ring — good exploration vs. convergence tradeoff | ~60–120 min |
| `nsga2` | Pareto-front optimization (profit + drawdown) | ~60–90 min |
| `production` | Full stack — 8 islands, walk-forward, holdout, Monte Carlo | ~4–8 h |

### 2.3 Config Validation Before Running

No more discovering a config error after 20 minutes of running:

```bash
python -m genetic_algorithm config validate my_config.yaml
python -m genetic_algorithm config validate my_config.yaml --strict   # warnings = errors
python -m genetic_algorithm config show my_config.yaml               # full resolved config
```

### 2.4 Experiment Queue + Scheduler

Queue multiple experiments and run them with a concurrency limit, automatic memory checks, and process recovery:

```bash
# Add to queue
python -m genetic_algorithm queue add config_a.yaml config_b.yaml --tag wave1
python -m genetic_algorithm queue add --dir config/benchmark/ --tag benchmarks

# Start scheduler (blocks; manages concurrency + restarts)
python -m genetic_algorithm queue start --max-concurrent 3

# Or with make
make benchmark    # queues all 8 benchmark configs
```

### 2.5 Experiment Registry

Every run is automatically tracked in `genetic_algorithm/data/registry.json`:

```bash
python -m genetic_algorithm experiment list
python -m genetic_algorithm experiment list --status completed
python -m genetic_algorithm experiment show E200_island_run1
python -m genetic_algorithm experiment compare E200 E201 E202
```

### 2.6 Reliable Checkpoints

Checkpoints are now written atomically (write to temp → rename) with SHA-256 checksums. A corrupted checkpoint is detected and rejected rather than used silently.

```bash
# Resume any run from its checkpoint
python -m genetic_algorithm run my_config.yaml --resume genetic_algorithm/runs/E200_island_run1/
```

### 2.7 Composable Components (for developers)

Each engine component is independently testable and importable:

```python
from genetic_algorithm.engine.checkpoint import CheckpointManager
from genetic_algorithm.engine.adaptive import AdaptiveController
from genetic_algorithm.engine.islands import IslandCoordinator
from genetic_algorithm.orchestration.registry import ExperimentRegistry
from genetic_algorithm.evaluation.cache import BacktestCache
```

---

## 3. Quick Start

### Install and verify

```bash
# Activate environment
source .venv/bin/activate

# Verify entry point works
python -m genetic_algorithm --help

# Smoke test: 2 generations × 5 individuals (~2 minutes)
python -m genetic_algorithm run quick_test --no-monitor --yes
# or: make smoke
```

A successful smoke test prints a summary like:

```
[Generation 1/2] best_fitness=0.31  avg=0.18  diversity=0.62
[Generation 2/2] best_fitness=0.38  avg=0.22  diversity=0.58
Evolution complete. Best fitness: 0.38
Strategies saved to: genetic_algorithm/runs/<experiment_id>/strategies/
```

### Run your first real exploration

```bash
python -m genetic_algorithm run standard
```

This runs `config/presets/standard.yaml`: 30-individual population, 12 generations, 2 pairs (BTC/USDT + ETH/USDT), holdout validation and deflated-Sharpe enabled.

---

## 4. Understanding Configs

### 4.1 Config Loading Pipeline

```
Your YAML file
    └── resolve_preset()   ← expands "preset: island" to full preset YAML
    └── deep_merge(DEFAULTS, preset)
    └── deep_merge(result, your overrides)
    └── validate_config()   ← checks errors + warnings
    └── ready config dict
```

You only need to write what **differs from the defaults**. Example — a minimal custom config:

```yaml
# my_scalp_5m.yaml
backtesting:
  pairs: [BTC/USDT, ETH/USDT, SOL/USDT]
  timerange: "20250101-20260301"

genetic_algorithm:
  population_size: 40
  generations: 20

strategy_constraints:
  timeframes: ["5m"]
  stoploss_range: [-0.05, -0.02]
  roi_range: [0.005, 0.03]
```

Everything else inherits from `DEFAULTS` in `config/schema.py`.

### 4.2 Preset Inheritance

Add `preset` to your YAML to start from a preset and override selectively:

```yaml
# my_production_btc.yaml
preset: production        # start from the production preset

backtesting:
  pairs: [BTC/USDT]       # override just the pair list
  timerange: "20240101-20260401"

genetic_algorithm:
  generations: 30         # run longer than the preset default
```

### 4.3 Discover Available Configs

```bash
python -m genetic_algorithm config list             # 5 presets
python -m genetic_algorithm config list --benchmarks  # + 8 benchmark configs
python -m genetic_algorithm config list --all         # + 54 legacy ga_config_*.yaml files
```

### 4.4 Inspect Resolved Config

Before running, see exactly what a config will use (all defaults merged in):

```bash
python -m genetic_algorithm config show production
python -m genetic_algorithm config show my_scalp_5m.yaml
```

---

## 5. Strategy Exploration Workflows

### 5.1 Exploration Loop (recommended)

```
1. Quick test → confirm data + env
2. Standard run → 30-pop baseline, see what fitness is achievable
3. Vary one dimension (pairs, timeframes, fitness weights)
4. Island run → better exploration of the search space
5. Harvest best strategies → deploy or refine
```

### 5.2 Restricting the Indicator Search Space

To focus evolution on specific indicators:

```yaml
indicators:
  available: [RSI, CCI, DONCHIAN, ATR, ROC]
  min_per_strategy: 2
  max_per_strategy: 4

strategy_constraints:
  timeframes: ["1h"]
  stoploss_range: [-0.12, -0.04]
```

The indicator enrichment evidence from SIS analysis suggests prioritizing: `CCI` (26.5× lift), `STOCH` (12.5×), `ATR` (6.5×), `ROC` (6.5×), `DONCHIAN` (synergy with ROC). These outperform `MACD` and `ICHIMOKU` in top strategies historically.

### 5.3 Fitness Weight Tuning

Default weights favour profit (28%) + drawdown (15%) + Sharpe (18%). Adjust based on your goal:

```yaml
# Risk-averse: maximize Sharpe/Sortino, constrain drawdown hard
fitness_weights:
  profit: 0.15
  sharpe_ratio: 0.30
  sortino_ratio: 0.20
  drawdown: 0.25
  win_rate: 0.05
  trade_frequency: 0.03
  monthly_stability: 0.02

fitness_penalties:
  max_drawdown: 0.15       # hard kill for strategies exceeding 15% drawdown
  min_win_rate: 0.45
  min_trades: 20
```

```yaml
# Aggressive: maximize raw profit
fitness_weights:
  profit: 0.45
  sharpe_ratio: 0.10
  sortino_ratio: 0.08
  drawdown: 0.12
  win_rate: 0.08
  profit_factor: 0.12
  trade_frequency: 0.05
```

Weights should sum to ~1.0. Run `config validate` to catch drift:

```bash
python -m genetic_algorithm config validate my_config.yaml
# WARNING: fitness_weights sum to 1.052 (expected ~1.0)
```

### 5.4 Multi-Pair Exploration

Evolve for cross-pair generalization (strategies that work on multiple pairs score a `cross_pair` bonus):

```yaml
backtesting:
  pairs: [BTC/USDT, ETH/USDT, SOL/USDT, XRP/USDT, BNB/USDT]

fitness_weights:
  cross_pair: 0.08    # reward cross-pair consistency
  monthly_stability: 0.06
```

### 5.5 Timeframe Sweeps

To find the best timeframe for a strategy family, run short experiments with each:

```bash
# Create three configs varying only the timeframe
python -m genetic_algorithm queue add \
    config/exploration/sweep_5m.yaml \
    config/exploration/sweep_15m.yaml \
    config/exploration/sweep_1h.yaml \
    --tag tf_sweep

python -m genetic_algorithm queue start
```

Then compare:

```bash
python -m genetic_algorithm experiment list --tag tf_sweep
python -m genetic_algorithm experiment compare E200 E201 E202
```

---

## 6. Going to Production

### 6.1 Production Checklist

Before committing to a multi-hour run:

```bash
# 1. Validate your config
python -m genetic_algorithm config validate my_production.yaml

# 2. Smoke test with the same config (overriding generations)
python -m genetic_algorithm run my_production.yaml --yes --no-monitor
# (add genetic_algorithm.generations: 2 to a copy for a quick sanity check)

# 3. Check disk space
python -m genetic_algorithm data report
```

### 6.2 Using the `production` Preset

The `production` preset is the most complete anti-overfit configuration:

```bash
python -m genetic_algorithm run production --name "btc_eth_apr26" --tag production
```

What it enables:

| Feature | Setting |
|---------|---------|
| Island model | 8 islands, ring topology, migrate every 3 gen |
| Walk-forward validation | 120d train / 30d val / 30d step |
| Holdout monitoring | 15% holdout, early-stop if degradation > 40% |
| Monte Carlo | 50 iterations per final strategy |
| Deflated Sharpe | Enabled — penalizes strategies that likely overfit |

### 6.3 Custom Production Config

Build on the production preset, extending the timerange and adding more pairs:

```yaml
# runs/production_apr2026.yaml
preset: production

backtesting:
  pairs: [BTC/USDT, ETH/USDT, SOL/USDT, XRP/USDT]
  timerange: "20230101-20260401"   # 3+ years of data

genetic_algorithm:
  population_size: 15              # per island
  generations: 25
  convergence_patience: 12

generic_island_model:
  num_islands: 8
  population_per_island: 15
  migration:
    topology: ring
    interval: 4
    rate: 0.15

holdout_validation:
  holdout_pct: 0.20                # 20% holdout (stricter)

monte_carlo:
  enabled: true
  iterations: 100                  # more iterations for final strategies
```

### 6.4 Reading Output Strategies

After a run, strategies are saved in:

```
genetic_algorithm/runs/<experiment_id>/
├── strategies/
│   ├── best_strategy.py          ← best overall individual
│   ├── hall_of_fame/             ← top N strategies across all generations
│   └── pareto_front/             ← Pareto-optimal strategies (if NSGA-II)
├── checkpoints/
│   └── checkpoint_gen10_*.json
├── evolution_stats.json
└── run.log
```

The `.py` files are directly usable FreqTrade strategy files. Test them with FreqTrade's own backtester:

```bash
freqtrade backtesting --strategy BestStrategy \
    --strategy-path genetic_algorithm/runs/<id>/strategies/ \
    --config user_data/config.json \
    --timerange 20260101-20260401
```

---

## 7. Batch Experiments & Queuing

### 7.1 Queue Multiple Configs

```bash
# Queue individual files
python -m genetic_algorithm queue add config_a.yaml config_b.yaml --tag wave1 --priority 10

# Queue entire directory
python -m genetic_algorithm queue add --dir genetic_algorithm/config/benchmark/ --tag benchmarks

# Queue the 8 standard benchmark configs
make benchmark
```

### 7.2 Start the Queue Scheduler

```bash
# Start with a concurrency limit (blocks until queue drains)
python -m genetic_algorithm queue start --max-concurrent 3

# Keep running and watch for new queued items
python -m genetic_algorithm queue start --max-concurrent 3 --persistent
```

The scheduler:
- Respects `--max-concurrent` slot limit
- Checks free memory (>1.5 GB required before each launch)
- Auto-recovers crashed child processes (marks them `failed` in registry)
- Writes a PID file at `genetic_algorithm/logs/scheduler.pid`

### 7.3 Queue Status

```bash
python -m genetic_algorithm queue status
python -m genetic_algorithm experiment list --status queued
python -m genetic_algorithm experiment list --status running
```

### 7.4 Priority

Lower number = launched first. Default = 50.

```bash
python -m genetic_algorithm queue add critical_run.yaml --priority 1
python -m genetic_algorithm queue add secondary_run.yaml --priority 90
```

---

## 8. Monitoring

### 8.1 Terminal Monitor

```bash
# Live dashboard while a run is ongoing
python -m genetic_algorithm monitor

# Show log tails
python -m genetic_algorithm monitor --live

# Filter by tag
python -m genetic_algorithm monitor --tag production

# Print once and exit (for scripting)
python -m genetic_algorithm monitor --once
```

### 8.2 Web Dashboard

```bash
python -m genetic_algorithm serve
# Open http://localhost:8000
```

The dashboard provides:
- Fitness evolution chart (best/avg/worst per generation)
- Population scatter (fitness vs. complexity)
- Hall of Fame browser
- Strategy gene tree inspector
- Live log stream
- Run history with detail pages
- Config editor

### 8.3 Data Report

```bash
python -m genetic_algorithm data report
```

Shows disk usage per experiment, total cache size, and oldest/newest runs.

### 8.4 Cleanup

```bash
# Preview what would be deleted (safe)
python -m genetic_algorithm data cleanup --dry-run

# Delete data older than retention policy
python -m genetic_algorithm data cleanup
```

---

## 9. Resuming and Checkpoints

Checkpoints are saved every 5 generations by default (configurable via `checkpoint.interval`).

### 9.1 Automatic Resume

If a run crashes or is interrupted with `Ctrl+C`, it saves a final checkpoint. Resume with:

```bash
python -m genetic_algorithm run my_config.yaml \
    --resume genetic_algorithm/runs/<experiment_id>/checkpoints/
```

The runner automatically loads the latest checkpoint from that directory, restoring:
- Full population with fitness scores
- Random state (reproducible continuation)
- Generation statistics
- Surrogate model state
- Holdout state
- Adaptive controller rates

### 9.2 Custom Checkpoint Interval

```yaml
genetic_algorithm:
  checkpoint_interval: 3     # save every 3 generations instead of 5
```

### 9.3 Checkpoint Integrity

Checkpoints use SHA-256 checksums. A corrupted or truncated checkpoint is detected:

```
[CHECKPOINT] Integrity check failed for checkpoint_gen10_*.json — skipping
[CHECKPOINT] Loading checkpoint_gen5_*.json instead
```

---

## 10. Advanced: Island Model

The island model runs multiple independent subpopulations that periodically exchange top strategies (migration). This prevents premature convergence and explores a wider search space.

### 10.1 Generic Island Model

```yaml
generic_island_model:
  enabled: true
  num_islands: 6
  population_per_island: 10   # total effective population: 60
  migration:
    topology: ring             # ring | fully_connected | tournament | hierarchical
    interval: 3                # migrate every N generations
    rate: 0.20                 # 20% of island population migrates
```

**Topology comparison:**

| Topology | Description | Best for |
|----------|-------------|---------|
| `ring` | Each island receives from one neighbour | Long runs, stable convergence |
| `fully_connected` | Every island migrates to every other | Faster convergence, less diversity |
| `tournament` | Best-scoring island donates to weaker | Focused improvement |
| `hierarchical` | Tree structure — champions propagate down | Very large island counts (20+) |

### 10.2 Regime-Specialist Islands

Enable regime detection to let each island specialize on a different market regime:

```yaml
island_model:
  enabled: true                # uses data-segmented regime islands

regime_aware:
  enabled: true
  detection_method: adx_di_hysteresis
```

Regime islands segment historically-labelled data (bull/bear/sideways) so that each island evolves strategies that are genuinely optimized for that regime. The best strategy per regime is selected at the end.

---

## 11. Advanced: NSGA-II Multi-Objective

Instead of collapsing all objectives into a single weighted fitness score, NSGA-II maintains a **Pareto front** — the set of strategies where no single strategy is better in all objectives simultaneously.

```yaml
genetic_algorithm:
  mode: nsga2
  selection_method: nsga2
  population_size: 30
  generations: 20

nsga2:
  objectives:
    - name: profit
      direction: maximize
      weight: 1.0
    - name: sharpe_ratio
      direction: maximize
      weight: 1.0
    - name: max_drawdown
      direction: minimize
      weight: 1.0

pareto_archive:
  enabled: true
  max_size: 50
```

After the run, `strategies/pareto_front/` contains all Pareto-optimal strategies. You can manually select from the front based on your current risk tolerance — a strategy with 8% profit + 5% drawdown vs. one with 14% profit + 12% drawdown is a genuine tradeoff, not a clear winner.

Use the preset directly:

```bash
python -m genetic_algorithm run nsga2
```

---

## 12. Advanced: Strategy Intelligence System (SIS)

SIS is a meta-learning layer that analyses a corpus of 2,900+ strategies to guide the GA search.

### How it works (5-phase pipeline)

1. **Corpus analysis** — extract 65 surrogate features from historical strategies
2. **Performance predictors** — LightGBM models predicting win rate / profit / drawdown from features alone
3. **Archetype clustering** — HDBSCAN finds 8–12 recurring "strategy DNA" patterns
4. **Temporal analysis** — identifies which archetypes go stale, and which pair well
5. **Pattern mining** — indicator enrichment (CCI 26.5× lift in top strategies; MACD depleted)

### Enabling SIS

```yaml
sis:
  enabled: true
```

### What SIS does during evolution

| Hook | Effect |
|------|--------|
| Seed filtering | Seeds are scored by 3 performance models; only top-scoring seeds survive |
| Archetype immigrants | Random immigrants are sampled to match underrepresented archetypes |
| Indicator bias | Mutation operator weights are adjusted based on enrichment scores |
| Convergence-aware aggression | SIS increases mutation diversity when the GA plateaus |
| Online adaptive weights | Bayesian blending of prior enrichment + live evidence from current run |

### Key enrichment findings (static priors)

```
CCI          +26.5× lift in top strategies
STOCH        +12.5× lift
DONCHIAN+ROC +21.5× synergy (strongest pair)
MACD          depleted in top strategies
ICHIMOKU      strongly depleted
cross_below   favoured operator
cross_above   avoided operator
```

You can use these as manual guidance even without SIS enabled — by restricting `indicators.available` to high-enrichment indicators.

---

## 13. Advanced: Walk-Forward + Anti-Overfit Stack

The full anti-overfit stack prevents the GA from finding strategies that look good in-sample but fail on new data.

### 13.1 Walk-Forward Validation

Evaluates each individual on rolling train/validation windows:

```yaml
walk_forward:
  enabled: true
  train_days: 120
  validation_days: 30
  step_days: 30
  aggregation: harmonic_mean    # harmonic_mean is harsher on inconsistent windows
  embargo_days: 5               # gap between train end and val start (prevents leakage)
```

Strategies must perform consistently across all windows to score well, not just one lucky in-sample period.

### 13.2 Holdout Monitoring

Keeps a fixed hold-out set (last 15% of timerange) hidden from evolution. Checks top-N strategies every K generations:

```yaml
holdout_monitoring:
  enabled: true
  interval: 2          # check every 2 generations
  top_n: 5
  early_stop: true
  early_stop_threshold: 0.60   # stop if holdout < 60% of in-sample fitness
  penalty_factor: 0.80         # multiply fitness by 0.8 if holdout is degraded
```

### 13.3 Deflated Sharpe Ratio

Penalizes strategies with a high probability of overfitting based on number of trials:

```yaml
deflated_sharpe:
  enabled: true
```

The penalty is applied as a fitness multiplier. Strategies that are genuinely skilled survive; lucky strategies do not.

### 13.4 Monte Carlo Robustness

Applied to final strategies (not during evolution — too slow):

```yaml
monte_carlo:
  enabled: true
  iterations: 100
```

Shuffles trade order 100 times to estimate the range of possible outcomes. A strategy with robustness_score < 0.5 is discarded.

### 13.5 Combinatorial Purged Cross-Validation (CPCV)

Opt-in, applied to final candidate strategies only:

```yaml
cpcv:
  enabled: true
  n_groups: 6
  n_test_groups: 2
  purge_pct: 0.01
  embargo_pct: 0.01
  pbo_threshold: 0.5       # Probability of Backtest Overfitting threshold
  penalty_weight: 0.20
```

CPCV generates all $\binom{N}{S}$ train/test path combinations and computes the Probability of Backtest Overfitting (PBO). A PBO > 0.5 means selection of the "best" in-sample strategy is likely to be noise.

---

## 14. Makefile Shortcuts

```bash
make help               # list all targets

# Testing
make test               # all 1400+ GA tests
make test-ga            # genetic_algorithm/tests/ only
make test-infra         # engine + orchestration tests
make test-quick         # ~1 second (schema + registry + CLI)

# Runs
make smoke              # 2 gen × 5 pop quick check
make run CONFIG=my.yaml # run with a specific config
make benchmark          # queue all 8 benchmark configs

# Config
make validate-configs   # validate all presets + benchmark configs
make configs            # list available presets + benchmarks

# Operations
make queue-status       # show queue and running experiments
make monitor            # open terminal monitor
make experiments        # list recent experiments
make disk-report        # disk usage report
make cleanup            # clean up old runs (asks confirmation)

# Dev
make lint               # flake8 check
```

---

## 15. Reference — Full CLI Command Tree

```
python -m genetic_algorithm
│
├── run <config> [options]
│   ├── --tag <tag>            (repeatable)
│   ├── --name <name>
│   ├── --no-monitor
│   ├── --dashboard
│   ├── --resume <checkpoint_dir>
│   └── --yes / -y
│
├── monitor [options]
│   ├── --live
│   ├── --filter running|queued|completed|failed|all
│   ├── --tag <tag>
│   ├── --once
│   └── --interval <seconds>
│
├── queue
│   ├── add [configs...] [--dir <dir>] [--tag <tag>] [--priority <n>]
│   ├── start [--max-concurrent <n>] [--persistent]
│   ├── stop
│   └── status
│
├── experiment
│   ├── list [--status <s>] [--tag <tag>] [--limit <n>]
│   ├── show <experiment_id>
│   └── compare <id1> <id2> ...
│
├── data
│   ├── report
│   ├── cleanup [--dry-run]
│   └── backfill
│
├── config
│   ├── validate <config> [--strict]
│   ├── list [--benchmarks] [--all]
│   └── show <config>
│
└── serve [--host <host>] [--port <port>]
```

---

## 16. Config Key Reference

### Core GA parameters

| Key | Default | Notes |
|-----|---------|-------|
| `genetic_algorithm.population_size` | 30 | Individuals per generation (or per island) |
| `genetic_algorithm.generations` | 12 | Max generations |
| `genetic_algorithm.mutation_rate` | 0.20 | Base mutation rate |
| `genetic_algorithm.crossover_rate` | 0.75 | Probability of crossover |
| `genetic_algorithm.elite_size` | 4 | Top N carried over unchanged |
| `genetic_algorithm.random_immigrants` | 5 | Random new individuals injected per generation |
| `genetic_algorithm.convergence_patience` | 8 | Generations without improvement before stopping |
| `genetic_algorithm.adaptive_mutation` | true | Escalate mutation rate when stuck |
| `genetic_algorithm.fitness_sharing` | true | Penalize crowded fitness regions |
| `genetic_algorithm.mode` | `single_objective` | `single_objective` or `nsga2` |

### Selection methods

| Value | Description |
|-------|-------------|
| `tournament` | Tournament selection (default) |
| `rank` | Rank-based selection |
| `nsga2` | Non-dominated sorting (NSGA-II only) |
| `roulette` | Fitness-proportional |

### Island topologies

| Value | Description |
|-------|-------------|
| `ring` | Circular — each island receives from one neighbour |
| `fully_connected` | All-to-all migration |
| `tournament` | Best island donates to weakest |
| `hierarchical` | Tree structure |

### Walk-forward aggregation

| Value | Description |
|-------|-------------|
| `mean` | Arithmetic mean across windows |
| `harmonic_mean` | Harsher — punishes inconsistent windows |
| `min` | Most conservative — score = worst window |
| `median` | Robust to outlier windows |

---

## Tips & Common Pitfalls

**"My fitness is stuck around 0.2–0.3 after many generations."**  
→ Try the island model — it maintains better diversity.  
→ Increase `random_immigrants` to 10–15.  
→ Check `indicators.available` — broad indicator pools are harder to optimize.

**"The run crashes partway through."**  
→ Check `python -m genetic_algorithm data report` for disk space.  
→ Check `genetic_algorithm/logs/` for error messages.  
→ Resume from the last checkpoint: `--resume genetic_algorithm/runs/<id>/checkpoints/`

**"My strategy looks great in backtesting but poor out-of-sample."**  
→ Enable `holdout_monitoring` and `walk_forward`.  
→ Enable `deflated_sharpe`.  
→ Increase `holdout_pct` to 0.20–0.25.  
→ Use a longer `timerange` (3+ years) to reduce lookback bias.

**"I want to run 10 experiments overnight unattended."**  
```bash
python -m genetic_algorithm queue add --dir config/exploration/ --tag overnight
python -m genetic_algorithm queue start --max-concurrent 2 --persistent &
```

**"I want reproducible experiments."**  
```yaml
genetic_algorithm:
  random_seed: 42
```

**"Config validate shows weight warnings."**  
`fitness_weights` values should sum to ~1.0. Use `config show` to see the merged result, then adjust until `sum(weights) ≈ 1.0`.
