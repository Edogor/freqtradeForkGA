# Genetic Algorithm System — Comprehensive Summary

> A production-ready system for **autonomous evolution of cryptocurrency trading strategies** using genetic algorithms, integrated directly with the FreqTrade backtesting engine.

---

## Table of Contents

1. [High-Level Architecture](#1-high-level-architecture)
2. [Strategy Representation (The Genome)](#2-strategy-representation-the-genome)
3. [Genetic Operators](#3-genetic-operators)
4. [Fitness Evaluation](#4-fitness-evaluation)
5. [Overfitting Prevention](#5-overfitting-prevention)
6. [Island Model (Multi-Population Evolution)](#6-island-model-multi-population-evolution)
7. [NSGA-II Multi-Objective Optimization](#7-nsga-ii-multi-objective-optimization)
8. [LLM-Guided Evolution](#8-llm-guided-evolution)
9. [Automation & Experiment Management](#9-automation--experiment-management)
10. [Key Results & Observations](#10-key-results--observations)
11. [File Reference](#11-file-reference)

---

## 1. High-Level Architecture

The system evolves trading strategies through a standard GA loop: **initialize → evaluate → select → reproduce → replace → repeat**, with several advanced extensions (island model, NSGA-II, LLM guidance, regime awareness).

```
┌──────────────────────────────────────────────────────────────────────┐
│                          CONFIGURATION                               │
│   YAML config → GA parameters, fitness weights, validation mode      │
└──────────────┬───────────────────────────────────────────────────────┘
               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     INITIALIZATION                                    │
│   Random strategies + Hall of Fame injection + LLM seeds             │
└──────────────┬───────────────────────────────────────────────────────┘
               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                  EVOLUTION LOOP (per generation)                      │
│                                                                       │
│  ┌─────────┐   ┌──────────┐   ┌───────────┐   ┌──────────────────┐ │
│  │ EVALUATE │──▶│  SELECT  │──▶│ REPRODUCE │──▶│ REPLACE + ADAPT  │ │
│  │ fitness  │   │ parents  │   │ crossover │   │ elite + offspring │ │
│  │ backtest │   │ tourney/ │   │ mutation  │   │ + immigrants      │ │
│  │ validate │   │ rank     │   │           │   │ + adaptive rates  │ │
│  └─────────┘   └──────────┘   └───────────┘   └──────────────────┘ │
│                                                                       │
│  Optional: Island migration, merge rounds, LLM injection             │
│  Early stop: Convergence patience, holdout degradation trend         │
└──────────────┬───────────────────────────────────────────────────────┘
               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                       FINALIZATION                                    │
│   Save top-5 strategies → Python files                               │
│   Update Hall of Fame → persistent JSON archive                      │
│   Overfit analysis → SAFE / WARNING / OVERFIT assessment             │
│   Generate plots + visualizations                                    │
└──────────────────────────────────────────────────────────────────────┘
```

**Entry point:** `genetic_algorithm/run_ga.py` loads a YAML config, instantiates either a standard `GeneticAlgorithm` or a `GenericIslandModelEvolution`, and runs the full pipeline.

---

## 2. Strategy Representation (The Genome)

Every trading strategy is encoded as a `StrategyGene` — a structured genetic representation that can be converted into executable FreqTrade Python code.

### 2.1 Gene Components

| Component | Description | Examples |
|-----------|-------------|---------|
| **Indicators** | Technical analysis indicators with typed parameters | RSI(period=14), MACD(fast=12, slow=26), BBANDS(period=20, std=2.0) |
| **Entry Conditions** | AND/OR logic combining indicator signals | RSI < 30 AND MACD_cross_above(0) |
| **Exit Conditions** | When to close a position | RSI > 70 OR BBANDS_upper_cross |
| **Risk Management** | Stoploss, trailing stop, ROI table, max open trades | stoploss=-0.05, roi={0: 0.03, 30: 0.01} |
| **Multi-Timeframe** | Optional higher-timeframe informative indicators | 4h EMA for trend filter |
| **Short Trading** | Independent short entry/exit conditions (optional) | Separate condition set for shorts |
| **Regime Specialization** | Market regime preference + runtime filtering | Prefer bullish, skip bearish candles |

### 2.2 Indicator Library

Organized into **five families**, used for island specialization:

| Family | Indicators |
|--------|-----------|
| **Momentum** | RSI, MACD, STOCH, CCI, MFI, ROC, WILLR |
| **Trend** | EMA, SMA, TEMA, KAMA, SUPERTREND, AROON, ICHIMOKU, SAR |
| **Volatility** | BBANDS, ATR, DONCHIAN |
| **Volume** | OBV, CMF, VROC, VWAP |
| **Candlestick** | CDL_ENGULFING, CDL_DOJI, CDL_HAMMER, CDL_SHOOTINGSTAR, etc. |

Each strategy uses 2–5 indicators (configurable). The `IndicatorFactory` creates random instances with parameter ranges appropriate to each type (e.g., RSI period: 7–21, MACD fast: 8–21).

### 2.3 Condition Operators

Conditions compare indicator values to thresholds using operators:

| Operator Type | Examples |
|--------------|---------|
| **Comparison** | `<`, `>`, `<=`, `>=` |
| **Crossover** | `cross_above`, `cross_below` |
| **Range** | `between(low, high)` |
| **Trend** | `increasing(period)`, `decreasing(period)` |
| **Pattern** | Candlestick pattern triggers (CDL > 0) |

### 2.4 Code Generation

`StrategyGenerator.generate_strategy_code()` converts a `StrategyGene` into a complete FreqTrade strategy Python class. This generated code is then backtested directly via the FreqTrade Python API — no shell calls, no subprocess overhead.

---

## 3. Genetic Operators

### 3.1 Selection Methods

| Method | How It Works | Best For |
|--------|-------------|----------|
| **Tournament** (default) | Pick `k` random individuals, return the fittest | Strong selection pressure, simple |
| **Rank-based** | Probability proportional to rank (not raw fitness) | Avoids super-individual dominance |
| **Roulette** | Probability proportional to fitness value | Classic, biased toward high fitness |
| **NSGA-II** | Lower Pareto rank preferred, then higher crowding distance | Multi-objective optimization |

### 3.2 Crossover Methods

| Method | Description |
|--------|-------------|
| **Uniform** (default) | Each gene component randomly inherited from one parent |
| **Single-point** | Split both parents at a random point, swap tails |
| **Component** | Swap entire blocks (all indicators from parent A, conditions from parent B) |

Post-crossover repair ensures valid operator/threshold combinations, minimum condition counts, and deduplication.

### 3.3 Mutation Types

**Parameter mutations:**
- Indicator periods nudged within valid ranges (Gaussian perturbation)
- Condition thresholds adjusted (RSI: 0–100, CCI: -300 to 300)
- Risk parameters (stoploss, ROI, trailing stop) fine-tuned

**Structural mutations:**
- **Add indicator**: Insert a new random indicator (if below max)
- **Remove indicator**: Delete one (if above min)
- **Replace indicator**: Substitute one for another from the same or different family
- Adaptive weights based on feature importance

### 3.4 Adaptive Mutation Rate

When the population stagnates (no fitness improvement for N generations), the mutation rate automatically ramps up:

```
base_rate = 0.20
stagnation_factor = 1.0 + (no_improvement_count × adaptation_step)
effective_rate = min(base_rate × stagnation_factor, max_rate)  # capped at ~0.65
```

On improvement, the rate cools back down. This balances exploitation (low mutation when converging) with exploration (high mutation when stuck).

### 3.5 Elite Preservation & Hall of Fame

- **Elite size** (e.g., 4): Top individuals survive unchanged into the next generation.
- **Hall of Fame**: A persistent JSON archive of the best strategies ever found across all runs, deduplicated by genetic fingerprint. These can be re-injected into future runs as "seed" individuals.
- **Random immigrants**: Fresh random strategies injected each generation to prevent premature convergence.

---

## 4. Fitness Evaluation

### 4.1 Direct Backtesting

The `DirectBacktester` integrates directly with the FreqTrade Python API:

1. The `StrategyGene` is converted to Python source code
2. The code is loaded as a dynamic FreqTrade strategy class
3. A backtest is executed in-process (mocked exchange, real OHLCV data)
4. Results are cached by strategy content hash (LRU, up to 10,000 entries)

### 4.2 Metrics Collected

| Metric | Description |
|--------|-------------|
| **Profit %** | Total return over the backtest period |
| **Sharpe Ratio** | Risk-adjusted return (excess return / volatility) |
| **Sortino Ratio** | Like Sharpe, but only penalizes downside volatility |
| **Max Drawdown** | Largest peak-to-trough decline |
| **Win Rate** | Percentage of winning trades |
| **Trade Count** | Number of trades executed |
| **Profit Factor** | Gross profit / gross loss |
| **Monthly Returns** | Per-month P&L breakdown (for stability analysis) |
| **Per-Pair Profit** | Breakdown by trading pair |

### 4.3 Fitness Function (Weighted Composite)

The default fitness is a weighted sum:

```
fitness = 0.40 × profit + 0.30 × sharpe + 0.10 × sortino + 0.10 × profit_factor + 0.10 × trade_frequency
```

Weights are fully configurable. A **complexity penalty** is applied to strategies that use too many indicators, discouraging overly complex solutions.

### 4.4 Fitness Sharing (Diversity Preservation)

To prevent the entire population from converging to one solution:

1. **Calculate pairwise distance** between all individuals
   - Structural distance: Indicator types, condition counts, parameter values
   - Behavioral distance (optional): Compare per-pair profits and monthly returns
2. **Sharing function**: If two strategies are within `radius` (0.25–0.3), their fitness is reduced proportionally
3. **Effect**: Crowded niches get penalized → population maintains diversity across different strategy archetypes

---

## 5. Overfitting Prevention

This is the most critical challenge: strategies that look great on historical data but fail in live trading. The system has **four layers of defense**:

### 5.1 Walk-Forward Validation

Trains on a sliding window of past data, validates on the next unseen period, then steps forward:

```
|------ Train (60d) ------|-- Val (15d) --|  → step (15d) →
                     |------ Train (60d) ------|-- Val (15d) --|
```

The final fitness is the average validation performance across all windows. This prevents strategies from memorizing specific market patterns.

### 5.2 Pair-Split Validation

Instead of temporal splitting, the data is split **by trading pair**:

- **Training pairs**: BTC/USDT, BNB/USDT, XRP/USDT
- **Validation pairs**: ETH/USDT, SOL/USDT

The composite fitness is:

```
composite = train_fitness × 0.6 + val_fitness × 0.4
```

The **generalization ratio** (`gen_ratio = val_fitness / train_fitness`) measures cross-pair robustness:
- ~1.0 = excellent generalization
- < 0.8 = the strategy may be overfitting to specific pair behavior

### 5.3 Monte Carlo Simulation

After evolution completes, the top strategies are subjected to trade-permutation testing:
- Shuffle the order and outcomes of trades
- Repeat N times (e.g., 1,000 permutations)
- Report the percentage of permutations where the strategy remains profitable

A robustness score ≥ 80% indicates the strategy's edge is not due to lucky trade sequencing.

### 5.4 Holdout Monitoring & Early Stop

A portion of the data (e.g., 20%) is "locked away" during evolution. Periodically, the top strategies are tested on this holdout set. If holdout performance degrades beyond a threshold (e.g., 60%) for multiple consecutive checks, evolution stops early — the population is overfitting.

```yaml
holdout_monitoring:
  early_stop_threshold: 0.60    # maximum acceptable degradation
  early_stop_checks: 2          # consecutive bad checks needed to trigger
  trend_early_stop: true        # also check if degradation is trending up
```

### 5.5 Overfit Classification

Each final strategy receives a label:

| Label | Meaning |
|-------|---------|
| **SAFE** | Low degradation between train/val, passes Monte Carlo |
| **WARNING** | Moderate degradation, may be marginal |
| **OVERFIT** | High degradation, should not be deployed |
| **UNKNOWN** | Insufficient data to classify (e.g., holdout not run) |

---

## 6. Island Model (Multi-Population Evolution)

### 6.1 Concept

Instead of one large population, the system runs **N independent sub-populations (islands)** that evolve in parallel and periodically exchange individuals. This dramatically improves:
- **Exploration**: Different islands explore different parts of the solution space
- **Diversity**: Migration prevents total convergence while allowing specialization
- **Robustness**: Best individuals are tested across multiple contexts

### 6.2 Two Variants

#### Regime-Locked Islands (`IslandModelEvolution`)

Each island specializes in a **market regime** detected via technical analysis:

```
┌────────────────────────────────────┐
│           Master Island            │
│   (balanced data, generalist)      │
│                                    │
│    ▲ receives      sends ▼         │
├────┼──────────────────────┼────────┤
     │                      │
┌────┴────┐  ┌─────────┐  ┌┴────────┐
│ Bullish │  │ Bearish │  │Sideways │
│ Island  │  │ Island  │  │ Island  │
│ (uptrend│  │(downtrnd│  │(range   │
│  data)  │  │  data)  │  │  data)  │
└─────────┘  └─────────┘  └─────────┘
```

Regime detection uses configurable methods: SMA crossover + ADX, ADX + Directional Movement, Bollinger position, HMM, volatility clustering, or rolling returns.

#### Generic Islands (`GenericIslandModelEvolution`)

A more flexible model where islands are differentiated by **indicator families** rather than data regimes. Every island sees the full dataset but evolves with a unique combination of constraints:

**Three specialization dimensions:**

| Dimension | How It Works |
|-----------|-------------|
| **Seed rotation** | Each island gets a different random seed → different initial populations |
| **Indicator pool splitting** | Island 0 specializes in momentum+trend, Island 1 in trend+volatility, etc. (with 50% overlap) |
| **Pair rotation** | Each island trains on a different overlapping subset of pairs |

### 6.3 Migration Topologies

Islands exchange their best individuals at configurable intervals:

| Topology | Pattern | Overhead | Best For |
|----------|---------|----------|----------|
| **Ring** | Island 0→1→2→...→0 (circular) | O(N) | Large island counts, steady gene flow |
| **Fully Connected** | Every island sends to every other | O(N²) | Maximum information sharing, small N |
| **Tournament** | Pairs compete, winner sends to loser | O(N) | Competitive evolution pressure |
| **Hierarchical** | Bidirectional pair-wise exchange | O(N) | Symmetric communication |

**Migration mechanics:**
- Extract top-N individuals from source island (by raw fitness)
- **Deep-copy** the strategy gene (maintains independence)
- Replace the **worst** individuals in the target island
- Mark migrants as **unevaluated** → forces re-evaluation in the new context

### 6.4 Merge Rounds

Optional periodic "global pooling" phase:

1. Collect top-2N strategies from **every island**
2. Deduplicate by gene hash
3. Rank globally by fitness
4. Redistribute the global top-N back to **all islands**

This breaks local island stagnation and ensures the absolute best solutions are shared everywhere.

---

## 7. NSGA-II Multi-Objective Optimization

When `mode: 'nsga2'` is set, the system uses NSGA-II (Non-dominated Sorting Genetic Algorithm II) instead of a single weighted fitness.

### 7.1 Core Concepts

**Pareto Dominance**: Strategy A dominates strategy B if A is at least as good in all objectives and strictly better in at least one.

**Non-dominated sorting**: Assigns strategies to ranked fronts:
- **Front 1** (Pareto front): No strategy dominates any member
- **Front 2**: Dominated only by Front 1 members
- etc.

**Crowding distance**: Measures how isolated a strategy is within its front. Higher crowding = more unique = preferred (for diversity).

### 7.2 Objectives

Typically 2–3 objectives are optimized simultaneously:

```
objectives:
  - profit_pct          # Maximize return
  - sharpe_ratio        # Maximize risk-adjusted return
  - neg_max_drawdown    # Minimize drawdown (negated for maximization)
```

### 7.3 Selection

NSGA-II tournament selection prefers:
1. Lower Pareto rank (closer to the front)
2. If tied: higher crowding distance (more isolated)

This preserves a diverse set of trade-off solutions rather than converging to a single best.

---

## 8. LLM-Guided Evolution

An optional module that uses large language models (Grok, OpenAI, Anthropic, Groq, local models) to inject "intelligent" strategies into the population.

### 8.1 Three Integration Points

| Point | When | What Happens |
|-------|------|-------------|
| **Seed population** | Initialization | LLM generates `seed_ratio` (e.g., 40%) of the initial population with strategic indicator combinations |
| **Immigrants** | Every generation | LLM creates fresh strategies informed by top performers' weaknesses |
| **Mutation guidance** | Mid-evolution | When stagnation hits half-patience, LLM aggressiveness escalates |

### 8.2 Provider Architecture

A **router** supports multi-provider failover with priority ordering:

```
Primary: Grok-3-mini → Fallback: Groq Mixtral → Fallback: Local server
```

Each provider has:
- Independent cooldown tracking (on failure)
- Adaptive call interval (exponential backoff on errors, reset on success)
- Rate limit header parsing (Retry-After, x-ratelimit-reset)

### 8.3 Budget Enforcement

```yaml
max_calls_per_generation: 20     # prevent runaway API spend
max_calls_per_run: 500           # hard cap per complete GA run
```

When either budget is exhausted, the system silently falls back to random generation — the GA continues without interruption.

---

## 9. Automation & Experiment Management

### 9.1 Auto-Queue Daemon (`ga_auto_queue_v2.sh`)

A persistent daemon that:
1. Watches `genetic_algorithm/config/queue/` for YAML experiment configs
2. Launches up to N concurrent GA runs (default 5)
3. Moves completed configs to `done/` on finish
4. Logs everything to `genetic_algorithm/logs/auto_queue.log`

Configs are **priority-sorted** by filename prefix (e.g., `01_E27_...yaml` runs before `02_E28_...yaml`).

### 9.2 Wave Launcher (`run_parallel_wave.sh`)

Launches an entire wave of experiments (typically 6–8) in parallel:
- Pre-flight YAML validation
- Each process gets isolated output directories, checkpoints, and logs
- Records PIDs for monitoring
- Graceful Ctrl+C shutdown

### 9.3 Live Monitor (`ga_monitor_v2.sh`)

Real-time terminal dashboard showing:
- All running experiments with generation progress, best fitness, diversity
- Queue status (pending/running/completed counts)
- ETA estimates based on elapsed time and progress
- Configurable refresh interval (default 5s) and detail levels

### 9.4 Experiment History

Over **155+ experiments** have been run across **17+ waves**, exploring:

| Wave Range | Focus |
|------------|-------|
| W1–W2 | Baselines, bug discovery, initial tuning |
| W3–W4 | Island model debugging, walk-forward fixes |
| W5–W6 | Mutation rate sweeps, population sizing, LLM integration |
| W7–W8 | Selection method comparisons (rank vs tournament), patience tuning |
| W9–W11 | Elite sizing, seed reproducibility, LLM provider testing |
| W12–W13 | Diversity controls, NSGA-II, scaled populations |
| W14 | Multi-pair training, cross-asset generalization, island topology benchmark |
| W15 | Generic island model variants (ring, FC, specialization) |
| W17+ | Trade frequency constraints, pair-split validation |

---

## 10. Key Results & Observations

### 10.1 Best Known Configuration

**"R5 Multi-Pair"** remains the gold standard from early production runs:
- Fitness: 0.4127, Profit: +0.17%, 5/5 SAFE
- Used walk-forward + multi-pair cross-asset training

### 10.2 Top Experimental Results

| Experiment | Fitness | Notable Achievement |
|-----------|---------|-------------------|
| E8 (W2) | — | 5/5 SAFE, negative degradation (improved on holdout) |
| E33 (W6) | — | 85% Monte Carlo robustness (best ever) |
| E43 (W7) | 0.6350 | Highest raw fitness (LLM-guided) |
| E138 (W14) | **0.6703** | Overall best. Ring topology, 10×8 islands, pair-split |

### 10.3 Global Hall of Fame (Current)

| Rank | Fitness | Source |
|------|---------|--------|
| #1 | **0.6703** | E138 — Ring topology island model, Gen 11 |
| #2 | 0.6617 | E138 — Gen 8 |
| #3 | 0.6460 | E138 — Gen 9 |
| #4 | 0.6437 | E138 — Gen 5 |
| #5 | 0.6249 | E138 — Gen 1 |

### 10.4 Key Lessons Learned

1. **Ring topology dominates**: Outperforms fully-connected, tournament, and hierarchical for 15m data
2. **Pair-split validation works well**: gen_ratio ~0.95–1.10 indicates real generalization, not curve-fitting
3. **Soft trade-frequency penalties beat hard kills**: Zeroing fitness below a threshold destroys gradient signal for evolution
4. **5m timeframe underperforms**: Too noisy for the current fitness function weighting
5. **1h timeframe generates too few trades**: 12–14 trades inflates per-trade metrics but isn't practical
6. **Over-specialization hurts**: Restricting islands to narrow indicator families (E147) produced weaker results than diverse overlap
7. **Fitness sharing is essential**: Without it, populations converge prematurely to one strategy archetype
8. **Adaptive mutation works**: Auto-ramping from 0.20 to 0.65 during stagnation consistently helps escape local optima
9. **LLM seeding adds value — sometimes**: Provider reliability and prompt quality matter more than the model itself

---

## 11. File Reference

### Core Engine

| Path | Purpose |
|------|---------|
| `genetic_algorithm/run_ga.py` | Main entry point — loads config, runs GA |
| `genetic_algorithm/core/evolution.py` | `GeneticAlgorithm` — main evolution loop, patience, adaptive mutation |
| `genetic_algorithm/core/strategy_gene.py` | `StrategyGene` — genome representation |
| `genetic_algorithm/core/population.py` | Population management, fitness sharing, diversity |
| `genetic_algorithm/core/selection.py` | Tournament, rank, roulette, NSGA-II selection |
| `genetic_algorithm/core/crossover.py` | Uniform, single-point, component crossover |
| `genetic_algorithm/core/mutation.py` | Parameter + structural mutation operators |
| `genetic_algorithm/core/hall_of_fame.py` | Persistent strategy archive |
| `genetic_algorithm/core/nsga2.py` | NSGA-II non-dominated sorting + crowding |

### Island Models

| Path | Purpose |
|------|---------|
| `genetic_algorithm/core/island_model.py` | Regime-locked island model (bullish/bearish/sideways/master) |
| `genetic_algorithm/core/generic_island_model.py` | Generic island model (indicator families, pair rotation, topologies) |

### Evaluation

| Path | Purpose |
|------|---------|
| `genetic_algorithm/evaluation/fitness.py` | Fitness evaluation router (WF, pair-split, standard) |
| `genetic_algorithm/evaluation/direct_backtester.py` | FreqTrade Python API backtesting + result caching |
| `genetic_algorithm/evaluation/regime_aware.py` | Per-regime fitness aggregation |
| `genetic_algorithm/evaluation/monte_carlo.py` | Trade permutation robustness testing |
| `genetic_algorithm/evaluation/deflated_sharpe.py` | Overfitting-aware Sharpe ratio |
| `genetic_algorithm/evaluation/cpcv.py` | Combinatorial purged cross-validation |

### Strategy Generation

| Path | Purpose |
|------|---------|
| `genetic_algorithm/strategies/generator.py` | Converts StrategyGene → FreqTrade Python code |
| `genetic_algorithm/strategies/ensemble.py` | Combines top strategies via weighted voting |
| `genetic_algorithm/strategies/operator_registry.py` | Validates operators for indicator types |
| `genetic_algorithm/utils/indicator_factory.py` | Random indicator creation with typed parameter ranges |
| `genetic_algorithm/utils/regime_detector.py` | Market regime detection (SMA/ADX, HMM, Bollinger, etc.) |

### LLM Integration

| Path | Purpose |
|------|---------|
| `genetic_algorithm/llm/designer.py` | `StrategyDesigner` — seed generation, immigrants, mutation |
| `genetic_algorithm/llm/provider.py` | LLM provider implementations (OpenAI, Grok, Anthropic, Groq) |
| `genetic_algorithm/llm/router.py` | Multi-provider failover with cooldowns |
| `genetic_algorithm/llm/injector.py` | GA integration interface for LLM strategies |

### Automation

| Path | Purpose |
|------|---------|
| `ga_auto_queue_v2.sh` | Auto-queue daemon — watches queue dir, launches experiments |
| `ga_monitor_v2.sh` | Real-time terminal dashboard |
| `run_parallel_wave.sh` | Launch a wave of parallel experiments |
| `genetic_algorithm/config/queue/` | Pending experiment YAML configs |
| `genetic_algorithm/config/done/` | Completed experiment configs (138 so far) |
| `genetic_algorithm/output/exploration/` | Per-wave, per-experiment output directories |

### Configuration

| Path | Purpose |
|------|---------|
| `genetic_algorithm/config/ga_config.yaml` | Main GA configuration template |
| `genetic_algorithm/experiments/configs/` | Pre-built experiment configuration variants |
| `BEST_CONFIGS.md` | Tracks production-grade configurations |
| `CONFIG_RANKING.md` | Comprehensive history of all 155+ experiments |
