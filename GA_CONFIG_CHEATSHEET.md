# GA Configuration Cheatsheet

Complete reference for every configurable parameter. Organised by section.
Speed impact legend: 🐢 slower · 🐇 faster · ➖ neutral

---

## 1. `genetic_algorithm` — Core Parameters

### `random_seed`
- **What**: Integer seed for the PRNG; `null` = truly random each run.
- **Impact**: `null` means each run produces different results. A fixed seed makes runs reproducible for debugging.
- **Speed**: ➖
- **Advice**: Use `null` for production. Use a fixed integer only when you need to reproduce a specific run.

---

### `population_size`
- **What**: Number of strategies evolved simultaneously in one generation.
- **Impact**: Larger population = broader search space explored per generation, more genetic diversity, better final quality. Below ~20 the algorithm degrades to almost random search. Above ~200 you hit diminishing returns.
- **Speed**: 🐢 linear — each individual requires one full backtest evaluation.
- **Typical values**: Quick test 20–30 · Validation 50–80 · Production 80–150 · Server (parallel) 50.
- **Sweet spot**: 50–80 for overnight runs without parallelism.

---

### `generations`
- **What**: How many evolutionary cycles to run.
- **Impact**: More generations = more refinement of strong individuals, but also higher risk of premature convergence if diversity is low. `convergence_patience` can stop early.
- **Speed**: 🐢 linear — each generation evaluates (population_size − elite_size) backtests.
- **Typical values**: Quick 5–10 · Validation 12–20 · Production 25–50.

---

### `max_runtime_minutes`
- **What**: Hard wall-clock cap on total runtime. Evolution stops regardless of `generations` when this is exceeded.
- **Impact**: Safety net for server/overnight runs. Does not exist in all configs (add it when unsupervised).
- **Speed**: ➖ (only acts as a ceiling)
- **Advice**: Always set for unattended server runs (e.g. `720` = 12 h).

---

### `mutation_rate`
- **What**: Probability that any gene in a child individual is randomly altered after crossover.
- **Impact**: Controls exploration vs. exploitation. Too low → premature convergence. Too high → random walk, no learning.
- **Speed**: ➖ (affects quality, not wall-clock time directly)
- **Working range**: 0.10–0.30. Production default 0.15–0.20.
- **Interacts with**: `adaptive_mutation`, `max_mutation_rate`, `mutation_cooldown_factor`.

---

### `max_mutation_rate`
- **What**: Ceiling for the mutation rate when `adaptive_mutation` is raising it.
- **Impact**: Prevents the algorithm from entering pure-random-search mode when stuck.
- **Speed**: ➖
- **Advice**: Keep at 0.35–0.45. Setting it near 1.0 effectively disables the cap.

---

### `crossover_rate`
- **What**: Probability that two selected parents actually exchange genetic material (vs. child = copy of parent).
- **Impact**: High value (0.7–0.85) is standard; too low reduces recombination and slows convergence.
- **Speed**: ➖
- **Advice**: 0.70–0.80 works well for all topologies.

---

### `crossover_method`
| Value | Behaviour | Best for |
|---|---|---|
| `uniform` | Each gene independently inherited from either parent | **Recommended default** — most diversity |
| `single_point` | Split indicator/condition lists at one point, swap tails | Fast, good for tightly coupled genes |
| `component` | Swap entire blocks (all indicators, all entry conds, etc.) | Large-scale structure swaps |

- **Speed**: ➖ (negligible difference)
- **Advice**: `uniform` gives the best general-purpose diversity.

---

### `elite_size`
- **What**: Number of the top-scoring individuals that are copied unchanged to the next generation.
- **Impact**: Prevents losing the current best solution. Too large → population stagnates because elites fill too many slots. ~5–15 % of population is ideal.
- **Speed**: 🐇 slight — elites skip re-evaluation.
- **Advice**: `elite_size ≈ 0.10 × population_size`. E.g. pop 80 → elite 8–10.

---

### `tournament_size`
- **What**: Number of individuals randomly drawn from the population; the winner (highest fitness) becomes a parent.
- **Impact**: **Critical parameter.** Tournament size 1 = random selection (no evolutionary pressure). Size 3 is the gold standard. Large sizes (>5) cause premature convergence because top individuals monopolise breeding.
- **Speed**: ➖
- **Advice**: Use 3 for single-objective. Island model specialist configs often use 4 for slightly higher pressure.
- **⚠️ Avoid**: Setting to 1 (random walk) or >6 (elitist collapse).

---

### `selection_method`
| Value | Description |
|---|---|
| `tournament` | Pairwise tournament — **recommended, most stable** |
| `roulette` | Fitness-proportionate — sensitive to outlier fitnesses |
| `rank` | Selection based on rank not raw score — robust to outliers |

- **Advice**: Always use `tournament` unless you have a specific reason.

---

### `convergence_patience`
- **What**: Stop evolution early if the best fitness does not improve for this many consecutive generations.
- **Impact**: Saves runtime when evolution plateaus.
- **Speed**: 🐇 (early exit)
- **Advice**: 8–15. Higher values are better for large populations that need time to converge gradually. Island configs benefit from higher patience (10–15) due to regime diversity.

---

### `adaptive_mutation`
- **What**: When `true`, automatically raises `mutation_rate` (up to `max_mutation_rate`) when no improvement is seen for several generations.
- **Impact**: Helps escape local optima. Works as a built-in plateau-escaping mechanism.
- **Speed**: ➖
- **Advice**: Always `true` for production.
- **Interacts with**: `max_adaptation_factor`, `adaptation_step`, `mutation_cooldown_factor`.

---

### `max_adaptation_factor`
- **What**: Maximum multiplier applied to base `mutation_rate` during adaptation (e.g. `2.5` → rate can reach `0.15 × 2.5 = 0.375`).
- **Impact**: Caps how wild the mutation gets when stuck.
- **Speed**: ➖
- **Advice**: 2.0–2.5. Higher leads to more disruptive exploration when stuck.

---

### `adaptation_step`
- **What**: How much the adaptation factor increases each stagnant generation.
- **Impact**: Determines how quickly mutation escalates. Lower = gradual ramp (0.10); higher = fast ramp (0.20).
- **Speed**: ➖
- **Advice**: 0.10–0.15 is a good balance.

---

### `mutation_cooldown_factor`
- **What**: After the mutation rate is elevated and a new best is found, controls how fast it decays back to the base rate. `0.0` = instant reset; `0.9` = very slow decay.
- **Impact**: Smooth decay prevents the algorithm from immediately exploiting a new basin before having explored its neighbourhood.
- **Speed**: ➖
- **Advice**: `0.5` is a reasonable default. Use `0.0` (legacy) only if you want the old snapping behaviour.

---

### `fitness_sharing`
- **What**: Reduces the fitness of individuals that are very similar to others in the population, discouraging the population from clustering around a single niche.
- **Speed**: 🐢 slight — requires pairwise distance computation.
- **Advice**: `true` for production. Required when diversity matters.

---

### `sharing_radius`
- **What**: If the structural distance between two individuals is below this value (0–1), they share fitness.
- **Impact**: Smaller = only very similar individuals share → more clusters preserved. Larger = broader sharing → stronger diversity pressure.
- **Advice**: 0.15–0.30. Island configs can use smaller (0.18) since islands already provide structural separation.

---

### `diversity_threshold`
- **What**: Minimum genetic diversity in the population. When diversity drops below this, `random_immigrants` injection is doubled.
- **Impact**: Emergency mechanism to prevent convergence collapse.
- **Advice**: 0.10–0.20.

---

### `allow_self_crossover`
- **What**: When `false`, the same individual cannot be selected as both parents.
- **Impact**: `false` ensures actual recombination always occurs; setting `true` allows a child to be a copy of one parent (with mutations only).
- **Advice**: Always `false`.

---

### `random_immigrants`
- **What**: Number of completely random new individuals injected into each generation (replacing worst individuals).
- **Impact**: Continuous source of fresh genetic material. Prevents premature convergence. When diversity drops below `diversity_threshold`, this count is doubled automatically.
- **Speed**: 🐢 small — each immigrant needs one backtest.
- **Advice**: ~5–10 % of population. E.g. pop 80 → 5–8 immigrants.

---

### `mode`
| Value | Description |
|---|---|
| `single_objective` | Weighted sum fitness, single best strategy returned |
| `nsga2` | Pareto front, returns set of non-dominated strategies |

- **⚠️ Incompatibility**: When `nsga2` is selected, `fitness_weights` still exist but are used differently; the NSGA-II section's `objectives` take control. Do not mix the two.
- **Advice**: `single_objective` is more stable and easier to tune. Use `nsga2` when you explicitly want a Pareto trade-off between profit and risk.

---

### `behavioral_distance_weight`
- **What**: Controls whether fitness sharing uses structural gene distance (0.0) or actual backtest performance distance (1.0), or a blend.
- **Impact**: Pure behavioral (1.0) measures diversity by how differently strategies actually trade, not just what indicators they use. More meaningful but only works for evaluated individuals.
- **Speed**: ➖ (behavioral data is already available post-evaluation)
- **Advice**: 0.0 (default) is safe. Try 0.3–0.5 in mature runs where you want truly diverse trading behaviour.

---

### `adaptive_tournament`
- **What**: Dynamically adjusts `tournament_size` based on population diversity. High diversity → larger tournament (exploit); low diversity → smaller (explore).
- **Speed**: ➖
- **Stability**: ⚠️ Experimental — can interact unpredictably with `adaptive_mutation`. Use only if you understand the interaction.
- **Advice**: Leave `false` unless experimenting.

---

## 2. `nsga2` — Multi-Objective Optimisation

Only active when `genetic_algorithm.mode = 'nsga2'`.

### `objectives`
List of metrics to optimise simultaneously. Each entry has:
- `name`: metric key (e.g. `profit`, `max_drawdown`, `sharpe_ratio`)
- `type`: `maximize`, `minimize`, or `goldilocks`
- `scale`: normalisation divisor

**Advice**: 2–3 objectives is the practical limit. Beyond 3, the Pareto front becomes too large and selection pressure collapses.

### `pareto_front_size`
- Cap on how many non-dominated strategies are returned.
- **Advice**: 10–30. Larger gives more choice but bloats storage.

---

## 3. `fitness_weights` — Scoring

All weights must sum to 1.0. Each metric's contribution to the final fitness score.

| Key | What it measures | Notes |
|---|---|---|
| `profit` | Total % return | Primary driver. Increase for raw-profit focus (default 0.22–0.30). |
| `sharpe_ratio` | Risk-adjusted return vs volatility | Important for live trading. Increase for consistency. |
| `sortino_ratio` | Sharpe but only penalises downside volatility | Better than Sharpe for crypto — reward upside volatility. |
| `profit_factor` | Gross profit / gross loss | Favours strategies that win big and lose small. |
| `drawdown` | Maximum drawdown penalty | Increase (0.15–0.22) for conservative strategies. |
| `win_rate` | % winning trades | Avoid weighting too heavily — can push strategies to take tiny wins. |
| `trade_frequency` | Number of trades (too few or too many is bad) | Goldilocks metric; thresholds set in `trade_frequency_thresholds`. |
| `monthly_stability` | Std dev of monthly returns | Low weight (0.04) but helps penalise boom-bust strategies. |
| `cross_pair` | Consistency across all pairs | Set to 0.0 when using single-pair island config. |

**Common weight profiles**:
- **Consistency-focused** (live trading): Sharpe + Sortino + Drawdown ≥ 0.50
- **Profit-focused** (research): Profit ≥ 0.30, relax risk weights
- **Risk-managed**: Drawdown 0.20–0.25, profit 0.20–0.25

---

## 4. `fitness_penalties`

Hard thresholds below which strategies receive a scaled fitness reduction.

| Key | What it does | Notes |
|---|---|---|
| `min_trades` | Penalise if fewer trades than this | Set relative to timerange length. Shorter timeranges → lower min_trades. |
| `max_drawdown` | Penalise if drawdown exceeds this | Must be consistent with `strategy_constraints.max_drawdown`. |
| `min_win_rate` | Penalise if win rate below this | 0.35 is standard; island specialists may use 0.40. |
| `complexity_weight` | Per-indicator/condition penalty factor | Increases with number of indicators + conditions. Higher = simpler strategies. |
| `unused_indicator_weight` | Penalty for indicators not referenced by any condition | Cleans up "orphan" indicators. |
| `pair_loss_threshold` | Max acceptable loss (%) on any single pair | Prevents strategies that implode on one specific pair. |

---

## 5. `fitness_bounds`

Clamps extreme metric values before they are normalised, preventing outliers from skewing the whole population's relative ranking.

| Key | Default | Purpose |
|---|---|---|
| `profit_min/max` | −50 / 200 | Floor/ceiling for profit % |
| `sharpe_min/max` | −5 / 10 | Sharpe ratio clamp |
| `sortino_min/max` | −5 / 12 | Sortino ratio clamp |
| `profit_factor_max` | 10 | Caps suspiciously high PF (usually only 1–3 trades) |

**Advice**: Only change these if your timerange is very unusual. Leaving defaults is safe.

---

## 6. `trade_frequency_thresholds`

Defines the goldilocks zone for number of trades. Strategies outside it are penalised.

| Key | Typical value | Meaning |
|---|---|---|
| `very_few` | 3–5 | Heavily penalised below this |
| `few` | 6–10 | Some penalty |
| `ideal_min` | 6–10 | Start of ideal zone |
| `ideal_max` | 40–50 | End of ideal zone |
| `moderate_excess` | 80–100 | Slight penalty begins above ideal_max, then sharper above this |

**Advice**: Scale with timerange. 12-month range → ideal_min = 10, ideal_max = 50. 90-day range → ideal_min = 5, ideal_max = 20.

---

## 7. `backtesting` — Data & Environment

| Key | What | Advice |
|---|---|---|
| `timerange` | `YYYYMMDD-YYYYMMDD` date window | Longer = slower but more robust. Min 90 days for production. |
| `stake_amount` | `0.01–0.99` = fraction of wallet; `≥1` = fixed amount | Use fraction for relative performance comparison. |
| `pairs` | List of trading pairs | 2–3 pairs for fast runs; 5–10 for robust production. More pairs = 🐢. |
| `max_open_trades` | Max concurrent trades per strategy | Must match `strategy_constraints.max_open_trades_range`. |
| `fee` | Exchange fee per trade | Binance 0.001 (0.1%). Slippage adds realistic cost. |
| `slippage_pct` | Additional slippage fraction | 0.0005 = 0.05% — conservative for BTC/USDT/ETH. |
| `auto_download_data` | Download missing OHLCV from exchange | Set `false` for offline/air-gapped servers. |
| `enable_cache` | Cache backtest results | Always `true` — dramatically reduces repeated evaluation time. |
| `timeout` | Max seconds per single strategy backtest | 45–120 s. Kill hung backtests. Lower = 🐇 but risks cutting slow but valid strategies. |

**Speed rule of thumb**: `total_time ≈ population_size × generations × avg_backtest_seconds / parallel_workers`

---

## 8. `walk_forward` — Overfitting Protection

The most important anti-overfitting mechanism.

| Key | What | Advice |
|---|---|---|
| `enabled` | Toggle walk-forward | Always `true` for production. Disable only for quick smoke tests. |
| `train_days` | Training window size in days | 60–120 days. Must leave room for validation windows. |
| `validation_days` | Out-of-sample validation window | 15–30 days. Needs ≥ `min_train_trades` during this period. |
| `step_days` | Slide increment | Equal to or half of `validation_days` for good coverage. |
| `mode` | `rolling` (fixed-size window) or `anchored` (growing window) | `rolling` is standard — prevents early data dominating. |
| `aggregation` | How to combine scores across windows | See table below. |
| `embargo_days` | Gap between train and validation | 3–5 days prevents data-leakage from autocorrelation. |
| `min_train_trades` | Skip windows with fewer trades | Prevents very sparse windows polluting scores. |
| `max_windows` | Cap total window count | Limits runtime for long timeranges. `null` = no cap. |
| `gap_penalty` | Penalise big train→validation gap (overfitting signal) | Enable in production; threshold 0.10, max_penalty 0.5. |

**Aggregation methods**:
| Method | Behaviour | Best for |
|---|---|---|
| `mean` | Average across windows | Balanced, default |
| `min` | Worst window score | Maximum conservatism |
| `harmonic_mean` | Penalises any zero/bad window very hard | **Recommended for consistency-focused runs** |
| `weighted` | More weight to recent windows | Adapts to recent market conditions |

**Speed impact**: Each walk-forward window multiplies evaluation time by `num_windows`. A 365-day range with train=60, step=15 creates ~20 windows → 20× slower than no walk-forward.

---

## 9. `holdout_validation` — Final Out-of-Sample Test

Runs after evolution completes on a completely unseen data slice (end of timerange).

| Key | What | Advice |
|---|---|---|
| `enabled` | Toggle | `true` for production; `false` for quick tests. |
| `holdout_pct` | Fraction of timerange reserved as holdout | 0.15–0.20. This data is never seen during evolution. |

**⚠️ Note**: When `regime_aware.enabled = true`, the regime module manages its own holdout via `holdout_ratio`. Do not double-count — consider disabling `holdout_validation` when `regime_aware` is on.

---

## 10. `holdout_monitoring` — Live Overfit Detection

Checks whether top strategies are degrading on holdout data *during* evolution and can stop early or penalise overfit individuals.

| Key | What | Advice |
|---|---|---|
| `enabled` | Toggle | `true` in production. |
| `interval` | Check every N generations | 2–3. |
| `top_n` | How many top strategies to evaluate on holdout | 3–5. |
| `early_stop` | Stop evolution if degradation is consistently high | `true`. |
| `early_stop_threshold` | Degradation fraction above which a check "fails" (e.g. 0.60 = 60% fitness drop) | 0.50–0.70. |
| `early_stop_checks` | Consecutive failing checks before stopping | 3. |
| `trend_early_stop` | Stop if degradation consistently worsens (trend-based) | `true`. |
| `fitness_penalty` | Apply penalty to overfit individuals' fitness | `true`. |
| `penalty_factor` | Aggressiveness (0.0 = no effect, 1.0 = zero fitness for overfit) | 0.7–0.9. |

**⚠️ Interaction with island model**: Disable `holdout_monitoring` when using island model configs (islands use regime segments as their own temporal diversity mechanism).

---

## 11. `regime_aware` — Market Regime Evaluation

Evaluates strategies separately on bullish, bearish, and sideways market segments.

| Key | What | Advice |
|---|---|---|
| `enabled` | Toggle | `true` for production; the best anti-overfitting tool alongside walk-forward. |
| `method` | Detection algorithm | `adx_di_hysteresis` is most reliable (avoids sideways-only misclassification). `ensemble` = best for island model. |
| `benchmark_pair` | Pair used for regime detection | `BTC/USDT` for consistent labelling across runs. |
| `detection_timeframe` | TF for regime detection | `4h` gives smoother regimes than `1h`. |
| `period_days` | Target segment length in days | 60–90. Shorter = more segments but noisier. |
| `min_period_days` | Minimum acceptable segment length | ~60% of `period_days`. |
| `embargo_days` | Gap between regime segments | 3–5 days — same principle as walk-forward embargo. |
| `segments_per_regime` | How many segments per regime type | 3–4. More = 🐢 but more robust. |
| `holdout_ratio` | Fraction of segments reserved as holdout | 0.20. |
| `aggregation` | How to combine regime segment scores | `harmonic_mean` recommended (see walk_forward table). |
| `segmentation` | `adaptive` (change-point based) or `fixed` (fixed-width) | `adaptive` is better — internally consistent segments. |

**Sub-features inside `regime_aware`**:

#### `regime_specialization`
- Allows strategies to evolve a `preferred_regime` gene (bullish/bearish/sideways).
- `specialist_boost`: weight multiplier for their preferred regime's score.
- `diversity_weight`: bonus for cross-regime consistency (generalist pressure).
- **Advice**: Enable for island model where each island targets one regime.

#### `ml_regime`
- LightGBM classifier as a meta-ensemble voter on top of rule-based methods.
- **⚠️ Must train the model first** before enabling.
- Adds complexity — only use after all other pieces are stable.

#### `in_strategy_regime`
- Injects runtime regime detection code into generated strategies, so they can filter entries by current regime at live-trading time.
- **Stability**: ⚠️ Experimental — adds non-trivial complexity to generated code. Test thoroughly before deploying live strategies.

#### `mtf_enabled` (within `regime_aware`)
- Multi-timeframe regime detection: fuses classifications from several timeframes.
- **Speed**: 🐢 — multiple TA computations per segment.
- **Advice**: Useful for robustness but disable for initial experiments.

---

## 12. `strategy_constraints` — Gene Space Limits

These constrain what the GA is *allowed to generate*, independent of fitness penalties.

| Key | What | Notes |
|---|---|---|
| `min_trades` | Minimum trades for a strategy to be considered valid (hard reject) | Distinct from `fitness_penalties.min_trades` which is a soft penalty. |
| `max_drawdown` | Hard reject above this drawdown | Consistent with `fitness_penalties.max_drawdown`. |
| `min_win_rate` | Hard reject below this win rate | |
| `timeframes` | Which trading timeframes strategies can use | Shorter TFs (`15m`) = 🐢 more candles to process. |
| `stoploss_range` | `[min, max]` as negatives | Tighter SL = more frequent stops but less catastrophic loss. |
| `roi_range` | `[min, max]` profit targets | |
| `max_open_trades_range` | `[min, max]` concurrent trades per strategy | Must be consistent with `backtesting.max_open_trades`. |
| `min_exit_conditions` | Minimum number of exit conditions required | ≥1 prevents exit-less strategies. |

---

## 13. `multi_timeframe` — Higher-TF Confirmation

Allows generated strategies to use indicators computed on a higher timeframe as confirmation signals.

| Key | What | Notes |
|---|---|---|
| `enabled` | Toggle | Disabled for speed/simplicity in most configs. |
| `available` | List of higher TFs available | Must be strictly higher than `strategy_constraints.timeframes`. |
| `max_timeframes` | Max higher TFs one strategy can use | 1–2. More = more complexity + slower backtest. |
| `higher_timeframe_preference` | Indicators preferred on higher TFs | Trend indicators (EMA, SMA, ADX) work best on higher TFs. |

**Speed**: 🐢 — each higher TF requires downloading and processing additional OHLCV data.
**⚠️ Compatibility**: Disable when using `island_model` — MTF + island model tested unreliably. The island config's regime segments already provide multi-regime diversity.

---

## 14. `indicators`

### `available`
List of indicator types the GA can pick. Fewer choices = faster convergence on known-good indicators. More = broader search at cost of convergence time.

### `max_per_strategy` / `min_per_strategy`
- Lower max (2–4) → simpler, more generalisable strategies, faster backtest.
- Higher max (5–7) → richer strategies, more overfit risk, slower backtest.
- **Advice**: 2–4 for production.

### `min_entry_conditions` / `min_exit_conditions`
- Prevents trivial strategies with a single condition or no exit logic.

### Per-indicator parameter ranges
Every indicator has `[min, max]` ranges for its period and threshold parameters. The GA samples uniformly within these ranges.

| Indicator | Key Parameters | Notes |
|---|---|---|
| RSI | `period`, `buy_threshold`, `sell_threshold` | Classic — works across regimes. |
| MACD | `fast_period`, `slow_period`, `signal_period` | Trend-following. Ensure `fast < slow`. |
| BBANDS | `period`, `std_dev` | Volatility + mean-reversion signals. |
| EMA / SMA | `period` | Trend direction. SMA is smoother; EMA reacts faster. |
| STOCH | `k_period`, `d_period`, thresholds | Momentum oscillator. |
| ATR | `period` | Volatility — usually used for SL sizing, not just signals. |
| ADX | `period`, `threshold` | Trend strength — great complement to directional indicators. |
| CCI / MFI / WILLR | periods + thresholds | Oscillators; tend to be noisier, use sparingly. |
| SUPERTREND | `period`, `multiplier` | Strong trend-following. |
| PSAR | `acceleration`, `maximum` | Excellent for trailing stop signals. |
| CMF / VROC | period + thresholds | Volume-based — adds info orthogonal to price. |
| Candlestick patterns | no periods | Binary (pattern present/absent). Combine with a confirming indicator. |

---

## 15. `island_model` / `generic_island_model` — Island Topology

The island model splits the population into separate sub-populations ("islands") that evolve independently and periodically exchange individuals via migration. This maintains diversity and allows regime specialization.

### Basic `island_model` (in `advanced` or top-level)
```yaml
island_model:
  enabled: true
  num_islands: 4
  migration_interval: 5   # Exchange every N generations
  migration_size: 5        # Individuals sent per migration event
```
- Simple version — all islands are identical, just isolated.

### Full `island_model` (top-level, server configs)
```yaml
island_model:
  enabled: true
  parallel_islands: true/false

  islands:
    - name: 'bullish'
      population_size: 30
      data_regime: 'bullish'
    - name: 'bearish'
      ...
    - name: 'master'
      data_regime: 'balanced'

  migration:
    specialist:
      interval: 3
      count: 2
      topology: 'fully_connected'
    master:
      interval: 2
      count: 3
```

### `generic_island_model` (distributed configs)
Adds `specialization` and `external_migration` for cross-machine exchange.

| Key | What | Notes |
|---|---|---|
| `num_islands` | Number of sub-populations | More islands = more diversity, but much slower without `parallel_islands`. |
| `population_per_island` | Per-island size | Total effective population = `num_islands × population_per_island`. |
| `parallel_islands` | Evolve islands concurrently using threads | See compatibility note below. |
| `migration.topology` | `ring`, `fully_connected`, `star` | `fully_connected` = all islands share migrants; `ring` = only neighbours. |
| `migration.interval` | Migrate every N generations | 3–5. Too frequent = populations merge prematurely. |
| `migration.count` | Individuals exchanged per event | 2–5. |
| `external_migration.enabled` | Export/import strategies from filesystem | For multi-machine distributed runs. |

**⚠️ Critical Compatibility Notes**:

| Combination | Status | Note |
|---|---|---|
| `island_model` + `parallel_evaluation` | ✅ Works | Islands share the worker pool. Workers are the bottleneck, not the islands. |
| `island_model` + `parallel_islands: true` | ✅ Works on server | Uses thread-based island orchestration. Needs `parallel_evaluation` workers to back it up. |
| `island_model` + `walk_forward` | ⚠️ Usually disabled | Regime segments already provide temporal diversity. Enabling both creates very long runs. |
| `island_model` + `holdout_monitoring` | ⚠️ Disable | Holdout monitoring was designed for single-population runs. Off in island configs. |
| `island_model` + `multi_timeframe` | ⚠️ Untested/unstable | Complex interaction — avoid until stable. |
| `island_model` + `nsga2` | ⚠️ Untested | Mode conflict potential. Not recommended. |
| `parallel_islands: true` + single machine, no `parallel_evaluation` | 🐢 Deadlock risk | Threads compete for a sequential evaluator. Always pair with `parallel_evaluation`. |

---

## 16. `parallel_evaluation`

Controls the multi-process backtest evaluation pool.

| Key | What | Notes |
|---|---|---|
| `enabled` | Toggle | False by default. Enable on multi-core machines. |
| `num_workers` | Worker process count. `null` = auto (cpu_count − 1) | Leave `null` unless you want to limit. |
| `backtest_timeout` | Per-backtest timeout in seconds | 60–120. Must match or exceed `backtesting.timeout`. |
| `memory_aware` | Dynamically cap workers if RAM is low | Always `true` — prevents OOM kills. |
| `estimated_worker_memory_mb` | RAM estimate per worker (MB) | FreqTrade + data ≈ 600–1000 MB per worker. |
| `min_free_ram_mb` | Minimum free RAM to keep (MB) | 2048 = 2 GB for OS + main process. |

**Speed**: 🐇 Near-linear scaling up to cpu_count − 1. Beyond that, Python GIL and I/O contention kills gains.

**Server rule**: 6-core → `num_workers: 5`. 8-core → 6–7.

---

## 17. `advanced.llm` — LLM Strategy Injection

Uses a large language model (Groq/llama) to generate candidate strategies and inject them as immigrants.

| Key | What | Notes |
|---|---|---|
| `enabled` | Toggle | Requires `GROQ_API_KEY` env var. |
| `model` | LLM model name | `llama-3.3-70b-versatile` is fast and capable. |
| `temperature` | LLM sampling temperature | 0.6–0.8. Higher = more novel but less valid strategies. |
| `seed_strategies_per_generation` | LLM strategies injected per generation | Competes with `random_immigrants` slots. |
| `include_top_performers` | Pass best strategies as examples to LLM | `true` — guides LLM toward the known good region. |
| `diversity_boost` | Encourage LLM to differ from existing examples | `true`. |
| `min_call_interval` / `max_call_interval` | Rate limiting (seconds between API calls) | Prevents Groq rate-limit errors. |
| `max_calls_per_generation` / `max_calls_per_run` | Hard caps on API calls | Budget control. |
| `batch_enabled` / `max_batch_size` | Request multiple strategies per API call | Efficient — reduces calls by up to batch_size factor. |
| `mutation_enabled` | Let LLM mutate existing strategies instead of generating from scratch | Focused exploration near known-good strategies. |
| `mutation_top_k` | How many top strategies to offer for mutation | 3. |

**⚠️ Note**: If the Groq API is unreachable or rate-limited, the GA falls back to random immigrants seamlessly — it will not crash.

---

## 18. `hall_of_fame`

Maintains a persistent registry of the best-ever strategies across checkpoints and restarts.

| Key | What | Notes |
|---|---|---|
| `enabled` | Toggle | Always `true` for production. |
| `max_size` | Maximum stored strategies | 30–100. |
| `inject_count` | How many HoF strategies to inject per generation | 2. Acts as a form of "memory elitism". |
| `min_fitness` | Minimum fitness to enter HoF | Prevents garbage from polluting the register. |
| `directory` | Filesystem path | Keep separate paths for different run types. |

---

## 19. `deflated_sharpe`

Statistical test that corrects the Sharpe ratio for multiple testing bias. Uses the number of strategies evaluated to penalise overfitted high-Sharpe outliers.

| Key | What | Notes |
|---|---|---|
| `enabled` | Toggle | Always `true` in production. |
| `min_trades` | Minimum trades for the test to be valid | 5. |
| `benchmark_sr` | Baseline Sharpe ratio to test against | 0.0 = test if SR > 0. |
| `skewness_correction` | Apply higher-moment correction | `true` — important for crypto's fat-tailed returns. |

**Speed**: ➖ (computed post-evaluation, negligible overhead)

---

## 20. `parsimony`

Post-evaluation pruning: simplifies strategies that score nearly the same with fewer indicators.

| Key | What | Notes |
|---|---|---|
| `enabled` | Toggle | `true` in production. |
| `tolerance` | Fitness loss acceptable from pruning (e.g. 0.02 = 2% drop OK) | Lower = more aggressive pruning. |
| `max_removals_per_strategy` | Cap on how many indicators can be pruned | 1–2. |

**Benefits**: Reduces overfitting, speeds up live-trading strategies, reduces backtest time for future generations.
**Speed**: 🐢 slight — runs after evaluation, requires a re-evaluation pass.

---

## 21. `storage`

| Key | What | Notes |
|---|---|---|
| `database` | SQLite DB path | Keep separate files per run type. |
| `strategy_dir` | Where generated `.py` strategy files are saved | Used by FreqTrade directly for live trading. |
| `checkpoint_dir` | Where generation checkpoints are saved | Critical for resuming interrupted runs. |
| `checkpoint_interval` | Save checkpoint every N generations | Lower (3) for unattended server runs; 5 is fine locally. |
| `keep_history` | Retain all generations' data or only latest | `true` for analysis; `false` to save disk space. |

---

## 22. `logging`

| Key | What | Notes |
|---|---|---|
| `level` | `DEBUG`, `INFO`, `WARNING` | `INFO` for production; `DEBUG` for development. |
| `file` | Log file path | Keep per-run. Tail with `tail -f` for remote monitoring. |
| `console` | Also print to stdout | `false` in daemonised server runs to avoid log spam. |

---

## 23. `holdout_test` (legacy)

Simple alternative to `holdout_validation` for a manually specified separate holdout timerange.

```yaml
holdout_test:
  enabled: true
  timerange: "20250301-20250401"  # Must NOT overlap with backtesting.timerange
  top_n: 5
```

**Note**: `holdout_validation` (using `holdout_pct`) is more automated. Use `holdout_test` when you want precise control over the test date window.

---

## 24. `short_selling`

```yaml
short_selling:
  enabled: false
  probability: 0.5
```

- Require futures/margin exchange support.
- When `enabled: true`, strategies can evolve short entry/exit conditions.
- `probability`: fraction of new individuals seeded with short capability.
- **⚠️ Note**: Only enable if your exchange config and data support margin/futures.

---

## 25. `web_dashboard`

Local web dashboard for visualising evolution in real time (Streamlit-based).

```yaml
web_dashboard:
  host: "0.0.0.0"
  port: 8501
  open_browser: false
  save_generation_snapshots: true
  max_snapshot_generations: 500
```

- **Speed**: ➖ (runs as a separate process)
- **Advice**: `open_browser: false` for server runs. View in browser manually on `http://server-ip:8501`.

---

## Key Topology Combinations — Stability Reference

| Topology | Walk-Forward | Regime-Aware | Parallel Eval | Notes |
|---|---|---|---|---|
| Standard single-pop | ✅ | ✅ | ✅ | Most stable baseline |
| Island model (sequential) | ⚠️ Slow | ✅ via regime segments | ✅ | Disable WF; use regime for temporal diversity |
| Island model (parallel_islands) | ⚠️ Slow | ✅ | **Required** | Must have `parallel_evaluation` or parallel island threads starve |
| Distributed (multi-machine) | ⚠️ Slow | Optional | ✅ per machine | Each machine runs independently; `external_migration` bridges them |
| NSGA-II | ⚠️ | ⚠️ | ✅ | `nsga2` mode + `walk_forward` = very long runs; test capacity first |

---

## Quick Reference — Speed Budget

| Change | Speed impact |
|---|---|
| Double `population_size` | 2× slower |
| Double `generations` | 2× slower |
| Add one more trading pair | +20–40% slower |
| Enable `walk_forward` (10 windows) | ~10× slower |
| Enable `regime_aware` (3 regimes × 3 segments) | ~9× slower |
| Enable `parallel_evaluation` (5 workers) | ~5× faster |
| Reduce `max_per_strategy` from 5 to 3 | ~20–30% faster |
| Switch timeframe from `15m` → `1h` | ~4× faster |
| Disable `fitness_sharing` | ~5–10% faster |

**Rough runtime estimate (no parallelism)**:
```
runtime_minutes ≈ (population × generations × avg_backtest_sec) / 60
```
With walk-forward (N windows): multiply by N.
With parallel_evaluation (W workers): divide by W.
