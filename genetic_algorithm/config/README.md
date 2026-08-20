# GA Configuration Guide

## Audited V2 Baseline (Shadow Only)

For new repair/calibration work, resolve the conservative V2 baseline with:

```bash
python -m genetic_algorithm config show safe_v2
```

`safe_v2` is enforced: it uses the standard single-objective GA on 1h, disables the currently
untrusted Holdout/WF/MC/DSR/Surrogate/Regime/Pair-Validation/CPCV/SIS/Island/NSGA-II/LLM/Short paths,
turns hidden fee noise off, and marks the run as shadow-only and not automation-eligible. It does
not certify a strategy as safe. Its purpose is to provide one reproducible comparison baseline while
the V2 evaluator and runner are repaired.

The preset also enables strict V2 mark-to-market measurement and deterministic daily/trade block
bootstraps. The MTM order ledger supports closed long/short trades, leverage, multiple entries/DCA,
partial exits, and timestamped funding cashflows. Every trade and the final wallet must reconcile.
Open terminal positions without a terminal snapshot, aggregate-only historical funding, unsupported
trade semantics, or missing daily candles remain inconclusive; they are never replaced with
optimistic zero risk.

`evaluation_v2.annual_risk_free_rate` is an effective annual decimal rate (default `0.0`). The
versioned `calendar-effective-v1` contract converts it geometrically to 365 calendar-day periods,
uses sample volatility for Sharpe, full-sample target downside for Sortino, and geometric CAGR for
Calmar. The chosen annual and periodic rates are persisted with the result.

Trade expectancy follows `net-expectancy-clustered-v1`. It reports an equal-trade return and a
capital-weighted return based on the maximum committed stake plus/minus the entry fee. Both receive
deterministic lower confidence bounds. `evaluation_v2.expectancy_cluster_days` defines fixed UTC
close-time clusters (default one day), while `trade_bootstrap_block` is the moving-block length in
those clusters. Contemporaneous trades across pairs therefore remain together. The result also
records pair concentration and conservative serial/capital/temporal effective sample sizes.
Committed margin is intentionally not labelled stop-loss capital-at-risk.

`promotion_v2` remains disabled in the base preset. Enabling it requires an explicit
`required_scenarios` matrix containing every pair, timeframe, period, role, and cost multiplier.
This prevents an incomplete panel from silently becoming promotable. Current policy thresholds are
shadow-calibration defaults and can only produce `WOULD_PASS`, never execution authorization.

The current replay executor supports one manifest seed and deterministic spot costs (`fee` plus
fixed `slippage_pct`). It rejects dynamic slippage, separate spread/funding costs and short selling
until those paths have exact ledger-level provenance. Scenario periods are inclusive in the policy;
the runner converts them to Freqtrade's exclusive timerange end and verifies the reported coverage.

The preset expects the configured local data to exist (`auto_download_data: false`). Override pairs
or timeranges only in an experiment config that references `preset: safe_v2`; attempts to re-enable
one of the blocked features fail validation.

### Versioned validation contract

`config_schema_version: 2` is fail-closed. Unknown top-level keys, nested keys, list-item fields,
wrong container types, quoted booleans/numbers and deprecated paths stop CLI validation, queueing,
all three GA engines and V2 materialization through the same resolver. Error messages contain the
full sorted paths. Presets and programmatic overrides go through that same contract.

At immutable V2 boundaries the contract is stricter: the attempt-manifest builder accepts only
schema 2 configs containing every `DEFAULTS`-backed resolved field. Evolution and replay workers
revalidate the persisted config after integrity/hash verification, so a directly constructed or
internally rehashed artifact cannot bypass unknown-key, type, deprecation, or semantic checks.

Schema-v1 files remain loadable so historical experiments can still be inspected and reproduced;
undeclared or deprecated paths produce migration warnings. Use `config validate --strict` when a
legacy file must be warning-free. New automation should always inherit `safe_v2` and must never add
internal `_config_name`/`_island_name` fields to persisted user configuration.

The canonical runtime names are now `monte_carlo.num_permutations`,
`surrogate.min_training_samples`, percentage-valued `surrogate.filter_percentile`,
`generic_island_model.migration.count`, `regime_aware.method`, and
`genetic_algorithm.random_seed`. Their historical aliases are warnings in schema v1 and errors in
schema v2.

### Surrogate safety contract

Surrogate-assisted filtering is disabled by default and forbidden by enforced `safe_v2`. Outside
that baseline it remains experimental. It may influence search, but its estimates cannot enter the
Hall of Fame, final runner results, V2 evolution shortlist or candidate export without a successful
real backtest.

Activation requires a generation-out-of-sample validation gate. `min_training_samples` must be at
least 20, `min_validation_samples` at least 5, `validation_fraction` between 0.1 and 0.5 and
`min_validation_r2` between 0 and 1. Validation consists of complete newest GA generations while
training uses only earlier generations; random sibling splits are not accepted. The training target
is measured pre-sharing fitness. Restoring a checkpoint always requires rebuilding the model and
passing the gate again because model and training rows are deliberately not serialized.

These controls prevent estimates from masquerading as measured results. They do not demonstrate
that enabling the feature improves blind-OOS quality or compute efficiency; that still requires a
paired, equal-budget experiment against a surrogate-disabled control.

### Fitness-panel comparability

Every real GA evaluation now receives a deterministic `fitness_panel_id` and role. The identity
contains the declared pairs/timerange, costs, evaluator and fitness policy, plus the exact segment
set for regime-aware evaluation. Search seed, population size and generation budget are excluded
because they do not change the measurement itself.

Island-local fitness may only be ranked within its own panel. Generic-Island and Regime-Island
finalists are replayed on one `COMMON_REPLAY` panel before global ranking, Hall-of-Fame insertion or
runner export. If that replay cannot be completed, global results remain empty. Migration between
different panels transfers genomes for target-side reevaluation; it does not declare a winner from
incomparable raw scores.

A declared panel does not by itself prove that files behind a data path are unchanged. Therefore a
legacy Hall of Fame is run-local unless an explicit content-addressed data-manifest hash is part of
the panel. V2 replay/promotion already supplies that stronger code/data/attempt provenance.

## Quick Start

### Option 1: Use Example Config (Recommended for Real Data)

1. **Download market data:**
   ```bash
   freqtrade download-data --exchange binance --pairs BTC/USDT --timeframes 1h --days 90
   ```

2. **Check your data range:**
   ```bash
   freqtrade list-data --show-timerange
   ```

3. **Copy example config:**
   ```bash
   cp genetic_algorithm/config/ga_config_example.yaml my_config.yaml
   ```

4. **Edit `my_config.yaml`:**
   - Update `timerange` to match your data
   - Update `pairs` to match downloaded pairs

5. **Run GA:**
   ```bash
   python genetic_algorithm/run_ga.py --config my_config.yaml
   ```

### Option 2: Quick Test (Uses 2018 Test Data)

```bash
python genetic_algorithm/run_ga.py --config genetic_algorithm/config/ga_config_test.yaml
```

## Available Configs

| Config File | Purpose | Population | Generations | Data |
|-------------|---------|------------|-------------|------|
| `ga_config.yaml` | Default (test) | 100 | 50 | UNITTEST/BTC (2018) |
| `ga_config_example.yaml` | Template for real data | 20 | 10 | BTC/USDT (configurable) |
| `ga_config_test.yaml` | Quick test | 3 | 2 | UNITTEST/BTC (2018) |

## Configuration Sections

### Backtesting
```yaml
backtesting:
  timerange: "20250120-20250219"  # Date range YYYYMMDD-YYYYMMDD
  pairs: ["BTC/USDT"]              # Must have data for these
  stake_amount: 0.1                # Per-trade stake (in quote currency)
  max_open_trades: 1               # Concurrent trades
```

### Genetic Algorithm
```yaml
genetic_algorithm:
  population_size: 20   # Number of strategies per generation
  generations: 10       # Evolution iterations
  mutation_rate: 0.15   # Probability of mutation
  crossover_rate: 0.7   # Probability of crossover
```

## Troubleshooting

### "No data found" Error
```bash
# Check if you have data
freqtrade list-data

# If not, download it
freqtrade download-data --pairs BTC/USDT --timeframes 1h --days 90
```

### "Using 2018 test data" Warning
Your config has `UNITTEST/BTC` pairs. Update to real pairs:
```yaml
pairs:
  - "BTC/USDT"  # Instead of UNITTEST/BTC
```

### Empty Timerange Warning
Set a specific date range:
```yaml
timerange: "20250120-20250219"  # Instead of ""
```

## Command-Line Usage

```bash
# Use default config
python genetic_algorithm/run_ga.py

# Use custom config
python genetic_algorithm/run_ga.py --config my_config.yaml

# Validate config without running
python genetic_algorithm/run_ga.py --config my_config.yaml --validate-only

# Show all contract errors without launching anything
python -m genetic_algorithm config validate my_config.yaml
```
