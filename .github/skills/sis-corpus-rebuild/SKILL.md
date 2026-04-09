---
name: sis-corpus-rebuild
description: "Rebuild the SIS strategy corpus from latest Hall of Fame data, retrain predictors, and validate the pipeline. Use when: SIS corpus is stale, new experiments completed, predictor accuracy degraded, or preparing for a new evolution wave."
---

# SIS Corpus Rebuild

Rebuild the full Strategy Intelligence System pipeline from accumulated experiment data.

## When to Use

- After completing a wave of experiments (new HoF entries available)
- When SIS predictor accuracy has degraded
- Before launching SIS-enabled evolution waves
- When `strategy_corpus.parquet` is older than latest results

## Prerequisites Check

Before rebuilding, verify data availability:

```bash
# Check corpus freshness
ls -la genetic_algorithm/data/strategy_corpus.parquet

# Count hall of fame entries
ls genetic_algorithm/data/hall_of_fame/*.json 2>/dev/null | wc -l

# Check result directories
ls -d genetic_algorithm/data/*/final_results.json 2>/dev/null | wc -l

# Verify no experiments are actively writing results
./ga_current.sh
```

## Rebuild Procedure

### Phase 1: Build Corpus

```bash
python -m genetic_algorithm.intelligence.run_sis corpus --verbose
```

This scans `genetic_algorithm/data/hall_of_fame/` and result directories, extracts 65 surrogate features per strategy, and writes `strategy_corpus.parquet`.

Expected output: strategy count, feature count, wave distribution, source distribution.

### Phase 2: Train Predictors

```bash
python -m genetic_algorithm.intelligence.run_sis predict --verbose
```

Trains LightGBM models for win_rate, profit, and drawdown prediction. Reports cross-validation scores.

### Phase 3: Discover Archetypes

```bash
python -m genetic_algorithm.intelligence.run_sis cluster --verbose
```

Runs HDBSCAN clustering to identify strategy archetypes (momentum, mean-reversion, etc.).

### Phase 4: Temporal Analysis

```bash
python -m genetic_algorithm.intelligence.run_sis analyze --verbose
```

Identifies temporal weaknesses and complementary strategy pairs.

### Phase 5: Pattern Mining

```bash
python -m genetic_algorithm.intelligence.run_sis patterns --verbose
```

Mines indicator enrichment patterns (e.g., CCI 26.5x lift, STOCH 12.5x lift).

### Full Pipeline (All Phases)

```bash
python -m genetic_algorithm.intelligence.run_sis all --verbose
```

Runs all 5 phases sequentially. Takes 2-5 minutes depending on corpus size.

## Validation

After rebuild, verify the pipeline outputs:

```bash
# Check corpus size
python -c "
import pandas as pd
df = pd.read_parquet('genetic_algorithm/data/strategy_corpus.parquet')
print(f'Corpus: {len(df)} strategies, {len(df.columns)} features')
print(f'Waves: {sorted(df[\"wave\"].dropna().unique())}')
print(f'Fitness range: {df[\"fitness\"].min():.3f} - {df[\"fitness\"].max():.3f}')
"

# Run SIS tests
pytest tests/test_strategy_intelligence.py -v --tb=short
```

## Troubleshooting

- **Empty corpus**: Check that `hall_of_fame/` has JSON files and they contain valid strategy data
- **Low feature count**: Some enrichment data may be missing — run with `--skip-enrichment` to proceed without it
- **Predictor warnings**: Low sample count in some clusters is normal for rare archetypes
- **Test failures after rebuild**: Verify the corpus schema hasn't changed (65 expected features)
