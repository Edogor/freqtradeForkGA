# Strategy Intelligence System (SIS)

Analyzes the entire historical GA strategy corpus (~2900+ strategies across waves 1–30) to extract patterns, predict outcomes, and discover winning archetypes.

## Quickstart

```bash
# Build corpus from all historical HoF + generation snapshots
python -m genetic_algorithm.intelligence corpus

# Run the full pipeline (all phases, ~6 seconds)
python -m genetic_algorithm.intelligence all

# Individual phases
python -m genetic_algorithm.intelligence predict
python -m genetic_algorithm.intelligence cluster
python -m genetic_algorithm.intelligence analyze
python -m genetic_algorithm.intelligence patterns
```

## Architecture

```
genetic_algorithm/intelligence/
├── corpus.py          — Corpus builder: scans HoF + gen snapshots → strategy_corpus.parquet
├── predictors.py      — LightGBM multi-target regressors (fitness, profit, drawdown, …)
├── archetypes.py      — HDBSCAN clustering → named strategy archetypes
├── temporal_analysis.py — Weakness detection + complementary pair discovery
├── pattern_mining.py  — Indicator enrichment, synergies, operator patterns, risk params
└── run_sis.py         — CLI entry point (subcommands above)

genetic_algorithm/data/
├── strategy_corpus.parquet          — Raw corpus (2900+ strategies, 115 features)
└── strategy_corpus_clustered.parquet — Corpus enriched with archetype labels

genetic_algorithm/ml/models/
├── predictor_meta.json              — Model metadata + feature columns
└── predictor_<target>.pkl           — Per-target LightGBM models
```

## Phases

### Phase 1 — Corpus (`corpus.py`)

Scans all `hall_of_fame.json` files and `gen_*.json` snapshots. Extracts:
- **65 surrogate features** via `surrogate.extract_features()` (indicator presence, operator counts, threshold stats, ROI/stoploss params)
- **Metrics**: fitness, profit, drawdown, sharpe, trades, win rate, generalization ratio
- **Behavioral features**: monthly profit stats, per-pair profit stats (when available in HoF)
- **Meta**: wave, timeframe, run_id, origin (hall_of_fame vs gen_snapshot)

Deduplication by `strategy_fingerprint`. Output: 560 KB Parquet.

### Phase 2 — Predictors (`predictors.py`)

Walk-forward validation: trains on waves 1–28, tests on waves 29–30.

| Target | R² | Notes |
|--------|-----|-------|
| win_rate | 0.47 | Best predictor — stoploss + roi_min drive it |
| profit | −5.0 | Distribution shift across waves |
| Others | <0 | Each wave explores different regions |

Top features across all models: `stoploss`, `roi_min`, `roi_max`, `threshold_mean`, `period_mean`.

**Note**: Negative R² is expected and informative — the GA explores different strategy regions each wave, meaning later waves are out-of-distribution relative to earlier training data.

### Phase 3 — Archetypes (`archetypes.py`)

HDBSCAN clustering on 65 surrogate features produces named archetypes:

| Archetype | Fitness | Key Indicators | Timeframe |
|-----------|---------|----------------|-----------|
| AROON+ROC | 0.737 | ROC, AROON | 1h |
| DONCHIAN+ROC | 0.682 | ROC, DONCHIAN | 1h |
| VWAP+ROC | 0.581 | ROC, VWAP | 1h |
| CDL_DOJI+ROC | 0.653 | ROC, CDL_DOJI | 1h |
| RSI+MACD | 0.470 | RSI, MACD | 15m (low freq) |

The 1h timeframe consistently produces the highest-fitness archetypes.

### Phase 4 — Temporal Analysis (`temporal_analysis.py`)

Identifies strategy weaknesses from monthly/per-pair breakdowns (HIGH_VARIANCE, DEEP_LOSS_MONTH, WEAK_PAIR, HIGH_DRAWDOWN, etc.) and finds complementary strategy pairs.

**Current limitation**: HoF entries don't store per-month/per-pair profit series — these columns will be empty until full backtest results are stored in the corpus.

### Phase 5 — Pattern Mining (`pattern_mining.py`)

Compares top 20% vs bottom 20% strategies by fitness.

**Indicator enrichment** (top 5):
| Indicator | Top% | Bot% | Enrichment |
|-----------|------|------|-----------|
| CCI | 27.9% | 5.5% | **4.46x** |
| ADX | 37.3% | 10.1% | 3.45x |
| BBANDS | 31.7% | 8.9% | 3.30x |
| STOCH | 20.7% | 6.5% | 2.89x |
| DONCHIAN | 15.1% | 5.7% | 2.42x |

**Indicator synergies** (lift in top strategies):
| Pair | Lift | Fitness |
|------|------|---------|
| DONCHIAN + ROC | **21.5x** | 0.638 |
| BBANDS + ADX | 2.0x | 0.505 |
| ADX + STOCH | 1.9x | 0.507 |

**Risk parameters** (top vs bottom strategies):
| Parameter | Top | Bottom | Takeaway |
|-----------|-----|--------|---------|
| stoploss | −7.0% | −3.7% | Wider stoploss wins |
| roi_max | 3.6% | 1.8% | Higher ROI ceiling |
| n_indicators | 2.7 | 3.2 | Fewer indicators = better |
| max_open_trades | 2.1 | 2.7 | Fewer concurrent trades |

**Operator patterns**:
- `cross_below` and `<`: favored in top strategies (↑)
- `cross_above`: avoided in top strategies (↓)

## Key Takeaways for GA Configuration

1. **Prioritize 1h timeframe** — all high-fitness archetypes (>0.58) are on 1h
2. **DONCHIAN+ROC combo** — the single most predictive indicator pair (21x synergy lift)  
3. **CCI is underutilized** — only in 9% of strategies globally but 4.5x enriched in top ones
4. **Avoid MACD, ICHIMOKU, CDL_ENGULFING** — enriched in bottom strategies
5. **Wider stoploss (-7%)** and fewer indicators (2–3) characterize top strategies
6. **`cross_below`/`<` operators** are statistically better than `cross_above`

## CLI Reference

```bash
# Options
python -m genetic_algorithm.intelligence -v all          # verbose logging
python -m genetic_algorithm.intelligence predict --test-waves wave28 wave29
python -m genetic_algorithm.intelligence cluster --min-cluster-size 15 --min-samples 5
python -m genetic_algorithm.intelligence analyze --top-n 30
```
