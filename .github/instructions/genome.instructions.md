---
applyTo: "genetic_algorithm/genome/**"
description: "GA genome conventions. Use when: editing gene definitions, codegen templates, indicator parameters, operator validation, or strategy seed generation."
---

# GA Genome Conventions

**Module layout:**
- `gene.py` — Dataclass definitions: `StrategyGene`, `IndicatorGene`, `ConditionGene`, `RegimeGene`
- `codegen.py` — `StrategyGenerator`: random strategy creation + gene-to-IStrategy code generation
- `indicators.py` — Indicator parameter registry and `create_random_indicator()` factory
- `seeds.py` — Seed strategy definitions (HoF injection, SIS-seeded, hand-crafted priors)
- `individual.py` — Individual wrapper (gene + fitness + metadata)
- `ensemble.py` — Multi-strategy ensemble gene representation

> Note: `genetic_algorithm/core/strategy_gene.py` is a backward-compat shim → real dataclasses live here in `genome/gene.py`.

## Gene Dataclasses

**`IndicatorGene`** — one technical indicator:
- `type: str` — indicator name (e.g., `'RSI'`, `'MACD'`, `'BBANDS'`)
- `parameters: Dict[str, Any]` — indicator-specific params (period, etc.)
- `timeframe: Optional[str]` — `None` = base timeframe; non-None = informative (higher TF only)
- `instance_id: Optional[str]` — unique ID when multiple instances of same type (e.g., `'RSI_0'`, `'RSI_1'`)
- `param_bounds: Optional[Dict]` — evolved [min, max] per parameter (for adaptive mutation)

**`ConditionGene`** — one buy/sell logic condition:
- References an `IndicatorGene` by `indicator_id` and `timeframe`
- Uses one of 8 supported operators (see below)

**`RegimeGene`** — optional in-strategy regime filtering:
- When `enabled=True`, codegen injects ADX/DI regime detection into `populate_indicators()`
- Uses `merge_informative_pair` for higher-TF regime scores
- `entry_trend_min/max`: filter entries by composite trend score [-1, 1]

**`StrategyGene`** — full strategy genome:
- Serialized via `to_dict()` / deserialized via `from_dict()`
- `strategy_fingerprint()` returns a stable hash for caching
- Call `assign_instance_ids()` after any indicator list mutation
- Call `prune_orphaned_conditions()` after removing indicators
- `_enforce_roi_monotonicity()` runs automatically in `__post_init__`

## 8 Supported Operators

All conditions must use one of these (defined in `indicators.py` operator registry):

| Operator | Meaning |
|----------|---------|
| `<` | value below threshold |
| `>` | value above threshold |
| `cross_above` | value crosses above reference |
| `cross_below` | value crosses below reference |
| `increasing` | value increasing over lookback |
| `decreasing` | value decreasing over lookback |
| `between` | value between threshold and threshold_upper |
| `value_above_ago` | value above its own value N bars ago |

Always use `is_valid_operator()` from the operator registry before generating/mutating conditions — it enforces which operators are valid per indicator type.

## Codegen Rules

`StrategyGenerator.generate_strategy_code()` produces a valid `IStrategy` subclass:
- Class name placeholder is replaced by callers (do not hardcode)
- Informative pairs declared in `informative_pairs()` only when multi-timeframe indicators present
- Regime-aware strategies get ADX/DI computed in `populate_indicators()`
- If codegen fails, `_generate_fallback_strategy()` returns a minimal valid strategy

**Adding a new indicator type:**
1. Add parameter bounds to the indicator registry in `indicators.py`
2. Define valid operators for it in the operator registry
3. Add any special codegen handling in `codegen.py` if the indicator needs custom pandas-ta calls
4. Update `indicators.py` `create_random_indicator()` to sample the new type

## Timeframe Constraints

- Informative indicators must be on a **strictly higher** timeframe than the base: use `is_higher_timeframe(candidate, base)` for any validation
- Valid timeframes: `1m 3m 5m 15m 30m 1h 2h 4h 6h 8h 12h 1d 3d 1w 1M`
- Never assign an informative indicator the same or lower timeframe as the base pair
