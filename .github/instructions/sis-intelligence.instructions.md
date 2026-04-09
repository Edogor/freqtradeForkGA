---
applyTo: "genetic_algorithm/intelligence/**"
description: "SIS (Strategy Intelligence System) conventions. Use when: editing SIS modules, integrator hooks, corpus building, or predictor training code."
---

# SIS Development Conventions

**Pipeline order** (dependencies flow left-to-right):
`corpus.py` → `predictors.py` → `archetypes.py` → `temporal_analysis.py` → `pattern_mining.py` → `sis_integrator.py`

**5 integration hooks** in `sis_integrator.py`, called from `evolution.py`:
1. `filter_seeds()` — Remove predicted-bad initial strategies
2. `generate_immigrants()` — Inject archetype-diverse strategies
3. `get_indicator_bias()` — Weight proven indicators (CCI 26.5x, STOCH 12.5x lift)
4. `check_convergence()` — Detect stagnation, trigger diversification
5. `get_adaptive_weights()` — Shift fitness weights based on population state

**Key data files:**
- `genetic_algorithm/data/strategy_corpus.parquet` — 2900+ strategies, 65 features
- Rebuild trigger: after new HoF entries accumulate

**Testing:** `pytest tests/test_strategy_intelligence.py -v` (1000+ lines of coverage)

See [genetic_algorithm/intelligence/README.md](../../genetic_algorithm/intelligence/README.md) for full SIS docs.
