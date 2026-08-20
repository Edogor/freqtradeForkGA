# Genetic Algorithm Documentation Index

This folder contains all documentation for the FreqTrade Genetic Algorithm Strategy Optimizer.

> **Current audit baseline (2026-07-20):** The older feature pages describe intended or historical
> implementations; their "complete" labels are not production-readiness claims. Use the following
> documents as the current source of truth before changing evaluation or orchestration:
>
> - [GA_AUDIT_2026-07-20.md](GA_AUDIT_2026-07-20.md) — verified architecture, defects and evidence
> - [GA_EVALUATION_SPEC.md](GA_EVALUATION_SPEC.md) — target evaluation and promotion contract
> - [NEXT_WAVE_AUTOMATION_PLAN.md](NEXT_WAVE_AUTOMATION_PLAN.md) — guarded automation design
> - [AUTOMATION_RUNBOOK_V2.md](AUTOMATION_RUNBOOK_V2.md) — start, restart, status and kill switch
> - [WEEK_RUN_READINESS_V2.md](WEEK_RUN_READINESS_V2.md) — go/no-go checklist for the first week run
> - [GA_TODO.md](GA_TODO.md) — prioritized implementation backlog
> - [GENOME_MIGRATION_GUIDE.md](GENOME_MIGRATION_GUIDE.md) — fail-closed migration of legacy seeds

## 📂 Directory Structure

```
docs/
├── INDEX.md                  # This file
├── GA_AUDIT_2026-07-20.md    # Current verified audit baseline
├── GA_EVALUATION_SPEC.md     # Target metric and promotion semantics
├── NEXT_WAVE_AUTOMATION_PLAN.md
├── AUTOMATION_RUNBOOK_V2.md
├── WEEK_RUN_READINESS_V2.md
├── GENOME_MIGRATION_GUIDE.md
├── GA_TODO.md                # Prioritized repair and feature backlog
├── features/                 # Feature documentation
├── plots/                    # Generated visualization plots
└── troubleshooting/          # Bug fixes and debugging guides
```

---

## 📖 Features Documentation (`features/`)

### Core Features

| Document | Description | Status |
|----------|-------------|--------|
| [CONFIG_REFERENCE.md](features/CONFIG_REFERENCE.md) | Complete configuration reference for ga_config.yaml | ✅ Complete |
| [VISUALIZATION_GUIDE.md](features/VISUALIZATION_GUIDE.md) | How to visualize evolution progress and results | ✅ Complete |
| [MAX_OPEN_TRADES_FEATURE.md](features/MAX_OPEN_TRADES_FEATURE.md) | Per-strategy max_open_trades evolution | ✅ Complete |

### Walk-Forward Optimization

| Document | Description | Status |
|----------|-------------|--------|
| [WALK_FORWARD_GUIDE.md](features/WALK_FORWARD_GUIDE.md) | User guide for walk-forward optimization | ✅ Complete |

### Parallel Evaluation

| Document | Description | Status |
|----------|-------------|--------|
| [PARALLEL_EVALUATION_GUIDE.md](features/PARALLEL_EVALUATION_GUIDE.md) | Multi-process parallel backtesting | ✅ Complete |

### Market Regime Detection

| Document | Description | Status |
|----------|-------------|--------|
| [MARKET_REGIME_DATASET_SELECTION.md](features/MARKET_REGIME_DATASET_SELECTION.md) | Concepts and design for regime-aware evaluation | ✅ Complete |
| [REGIME_DETECTION_IMPLEMENTATION.md](features/REGIME_DETECTION_IMPLEMENTATION.md) | Implementation details and methods | ✅ Complete |

### Tier 3: Robustness & Anti-Overfitting

| Document | Description | Status |
|----------|-------------|--------|
| [TIER3_ROBUSTNESS_FEATURES.md](features/TIER3_ROBUSTNESS_FEATURES.md) | Monte-Carlo, Parsimony, Pareto Archive, Dynamic Bounds | ✅ Complete |

### LLM Strategy Designer (Phase 1A)

| Document | Description | Status |
|----------|-------------|--------|
| [LLM_STRATEGY_DESIGNER_GUIDE.md](features/LLM_STRATEGY_DESIGNER_GUIDE.md) | How to use LLMs to seed and diversify the GA population | ✅ Complete |

### Island Model & Parallel Evaluation

| Document | Description | Status |
|----------|-------------|--------|
| [GENERIC_ISLAND_MODEL.md](features/GENERIC_ISLAND_MODEL.md) | Generic Island Model — multi-population evolution with shared process pool | ✅ Complete |
| [CONFIG_VALIDATOR_GUIDE.md](features/CONFIG_VALIDATOR_GUIDE.md) | Versioned mechanical config contract and evidence rules for tuning hypotheses | ✅ Complete |

---

## 📋 Current Audit & Roadmap

| Document | Description |
|----------|-------------|
| [GA_AUDIT_2026-07-20.md](GA_AUDIT_2026-07-20.md) | Evidence-backed audit and current go/no-go decision |
| [GA_EVALUATION_SPEC.md](GA_EVALUATION_SPEC.md) | Proposed equity-, risk- and confidence-aware evaluation model |
| [NEXT_WAVE_AUTOMATION_PLAN.md](NEXT_WAVE_AUTOMATION_PLAN.md) | Transactional lifecycle and staged next-wave automation |
| [AUTOMATION_RUNBOOK_V2.md](AUTOMATION_RUNBOOK_V2.md) | Operational guarded-automation runbook |
| [WEEK_RUN_READINESS_V2.md](WEEK_RUN_READINESS_V2.md) | Evidence and remaining gates for first week run |
| [GA_TODO.md](GA_TODO.md) | Prioritized issues, acceptance criteria and work order |
| [GENOME_MIGRATION_GUIDE.md](GENOME_MIGRATION_GUIDE.md) | Reproducible legacy-genome migration and warm-start binding |

---

## 🔧 Troubleshooting (`troubleshooting/`)

| Document | Description |
|----------|-------------|
| [TROUBLESHOOTING.md](troubleshooting/TROUBLESHOOTING.md) | Consolidated troubleshooting guide |

---

## 📊 Historical Feature Documentation Status

The table below says whether a feature was implemented/documented historically. It does **not** mean
that the current end-to-end behavior is correct or approved for automated decisions. The current
runtime assessment is in [GA_AUDIT_2026-07-20.md](GA_AUDIT_2026-07-20.md#robustheitsfeatures-aktueller-nutzwert).

| Feature | Current audit status | Notes |
|---------|----------------------|-------|
| Walk-Forward Optimization | Redesign | Implemented as rolling fixed-gene validation, not re-optimization |
| Multi-Timeframe Strategies | Experimental | Implementation exists; effectiveness not established |
| NSGA-II Multiobjective | Disabled until fix | Offspring lifecycle currently freezes evolution after generation 0 |
| Parallel Evaluation | Repair required | Semantics differ from sequential evaluation |
| Market Regime Detection | Mechanically repaired, experimental | Coverage/aggregation fail closed; detector quality and OOS value remain unverified |
| **Regime Detection Accuracy** | Unverified | Historical phase label is not an OOS effectiveness proof |
| Elite Fitness Caching | P0 repair | Disk serialization loses fitness-relevant fields |
| **Monte-Carlo Robustness** | Remove from fitness | Current trade-shuffle profit test is not informative |
| **Parsimony Pressure** | Implemented, unproven | Requires controlled OOS ablation |
| **Pareto Archive** | Blocked | Depends on repaired NSGA-II and comparable replay metrics |
| **Dynamic Bounds** | Implemented, unproven | Requires controlled OOS ablation |
| Island Model | Active, repair required | Cross-island raw fitness/result provenance is not comparable |
| **LLM Strategy Designer** | Experimental | Provider integration tests are not cleanly collected by Pytest |

---

## 🔧 Configuration Files

These are existing examples, not audited production recommendations. Always complete the relevant
fail-closed validation work in [GA_TODO.md](GA_TODO.md) before using them for a new experiment.

| Config | Historical purpose |
|--------|--------------------|
| [fast_evolution_demo.yaml](../config/fast_evolution_demo.yaml) | Small demonstration config |
| [ga_config.yaml](../config/ga_config.yaml) | General example/default config |
| [ga_config_island.yaml](../config/ga_config_island.yaml) | Island-model example |

---

*Audit index updated: July 20, 2026. Historical feature pages were last broadly updated in April 2026.*
