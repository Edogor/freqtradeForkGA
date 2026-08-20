---
name: launch-wave
description: "Plan and launch a new wave of GA experiments: review previous results, generate 3-6 configs targeting different strategies, validate, queue, and monitor. Use when: starting a new evolution wave, planning a batch of experiments, or managing the automated GA pipeline."
argument-hint: "Wave number and focus (e.g. 'wave35 SIS momentum' or 'wave35 fast profit')"
---

# Launch Wave

Orchestrate a complete wave of GA experiments from planning through live monitoring.

## When to Use

- Starting a new numbered wave of experiments
- Planning a targeted batch (e.g., SIS comparison, indicator exploration, production run)
- After completing a wave and wanting to build on its lessons

## Prerequisites

Before generating configs, gather current state:

```bash
# What's running?
./ga_current.sh

# What's queued?
ls genetic_algorithm/config/queue/

# Recent completions
ls genetic_algorithm/config/done/ | tail -10

# SIS corpus freshness
ls -la genetic_algorithm/data/strategy_corpus.parquet

# Disk space available
du -sh genetic_algorithm/data/ genetic_algorithm/cache/
df -h .
```

If the SIS corpus is older than the most recent wave's results, rebuild it first:
```bash
python -m genetic_algorithm.intelligence.run_sis all --verbose
```

## Phase 1: Review Previous Wave

Read the reconciled parent `WaveResultSnapshotV2`, its matching immutable analysis decision and
the hash-verified candidate/scenario artifacts. Legacy rankings, logs and raw HOF metrics are
diagnostic only.

Key questions:
- Which candidates are eligible on the same common replay panel?
- Which population/search-budget hypothesis should be tested against an unchanged control?
- Did walk-forward pass? (>30% degradation = overfitting)
- Which timeranges had most data coverage?

## Phase 2: Compose a controlled wave proposal

A controlled wave contains an unchanged control and only hypotheses that the current planner can
express:

| Slot | Purpose | Base Preset |
|------|---------|-------------|
| 1-2 | **Exploit** best regions from previous wave | `presets/standard.yaml` |
| 3-4 | **Explore** new indicator combos or timeframes | `presets/standard.yaml` |
| 5 | **Optional-feature shadow arm** | Only when an explicit worker supports it |
| 6 | **Paired comparison** | One declared factor versus an unchanged control |

Available presets in `genetic_algorithm/config/presets/`:
- `standard.yaml` — fast discovery, single-pop, ~30-60 min
- `production.yaml` — island model, walk-forward, ~4-8 hrs
- `quick_test.yaml` — smoke test, 2-3 min
- `island.yaml` — island model baseline
- `nsga2.yaml` — multi-objective Pareto

**Config naming convention:** `<NN>_E<experiment_number>_<description>.yaml`
- `NN` = sort order within wave (00, 01, 02 …)
- Use wave-scoped experiment numbers (e.g., E177, E178 … for wave 35)
- Keep description short, snake_case

Configs are embedded in `ChildWavePlanV2` and frozen into attempt roots during materialization.
They are not placed in the mutable legacy queue.

## Phase 3: Validate Before Queuing

```bash
# Validate all proposed configs through the runtime contract
for f in genetic_algorithm/config/proposals/*.yaml; do
    python -m genetic_algorithm config validate "$f"
done
```

Common mistakes to check:
- elite/tournament/immigrant counts outside the configured population
- classic Island + walk-forward, which would otherwise ignore the requested validation
- duplicated pairs or incomplete data-manifest coverage
- unpaired seeds, changed panels or multiple undeclared factors

## Phase 4: Launch

```bash
# Materialize only an approved ChildWavePlanV2 through
# queue_approved_materialization(), then let AttemptSchedulerV2 execute it.
```

## Phase 5: Monitor

```bash
# Rich live dashboard (recommended)
./ga_current.sh

# Live log following for active experiment
tail -f genetic_algorithm/logs/<experiment>.log

# Per-generation fitness snapshot
grep "\[STATS\]" genetic_algorithm/logs/<experiment>.log

# Breakthrough tracking
grep "\[NEW BEST\]" genetic_algorithm/logs/<experiment>.log

# Holdout overfitting signals
grep "\[HOLDOUT" genetic_algorithm/logs/<experiment>.log
```

## Phase 6: Archive Completed Configs

After a wave completes, move configs from `queue/` to `done/<wave_name>/`:
```bash
mkdir -p genetic_algorithm/config/done/wave<N>_<description>
mv genetic_algorithm/config/queue/<wave_configs> genetic_algorithm/config/done/wave<N>_<description>/
```

Then update `CONFIG_RANKING.md` with the new results.

## Binding constraints

- The config must pass the canonical schema and versioned mechanical invariants.
- Every treatment must have a same-panel, same-budget, paired-seed control.
- Final-test cells may be exposed only through the usage ledger.
- Approval, materialization and queue hashes must reconcile before execution.
- Fixed “optimal” population, island, pair or tournament ranges are not binding constraints; they
  are hypotheses requiring V2 evidence.
