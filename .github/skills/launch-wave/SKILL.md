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

Read these files to understand what worked and what failed:
- `CONFIG_RANKING.md` — ranked experiment list, top performers
- `KNOWN_ISSUES.md` — active anti-patterns to avoid
- `genetic_algorithm/data/hall_of_fame/hall_of_fame.json` — best strategies so far
- Latest `final_results.json` in `genetic_algorithm/data/`

Key questions:
- Which indicators dominated the HoF? (lens for SIS indicator bias)
- What population sizes avoided overfitting? (stay 10-15 for standard)
- Did walk-forward pass? (>30% degradation = overfitting)
- Which timeranges had most data coverage?

## Phase 2: Compose Wave (3-6 Configs)

A well-balanced wave mixes exploitation and exploration. Recommended composition:

| Slot | Purpose | Base Preset |
|------|---------|-------------|
| 1-2 | **Exploit** best regions from previous wave | `presets/standard.yaml` |
| 3-4 | **Explore** new indicator combos or timeframes | `presets/standard.yaml` |
| 5 | **SIS-enabled** production run | `presets/production.yaml` |
| 6 | **A/B comparison** (e.g., holdout on vs off) | Clone of slot 1-2 |

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

**Config placement:** `genetic_algorithm/config/queue/`

## Phase 3: Validate Before Queuing

```bash
# Validate all queued configs
for f in genetic_algorithm/config/queue/*.yaml; do
    python -c "import yaml; yaml.safe_load(open('$f'))" && echo "OK: $f" || echo "FAIL: $f"
done

# Check constraints (hook also runs automatically on file creation)
python .github/hooks/scripts/validate_ga_config.py < /dev/null
```

Common mistakes to check:
- `population_size > 15` for standard GA → overfitting
- Island + walk-forward combined → data conflict
- `tournament_size < 3` → random search
- `auto_download: false` without verified data → silent 0-trade runs

## Phase 4: Launch

```bash
# Preferred: queue daemon manages order automatically
./ga_auto_queue_v2.sh --wave <wave_number> --max 3 --persistent

# Or launch a single config directly
source .venv/bin/activate
python genetic_algorithm/run_ga.py --config genetic_algorithm/config/queue/<name>.yaml
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

## Critical Constraints (Never Violate)

| Rule | Source |
|------|--------|
| Standard GA: `population_size` 10-15 | >15 → 59-65% overfitting |
| Island model: ≥60 per island | <60 → 62-100% overfitting |
| Never: island + walk-forward | Data partitioning conflict |
| Always: `enable_cache: true` | 2-5x speedup |
| `tournament_size` must be 3-6 | <3 = random, >6 = premature convergence |

See [GA_CONFIG_CHEATSHEET.md](../../GA_CONFIG_CHEATSHEET.md) for all parameter ranges.
See [KNOWN_ISSUES.md](../../KNOWN_ISSUES.md) for active anti-patterns.
