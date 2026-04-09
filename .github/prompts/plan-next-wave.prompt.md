---
description: "Plan and launch the next wave of GA experiments. Use when: starting a new evolution wave, queuing experiments, or managing the automated pipeline. Covers config generation → queue → launch → monitoring."
mode: "agent"
tools: ["read_file", "create_file", "grep_search", "file_search", "run_in_terminal", "runSubagent"]
---

# Plan Next GA Wave

Orchestrate the next batch of automated evolution experiments.

## Workflow

### 1. Review Current State

```bash
# Check what's running
./ga_current.sh

# Check queue status
ls genetic_algorithm/config/queue/

# Check completed experiments
ls genetic_algorithm/config/done/ | tail -20

# Disk space
du -sh genetic_algorithm/data/ genetic_algorithm/cache/
```

### 2. Analyze Previous Wave

- Read latest results from `genetic_algorithm/data/` and `CONFIG_RANKING.md`
- Identify what worked (high fitness, passed walk-forward) and what failed
- Check SIS corpus freshness: `ls -la genetic_algorithm/data/strategy_corpus.parquet`

### 3. Generate Configs

Create 3-6 experiment configs targeting different strategies:
- 1-2 configs exploiting best-performing parameter regions
- 1-2 configs exploring novel indicator/timeframe combinations
- 1 SIS-enabled config leveraging intelligence feedback
- (Optional) 1 A/B comparison config

Place all configs in `genetic_algorithm/config/queue/`

### 4. Validate & Launch

```bash
# Validate all queued configs
for f in genetic_algorithm/config/queue/*.yaml; do
  python -c "import yaml; yaml.safe_load(open('$f'))" && echo "OK: $f" || echo "FAIL: $f"
done

# Launch queue daemon (if not already running)
./ga_auto_queue_v2.sh --wave <wave_name> --max 4 --persistent &

# Or launch specific experiments
python genetic_algorithm/run_ga.py --config genetic_algorithm/config/queue/<name>.yaml
```

### 5. Monitor

```bash
./ga_current.sh              # Live progress
./ga_monitor_v2.sh --live    # Detailed monitoring
```

## Naming Convention

`wave<N>_<variant>_<description>.yaml`

Examples:
- `wave35_sis_momentum.yaml` — SIS-enabled momentum strategy evolution
- `wave35_island_regime.yaml` — Island model with regime detection
- `wave35_baseline_control.yaml` — Control experiment for comparison
