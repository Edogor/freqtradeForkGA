---
description: "Generate a GA experiment YAML config for the automated evolution queue. Use when: creating new experiment configs, planning waves, designing SIS-enabled runs, or setting up A/B comparisons."
mode: "agent"
tools: ["read_file", "create_file", "grep_search", "file_search", "semantic_search", "run_in_terminal", "runSubagent"]
---

# Generate GA Experiment Config

Create a production-ready YAML experiment config for the GA queue.

## Context Gathering

1. Read the latest results from `genetic_algorithm/data/hall_of_fame/` and `CONFIG_RANKING.md` to understand what has worked
2. Check `KNOWN_ISSUES.md` and `GA_CONFIG_CHEATSHEET.md` for parameter constraints
3. Review recent configs in `genetic_algorithm/config/done/` for the latest wave patterns
4. If SIS is enabled, check `genetic_algorithm/intelligence/README.md` for integration requirements

## Config Generation Rules

**Mandatory constraints (never violate):**
- Standard GA: `population_size` 10-15
- Island model: ≥ 60 per island (total pop = islands × per-island)
- Never combine island model + walk-forward
- `tournament_size` ≥ 3
- `elite_size` ≈ 10% of population
- Always set `enable_cache: true`
- Always set `auto_download: false` only if data is verified present

**SIS configuration:**
- Enable with `sis.enabled: true` for production runs
- SIS requires `genetic_algorithm/data/strategy_corpus.parquet` to exist
- All 5 hooks are active by default when enabled

**Fitness weights (proven ranges):**
- `profit_weight`: 0.3-0.5
- `win_rate_weight`: 0.2-0.3
- `drawdown_weight`: 0.15-0.25
- `trade_count_weight`: 0.05-0.15

## Output

Place the generated config at: `genetic_algorithm/config/queue/<name>.yaml`
Follow naming convention: `<wave>_<variant>_<description>.yaml` (e.g., `wave35_sis_momentum.yaml`)

## Validation

After generating, validate the config:
```bash
python -c "import yaml; yaml.safe_load(open('genetic_algorithm/config/queue/<name>.yaml'))"
```

Confirm no constraint violations from the rules above.
