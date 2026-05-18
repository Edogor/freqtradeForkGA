---
description: "Create a production-quality GA config for a serious evolution run. Use when: preparing configs for a production wave with SIS integration, validated parameters, and robust settings."
mode: "agent"
tools: ["read_file", "create_file", "grep_search", "file_search", "semantic_search", "run_in_terminal", "runSubagent"]
---

# Production GA Config

Generate a battle-tested experiment config suitable for production evolution.

## Requirements

Production configs must:
- Use population size 10-15 (standard) or ≥60 per island (island model)
- Enable cache (`enable_cache: true`)
- Enable walk-forward validation (post-hoc)
- Enable SIS if corpus is fresh (`sis.enabled: true`)
- Target 25+ generations for convergence
- Use proven fitness weights from top-performing experiments

## Context Gathering

1. Read `CONFIG_RANKING.md` for the best-performing configs and their parameters
2. Read `GA_CONFIG_CHEATSHEET.md` for valid parameter ranges
3. Check `genetic_algorithm/data/strategy_corpus.parquet` freshness for SIS readiness
4. Review the latest wave's configs in `genetic_algorithm/config/done/` for patterns

## Output

Write the config YAML to `genetic_algorithm/config/queue/` with a descriptive filename (e.g., `E<N>_<description>.yaml`). Include comments explaining non-obvious parameter choices.