---
description: "Plan next steps for the GA system: review recent results, identify improvements, and create an actionable plan. Use when: deciding what to work on next for the GA infrastructure, SIS pipeline, or evolution workflow."
mode: "agent"
tools: ["read_file", "grep_search", "file_search", "semantic_search", "run_in_terminal", "runSubagent"]
---

# Plan GA Next Steps

Review the current state of the GA system and create an actionable improvement plan.

## Analysis Process

1. **Check recent results**: Read `CONFIG_RANKING.md`, `EXPLORATION_LOG.md`, and latest `final_results.json` files
2. **Review known issues**: Read `KNOWN_ISSUES.md` and `GA_FIXES_AND_IMPROVEMENTS.md`
3. **Check infrastructure status**: Review `INFRASTRUCTURE_REDESIGN.md` for pending items
4. **Assess SIS health**: Check corpus freshness, predictor accuracy, archetype coverage
5. **Review test coverage**: Run `pytest genetic_algorithm/tests/ -q` and check for gaps

## Output Format

Provide a prioritized list of next steps:
1. **Critical fixes** — Bugs or issues blocking experiments
2. **High-value improvements** — Changes that improve evolution quality or reduce overfitting
3. **Infrastructure** — Orchestration, monitoring, lifecycle management improvements
4. **Exploration** — New config strategies, indicator combinations, or parameter regions to try