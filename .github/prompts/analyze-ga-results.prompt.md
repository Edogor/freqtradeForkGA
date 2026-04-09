---
description: "Analyze GA experiment results, diagnose issues, and recommend next steps. Use when: reviewing backtest results, diagnosing overfitting, comparing wave performance, or planning the next evolution wave."
mode: "agent"
tools: ["read_file", "grep_search", "file_search", "semantic_search", "run_in_terminal", "runSubagent"]
---

# Analyze GA Results

Review completed experiment results and provide actionable recommendations.

## Analysis Steps

1. **Gather results**: Read `genetic_algorithm/data/` for recent run outputs, `final_results.json` files, and hall of fame entries
2. **Check for overfitting signals**:
   - Walk-forward degradation > 30% from in-sample → overfitting
   - Monte Carlo p-value > 0.05 → likely curve-fitted
   - Very high in-sample profit but few trades → suspicious
3. **Compare across experiments**: Use `CONFIG_RANKING.md` and `EXPLORATION_LOG.md`
4. **Identify patterns**: Which indicators, timeframes, and parameter ranges produce robust strategies?

## Diagnosis Checklist

- [ ] Population size within safe range?
- [ ] Sufficient generations for convergence?
- [ ] Walk-forward validation applied?
- [ ] SIS enabled and corpus fresh?
- [ ] Cache hit rate acceptable?
- [ ] Any 0-trade silent failures?

## Output Format

Provide:
1. **Summary**: Key metrics from the analyzed experiments
2. **Issues found**: Any overfitting, convergence problems, or configuration errors
3. **Recommendations**: Specific config changes for the next wave
4. **Config suggestion**: If appropriate, draft a follow-up config addressing found issues
