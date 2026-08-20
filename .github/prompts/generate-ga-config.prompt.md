---
description: "Generate a schema-V2 GA experiment proposal for the canonical planner."
mode: "agent"
tools: ["read_file", "create_file", "grep_search", "file_search", "semantic_search", "run_in_terminal"]
---

# Generate a GA Experiment Proposal

Create a schema-V2 config proposal, not a directly runnable legacy queue file.

## Evidence rules

1. Read only hash-verified V2 parent results and their analyzer decision.
2. Treat population, generations, pair count, mutation, elitism and tournament pressure as
   hypotheses. Do not call fixed ranges “proven” from historical aggregate rankings.
3. Change one declared factor per arm unless the plan explicitly declares a factorial experiment.
4. Use paired seeds, the same comparison panel and the same budget for control and treatment.
5. Profit never compensates for a failed drawdown, tail-risk, activity or sample-size gate.

## Mechanical constraints

- Use `config_schema_version: 2` and the enforced `safe_v2` preset.
- Keep elite, tournament and immigrant counts within the configured population.
- Use only implemented selector/crossover names.
- Keep pair lists non-empty and unique.
- Supply explicit positive `parallel_evaluation.num_workers` for evolution materialization.
- Do not enable features excluded by `safe_v2`.

Validate the proposal with:

```bash
python -m genetic_algorithm config validate <config.yaml>
```

The proposal must enter `ChildWavePlanV2` and the approval/materialization path. Do not write a
mutable file into `config/queue` as an execution instruction.
