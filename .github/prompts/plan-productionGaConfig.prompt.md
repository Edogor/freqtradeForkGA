---
description: "Plan a controlled GA configuration experiment for the V2 wave planner."
mode: "agent"
tools: ["read_file", "create_file", "grep_search", "file_search", "semantic_search", "run_in_terminal"]
---

# Plan a Production GA Config Experiment

Production suitability is decided by the V2 replay and promotion contract, not by a fixed
population size or a legacy aggregate fitness score.

Build a control plus one or more explicitly declared treatment arms:

- Keep data, scenario panel, costs, seeds and compute budget paired.
- Change one factor at a time unless a factorial design is declared.
- Treat population, generations, mutation, selection pressure, pair count and optional features as
  hypotheses whose effect must be measured out of sample.
- Require non-compensating return-LCB, expectancy-LCB, drawdown-UCB, ES-UCB, activity and effective
  sample-size gates.
- Keep automatic execution on `safe_v2`; experimental features need a shadow arm and evidence
  before eligibility changes.

Validate each resolved config with:

```bash
python -m genetic_algorithm config validate <config.yaml>
```

Output a `ChildWavePlanV2` proposal for review and materialization. Do not bypass approval by
placing mutable YAML directly in the legacy queue.
