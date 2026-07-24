---
description: "Plan the next hash-bound GA V2 wave from verified parent evidence."
mode: "agent"
tools: ["read_file", "create_file", "grep_search", "file_search", "run_in_terminal"]
---

# Plan the Next GA V2 Wave

1. Load a reconciled parent `WaveResultSnapshotV2` and its single matching immutable
   `WaveAnalysisV2` decision. Do not rank logs, mutable registries or raw HOF metrics.
2. Select only eligible candidates from one comparable replay panel using the non-dominated
   Return-LCB/Expectancy-LCB/DD-UCB/ES-UCB contract.
3. Create an unchanged control plus paired treatment arms. Change one factor per arm unless a
   factorial design is explicitly declared.
4. Treat population, generations, mutation, pair count and optional features as hypotheses, not
   fixed “proven” ranges.
5. Generate `ChildWavePlanV2`, materialize all configs/manifests/candidates/worker specs, bind the
   approval to plan and materialization hashes, then queue atomically.
6. Execute only through the SQLite V2 scheduler and collect the complete declared Attempt set.

Before approval, require:

- canonical schema validation;
- identical comparison panels, paired seeds and declared budgets;
- no reused Final-Test cell;
- content-addressed code, data, config and candidate inputs;
- no unsupported worker kind or `safe_v2` feature;
- a complete immutable materialization receipt.

Legacy `config/queue` files, `ga_auto_queue_v2.sh`, direct `run_ga.py` launches and log-derived
rankings are not the canonical next-wave path.
