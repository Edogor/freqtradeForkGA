---
applyTo: "genetic_algorithm/orchestration/**"
description: "GA orchestration conventions. Use when: editing ExperimentRegistry, DataLifecycle, Scheduler, Monitor, or experiment lifecycle management code."
---

# GA Orchestration Conventions

**Key modules:**
- `registry.py` — `ExperimentRegistry`: atomic JSON store with `fcntl` file locking for experiment CRUD (register → start → complete/fail/cancel)
- `lifecycle.py` — `DataLifecycle`: policy-driven archival, compression, and cleanup of experiment data
- `scheduler.py` — Experiment scheduling and queue management
- `monitor.py` — `ExperimentMonitor`: live experiment status tracking

**Concurrency:**
- Registry uses `fcntl.flock` for process-level file safety (Linux only)
- All registry mutations go through `_save_registry()` which acquires the lock
- Temporary files + atomic rename for crash-safe writes

**Integration with run_ga.py:**
- `register()` after config validation
- `start()` before evolution begins (records PID, log path, data dir)
- `complete()` after successful post-evolution analysis (records best_fitness, best_profit)
- `fail()` on any exception during evolution or post-analysis
- `cancel()` on KeyboardInterrupt

**Testing:** Tests use `tmp_path` fixtures for isolated registry files — never write to the real registry.
