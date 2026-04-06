# GA Infrastructure Redesign Plan

> **Branch**: `feature/ga-infrastructure-redesign`  
> **Based on**: `feature/pre-merge-integration`  
> **Created**: 2026-04-06  
> **Status**: Planning & Analysis Phase

---

## 1. Motivation

The GA system has grown organically from a simple evolutionary optimizer into a complex multi-component system with:
- Island model evolution with multiple topologies
- Walk-forward & Monte Carlo validation
- Strategy Intelligence System (SIS) with ML predictors
- Web dashboard with live WebSocket updates
- LLM-assisted strategy seeding
- Automated queue management with persistent daemon
- 4 overlapping monitor scripts

This organic growth created **fragmentation** in monitoring, data storage, experiment tracking, and workflow automation. This branch aims to systematically improve the infrastructure foundation.

---

## 2. Current Architecture (As-Is)

### 2.1 Experiment Lifecycle

```
Create Config (YAML)
    ↓
Queue: config/queue/ or config/exploration/wave{N}/
    ↓
Launch: ga_auto_queue_v2.sh (daemon) OR run_parallel_wave.sh (one-shot)
    ↓
Execute: python run_ga.py --config <yaml>
    ↓  ┌─────────────────────────────────────────────┐
    ↓  │ Per Generation:                              │
    ↓  │  evaluate → cache → checkpoint → hall_of_fame│
    ↓  │  → experiment_tracker → select → reproduce   │
    ↓  └─────────────────────────────────────────────┘
    ↓
Complete: final_results.json + top-5 strategy .py files
    ↓
Archive: config moved to config/done/wave{N}/
```

### 2.2 Data Storage (Current)

| Store | Location | Purpose | Cleanup |
|-------|----------|---------|---------|
| Backtest cache | `data/cache/` | Speed up repeated evaluations | ❌ None (grows unbounded) |
| Checkpoints | `data/checkpoints_{wave}_{exp}/` | Resume from crash | ❌ 50+ dirs accumulated |
| Hall of Fame | `data/hall_of_fame_{wave}_{exp}/` | Best strategies archive | Capped at 50 entries |
| Run History | `data/runs/{run_id}/` | Full audit trail per run | ❌ Never cleaned |
| SIS Corpus | `data/strategy_corpus.parquet` | ML training data | Manual rebuild |
| Output | `output/` | Generated strategy .py files | ❌ Accumulates |
| Logs | `logs/` | Runtime logs per experiment | ❌ Grows with runs |

### 2.3 Monitoring (Current — 4 Overlapping Tools)

| Script | Lines | Focus |
|--------|-------|-------|
| `ga_monitor.sh` (v1) | 427 | Basic dashboard (LEGACY) |
| `ga_monitor_v2.sh` | 1075 | Enhanced: queue, detail levels, island support |
| `ga_current.sh` | ~300 | Running-only, rich per-island detail |
| `wave_monitor.sh` | 221 | Single wave, compact |

**Problem**: Each independently reimplements log parsing regex. Bugs fixed in one aren't fixed in others.

### 2.4 Launch & Queue (Current)

| Script | Type | Scalable? |
|--------|------|-----------|
| `ga_auto_queue_v2.sh` | Persistent daemon | ✅ Yes |
| `ga_auto_queue.sh` (v1) | Simple daemon | ⚠️ Legacy |
| `run_parallel_wave.sh` | One-shot batch | ✅ Yes |
| `launch_wave{30,31,32}.sh` | Hardcoded per-wave | ❌ No |

### 2.5 PID & State Tracking (Current — Fragmented)

| File | Writer | Reader |
|------|--------|--------|
| `auto_queue_daemon.pid` | Queue daemon | Monitors |
| `auto_queue_tracked.txt` | Queue daemon | Monitors |
| `{wave}_pids_{TS}.txt` | Parallel launcher | Manual |
| `auto_queue_state.json` | Queue daemon | ga_monitor_v2 |
| `ps aux \| grep run_ga` | N/A (fallback) | All monitors |

---

## 3. Problems to Solve

### P1: No Central Experiment Registry
There is no single source of truth for "what experiments exist, what's running, what completed, what failed." Shell scripts scan PID files and log dirs; Python's ExperimentTracker writes to `data/runs/` but shell scripts can't efficiently read those.

### P2: Monitor Fragmentation
4 monitors reimplementing the same log parsing in bash. Each has different parsing bugs, different feature sets, different metric extraction approaches.

### P3: Unbounded Data Growth
Cache, checkpoints, runs, logs grow without any retention policy. Server currently at 60% disk. No automated cleanup.

### P4: SIS Corpus Staleness
The strategy corpus (ML training data) is rebuilt manually. No trigger on run completion. New SIS-enabled runs may use stale models.

### P5: Manual Wave Management
Creating a new wave requires: write configs, put in right directory, create launch script (or use queue). No automation for "what should we run next?"

### P6: Shell ↔ Python State Desync
Queue daemon (bash) and evolution engine (Python) maintain separate state. Monitors reconstruct state by parsing logs. No shared protocol.

---

## 4. Proposed Architecture (To-Be)

### 4.1 Pillar 1: Experiment Registry

A **single JSON file** that is the source of truth for all experiments:

```
genetic_algorithm/data/experiment_registry.json
```

```json
{
  "version": 1,
  "last_updated": "2026-04-06T12:34:56Z",
  "experiments": {
    "wave32_A": {
      "experiment_id": "E160",
      "wave": "wave32",
      "slot": "A",
      "status": "completed",
      "pid": null,
      "config_path": "genetic_algorithm/config/done/wave32/A_15m_rank_ring.yaml",
      "log_path": "genetic_algorithm/logs/wave32_A.log",
      "data_dir": "genetic_algorithm/data/runs/wave32_A",
      "checkpoint_dir": "genetic_algorithm/data/checkpoints_wave32_A",
      "hof_dir": "genetic_algorithm/data/hall_of_fame_wave32_A",
      "start_time": "2026-03-15T08:00:00Z",
      "end_time": "2026-03-15T18:30:00Z",
      "duration_hours": 10.5,
      "generations_completed": 60,
      "generations_total": 60,
      "best_fitness": 0.6842,
      "best_train_profit": 45.2,
      "best_val_profit": 12.1,
      "ga_type": "island_model",
      "sis_enabled": true,
      "total_strategies_evaluated": 3600,
      "hof_entries_contributed": 3,
      "errors": 0
    }
  },
  "waves": {
    "wave32": {
      "created": "2026-03-15",
      "experiments": ["wave32_A", "wave32_B", "wave32_C", "wave32_D", "wave32_E"],
      "completed": 5,
      "failed": 0,
      "best_fitness": 0.7102,
      "best_experiment": "wave32_A"
    }
  }
}
```

**Who writes**:
- `run_ga.py` — registers on start, updates on generation milestones, finalizes on completion
- `ga_auto_queue_v2.sh` — registers on launch, updates PID, marks failed if PID exits non-zero

**Who reads**:
- All monitors — single data source, no log parsing for status
- Web dashboard — replace dir scanning with registry lookup
- SIS corpus builder — know which runs have fresh data
- Cleanup tools — identify archivable/deletable data

### 4.2 Pillar 2: Unified Monitoring

**Goal**: One monitor tool with modes, backed by experiment registry.

```bash
# Replace 4 scripts with 1:
./ga_monitor.sh                     # default: running experiments
./ga_monitor.sh --wave wave32       # filter to wave
./ga_monitor.sh --all               # include completed
./ga_monitor.sh --live              # live log tails
./ga_monitor.sh --queue             # show queue contents
./ga_monitor.sh --summary           # aggregate stats
./ga_monitor.sh --detail compact    # fewer columns
```

**Internal architecture**:
- Read experiment_registry.json for status/metrics
- Only parse logs for real-time generation progress (not state reconstruction)
- Shared log parsing functions (source from common library)

### 4.3 Pillar 3: Data Lifecycle Manager

A Python utility managing data retention:

```python
# genetic_algorithm/utils/data_lifecycle.py

class DataLifecycleManager:
    """Automated data cleanup with configurable retention policies."""
    
    policies = {
        "cache": {"max_age_days": 14, "max_size_mb": 2000},
        "checkpoints": {"keep_latest_n": 3, "keep_best_n": 1},
        "runs": {"keep_days": 90, "archive_after_days": 30},
        "logs": {"keep_days": 30, "compress_after_days": 7},
    }
    
    def cleanup(self, dry_run=True): ...
    def archive(self, run_id): ...
    def disk_report(self): ...
```

**Triggers**:
- Before each new run (disk check already exists — extend to cleanup)
- On schedule (cron or daemon --cleanup interval)
- Manual: `python -m genetic_algorithm.utils.data_lifecycle --report`

### 4.4 Pillar 4: Automated Strategy Discovery Pipeline

**Vision**: Self-sustaining evolution loop.

```
Completed Runs
    ↓
Analyze Results (top performers, gaps, diversity)
    ↓
Generate Next-Wave Configs
    ↓  - Exploit: configs similar to top performers (larger pop, more gens)
    ↓  - Explore: configs for underrepresented archetypes/timeframes
    ↓  - Complement: configs targeting weaknesses of top strategies
    ↓
Queue → Auto-Launch → Monitor → Complete → Loop
```

**Components**:
- Config generator: template-based with parameter ranges
- Wave planner: decides what to run next based on portfolio gaps
- SIS feedback: which archetypes are missing? which timeframes unexplored?

### 4.5 Pillar 5: Unified Configuration & Launch

- Deprecate all per-wave `launch_wave{N}.sh` scripts
- Enhance `ga_auto_queue_v2.sh` as the single launch mechanism
- Config templates with `{{variable}}` substitution
- Experiment ID auto-assignment from registry

---

## 5. Implementation Phases (Rough Order)

### Phase 1: Foundation (Registry + Cleanup)
- [ ] Design experiment_registry.json schema
- [ ] Add registry writes to run_ga.py (start, milestone, complete)
- [ ] Add registry reads to ga_auto_queue_v2.sh
- [ ] Implement basic data lifecycle cleanup
- [ ] Backfill registry from existing runs/ data

### Phase 2: Unified Monitoring
- [ ] Refactor ga_monitor_v2.sh to read from registry
- [ ] Consolidate log parsing into shared functions
- [ ] Deprecate ga_monitor.sh (v1), wave_monitor.sh
- [ ] Merge ga_current.sh features into ga_monitor_v2.sh

### Phase 3: Data Management
- [ ] Implement data_lifecycle.py with configurable policies
- [ ] Add auto-cleanup hooks to run startup
- [ ] Add SIS corpus auto-rebuild trigger
- [ ] Checkpoint retention policy (keep last N + best)

### Phase 4: Automated Pipeline
- [ ] Config template system
- [ ] Wave planner based on portfolio analysis
- [ ] SIS-driven config generation
- [ ] Self-sustaining loop integration

### Phase 5: Cleanup & Stabilization
- [ ] Remove deprecated scripts
- [ ] Update all documentation
- [ ] Server migration to new workflow
- [ ] Integration tests for infrastructure

---

## 6. Files to Consider (Current Inventory)

### Core GA (keep, may refactor)
- `genetic_algorithm/run_ga.py` — main entry point
- `genetic_algorithm/core/evolution.py` — main loop
- `genetic_algorithm/core/experiment_tracker.py` — run data persistence
- `genetic_algorithm/core/hall_of_fame.py` — strategy archive
- `genetic_algorithm/evaluation/direct_backtester.py` — backtest + cache

### Shell Scripts (consolidate)
- `ga_auto_queue_v2.sh` — **keep as primary** queue manager
- `ga_monitor_v2.sh` — **keep as primary** monitor (absorb others)
- `run_parallel_wave.sh` — **keep** for one-shot batch launches
- `ga_auto_queue.sh` — deprecate (v1)
- `ga_monitor.sh` — deprecate (v1)
- `launch_wave{30,31,32}.sh` — deprecate (use generic launcher)
- `wave_monitor.sh` — deprecate (fold into monitor_v2)

### New Files (to create)
- `genetic_algorithm/data/experiment_registry.json` — central registry
- `genetic_algorithm/utils/data_lifecycle.py` — data management
- `genetic_algorithm/utils/registry.py` — Python registry API
- `genetic_algorithm/utils/wave_planner.py` — automated config generation (Phase 4)

---

## 7. Server Data Summary (2026-04-06)

- **79 completed runs** in `data/runs/`
- **292 hall-of-fame files** across all experiments
- **248 archived configs** in `config/done/`
- **50+ checkpoint directories** (no cleanup)
- **38MB cache** (growing)
- **Server disk**: 124GB used / 221GB total (60%)
- **Currently running**: 7 experiments (wave31 A-D + SIS production E/F + wave34 HIE)
- **Queue daemon**: `ga_auto_queue_v2.sh --wave wave31 --max 5 --persistent`

---

## 8. Notes & Ideas

- The server has local changes to `ga_monitor_v2.sh` (--live mode, HoF block) and `direct_backtester.py` (memory LRU orderedDict) that should be merged
- Consider moving from pickle checkpoints to JSON or parquet for portability
- Web dashboard could become the primary monitoring interface (replace shell monitors for most use cases)
- Experiment registry could enable: auto-generated reports, strategy lineage tracking, SIS corpus versioning
- Consider a "strategy pool" concept: a curated, deduplicated collection of the best strategies across all runs, with provenance metadata
