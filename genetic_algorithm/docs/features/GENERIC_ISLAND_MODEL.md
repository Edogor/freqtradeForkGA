# Generic Island Model

> A heavily optimized multi-population GA that runs **parallel isolated sub-populations** (islands), periodically exchanging migrants. Designed around a single shared process pool to prevent OOM crashes on memory-constrained systems.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Memory Optimisation — Shared Process Pool](#3-memory-optimisation--shared-process-pool)
4. [Migration Mechanics](#4-migration-mechanics)
5. [Merge Rounds](#5-merge-rounds)
6. [Hall of Fame Integration](#6-hall-of-fame-integration)
7. [Configuration Reference](#7-configuration-reference)
8. [Recommended Configs](#8-recommended-configs)
9. [Known Anti-Patterns](#9-known-anti-patterns)

---

## 1. Overview

The `GenericIslandModel` (`generic_island_model.py`) orchestrates N independent `GeneticAlgorithm` instances (islands). Each island evolves its population independently, but migrants are periodically swapped between islands to maintain genetic diversity and prevent premature convergence.

Key design goals:
- **Memory safety**: shared single `ParallelEvaluator` (not one per island)
- **Topology-aware migration**: ring, fully-connected, or random topologies
- **Checkpoint / resume**: atomic saves after every generation
- **Thread-safe statistics**: safe logging and progress tracking from multi-threaded evolution

---

## 2. Architecture

```
GenericIslandModel
│
├─ _phase1_create_islands()        # Instantiates N GeneticAlgorithm objects
│    └─ _build_island_config()     # Disables per-island parallel (shared pool)
│
├─ _evolve_inner()                 # Main evolution loop
│    ├─ Creates ONE shared ParallelEvaluator
│    ├─ For each generation:
│    │    ├─ _evolve_island_one_generation()  # inject shared evaluator
│    │    ├─ _maybe_migrate()                # topology-based migration
│    │    └─ _maybe_merge_round()            # periodic full merges
│    └─ finally: shared evaluator shutdown
│
└─ _phase2_finalize()              # Collect best from all islands, save top-5
```

---

## 3. Memory Optimisation — Shared Process Pool

**Problem**: A naïve implementation creates one `ProcessPoolExecutor` per island. With 20 islands × 4 workers = 80 worker processes, each loading ~800 MB–1.2 GB of OHLCV data. Total memory demand on a 16 GB system → OOM crash at island 4–5.

**Fix** (implemented in `generic_island_model.py`):

1. `_build_island_config()` forces `parallel_evaluation.enabled = False` for each island GA.
2. `_evolve_inner()` creates **one** shared `ParallelEvaluator` at the island-model level.
3. `_evolve_island_one_generation()` injects the shared evaluator into the island GA before calling `evaluate_population()`.

**Result**: 4 workers total regardless of island count. Memory stable at ~11–13 GB on a 16 GB system.

---

## 4. Migration Mechanics

After every `migration_interval` generations, the island model runs migration:

| Topology | Behaviour |
|----------|-----------|
| `ring` | Each island sends migrants to the next island (circular) |
| `fully_connected` | Each island can receive migrants from any other island |
| `random` | Each island sends to a randomly selected target |

**Migration config keys:**
```yaml
island_model:
  migration_interval: 5        # How often to migrate (generations)
  migration_rate: 0.1          # Fraction of population to migrate
  migration_topology: ring     # ring | fully_connected | random
  migration_selection: best    # best | random | tournament
```

Migrants arriving at a destination island replace the worst individuals if the migrant fitness exceeds the local worst.

---

## 5. Merge Rounds

At a configurable interval, a full merge round is triggered:

- All island populations are combined into one pool.
- The top-N individuals by fitness are selected.
- They are redistributed across islands (with diversity seeding for remaining slots).

This prevents long-term isolation where islands diverge too far without exchanging useful building blocks.

```yaml
island_model:
  merge_rounds: 2              # Total number of merge rounds
  merge_at_generation: [10, 20]  # Or specify exact generations
```

---

## 6. Hall of Fame Integration

Each island maintains its own local Hall of Fame directory. The island model also writes to a shared global HoF at the end. This allows:

- Per-island specialization (each island's HoF may reflect a different market regime or pair)
- Global diversity in the final top-N output

---

## 7. Configuration Reference

```yaml
island_model:
  num_islands: 4               # Experimental factor; minimum is one
  population_per_island: 6     # Experimental factor; mechanical minimum is two
  migration_interval: 5        # Migrate every N generations
  migration_rate: 0.15         # Fraction of pop to migrate
  migration_topology: ring     # ring | fully_connected | random
  migration_selection: best    # best | random | tournament
  merge_rounds: 2              # Full merge-and-redistribute events
  independent_populations: true  # Each island evolves independently between migrations
  pair_split: false            # Each island gets a different trading pair (pair-split mode)
  hall_of_fame_dir: genetic_algorithm/data/hall_of_fame_island
```

### Parallel Evaluation (island-level)

Do **not** enable `parallel_evaluation` individually for each island. The island model manages the shared pool:

```yaml
parallel_evaluation:
  enabled: true          # Enabled at island-model level (manages shared pool)
  num_workers: 4         # Worker count for the shared pool
  backtest_timeout: 300  # Timeout per backtest (seconds)
```

---

## 8. Recommended Configs

### Minimal / Quick Test

```yaml
island_model:
  num_islands: 2
  population_per_island: 4
  migration_interval: 3
  migration_rate: 0.20
  migration_topology: ring
  merge_rounds: 0
```

### Balanced Production

```yaml
island_model:
  num_islands: 6
  population_per_island: 6
  migration_interval: 5
  migration_rate: 0.15
  migration_topology: ring
  migration_selection: best
  merge_rounds: 2
parallel_evaluation:
  enabled: true
  num_workers: 4
  backtest_timeout: 300
```

### Pair-Split (each island gets its own pair)

```yaml
island_model:
  num_islands: 4
  population_per_island: 6
  pair_split: true       # Pairs distributed across islands
  migration_topology: ring
  migration_interval: 5
  merge_rounds: 1
backtesting:
  pairs:
    - BTC/USDT
    - ETH/USDT
    - BNB/USDT
    - SOL/USDT
```

> See `config/ga_config_pair_split_C1.yaml` and `C2.yaml` for full examples.

---

## 9. Config-Vertrag und offene Hypothesen

`population_per_island` muss mechanisch mindestens 2 betragen; einen universell optimalen Wert 6,
60 oder eine andere feste Grenze gibt es nicht. Größere und kleinere Inselpopulationen verändern
Suchbreite, Replikationsbudget und Laufzeit gleichzeitig und müssen daher als gepaarter
V2-Experimentalarm verglichen werden.

Generic Islands können Walk-forward explizit pro Insel ausführen. Das ältere regimegebundene
`island_model` deaktiviert Walk-forward dagegen intern; dessen zentraler Config-Vertrag blockiert
die widersprüchliche Kombination inzwischen vor dem Start. Crossover, Migration und optionale
Validierungsfeatures bleiben experimentelle Faktoren, bis gleiche Panels, Budgets und Seeds einen
OOS-Vorteil belegen.

---

*Last updated: April 2026*
