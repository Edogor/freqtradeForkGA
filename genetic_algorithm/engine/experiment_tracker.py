"""
Experiment Tracker — persists generation snapshots to disk for the web dashboard.

Writes to ``genetic_algorithm/data/runs/{run_id}/``:
  - ``config.yaml``   — experiment config (written once at start)
  - ``gen_{NNNN}.json`` — per-generation stats + individuals
  - ``events.jsonl``   — append-only event log (NEW_BEST, MIGRATION, ERROR, …)
  - ``final_results.json`` — written at experiment completion

The ``DataService`` already scans ``data/runs/`` so no dashboard code changes
are needed — CLI experiments will automatically appear in the dashboard.

All writes use atomic tmp→rename to avoid partial reads.
"""

import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

RUNS_DIR = Path("genetic_algorithm/data/runs")


class ExperimentTracker:
    """Persists experiment data for the web dashboard."""

    def __init__(self, config: Dict[str, Any], run_id: Optional[str] = None):
        self.config = config
        ga = config.get('genetic_algorithm', {})
        self.run_id = run_id or config.get('_config_name', config.get('experiment_name', ga.get('experiment_name', f'run_{int(time.time())}')))
        self.run_dir = RUNS_DIR / self.run_id
        self._started_at = time.time()
        self._event_path = self.run_dir / "events.jsonl"
        self._initialised = False
        # T1.3 — Experiments DB writer.  ``None`` until first use; opened
        # lazily so legacy callers without sqlite remain unaffected.
        self._db = None  # type: Optional[Any]

    def _get_db(self):
        """Lazy-init the ExperimentDB writer (T1.3)."""
        if self._db is False:
            return None
        if self._db is None:
            try:
                from genetic_algorithm.data.experiment_db import ExperimentDB
                self._db = ExperimentDB()
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug(f"[TRACKER] ExperimentDB unavailable: {exc}")
                self._db = False  # type: ignore[assignment]
                return None
        return self._db

    def initialise(self):
        """Create run directory and write config. Call once at experiment start."""
        if self._initialised:
            return
        self.run_dir.mkdir(parents=True, exist_ok=True)
        # Write config
        config_path = self.run_dir / "config.yaml"
        self._atomic_write_yaml(config_path, self.config)
        self._append_event("STARTED", {"run_id": self.run_id})
        self._initialised = True
        # T1.3 — register the run in the experiments DB.
        db = self._get_db()
        if db is not None:
            try:
                ga_cfg = self.config.get('genetic_algorithm', {})
                generations_total = ga_cfg.get('generations')
                db.record_run_start(
                    self.run_id,
                    name=self.run_id,
                    ga_type=_detect_ga_type_from_config(self.config),
                    generations_total=generations_total,
                    config=self.config,
                )
            except Exception as exc:
                logger.debug(f"[TRACKER] DB record_run_start failed: {exc}")
        logger.info(f"[TRACKER] Initialised run directory: {self.run_dir}")

    def save_generation(self, generation: int, stats, population, island_stats: Optional[List[Dict]] = None):
        """Save a generation snapshot.

        Args:
            generation: Generation number (0-based)
            stats: PopulationStats dataclass or dict
            population: Population object (must have .individuals)
            island_stats: Optional per-island stats dicts
        """
        if not self._initialised:
            self.initialise()

        # Convert stats to dict
        if hasattr(stats, '__dataclass_fields__'):
            import dataclasses
            stats_dict = dataclasses.asdict(stats)
        elif hasattr(stats, '__dict__'):
            stats_dict = {k: v for k, v in vars(stats).items() if not k.startswith('_')}
        elif isinstance(stats, dict):
            stats_dict = stats
        else:
            stats_dict = {}

        # Build individual dicts
        individuals = []
        for ind in population:
            try:
                individuals.append(ind.to_dict())
            except Exception:
                pass

        snapshot = {
            "generation": generation,
            "timestamp": time.time(),
            "stats": stats_dict,
            "individuals": individuals,
        }
        if island_stats:
            snapshot["island_stats"] = island_stats

        path = self.run_dir / f"gen_{generation:04d}.json"
        self._atomic_write_json(path, snapshot)
        logger.debug(f"[TRACKER] Saved generation {generation} ({len(individuals)} individuals)")

    def record_event(self, event_type: str, data: Optional[Dict[str, Any]] = None):
        """Append an event to the event log."""
        self._append_event(event_type, data or {})

    def record_new_best(self, generation: int, fitness: float, metrics: Optional[Dict] = None):
        """Record a NEW_BEST event."""
        self._append_event("NEW_BEST", {
            "generation": generation,
            "fitness": fitness,
            "profit": metrics.get("profit") if metrics else None,
        })

    def save_final_results(self, best_individual, generation_stats: list):
        """Write final_results.json at experiment completion."""
        if not self._initialised:
            return

        result = {
            "run_id": self.run_id,
            "completed_at": time.time(),
            "elapsed_seconds": time.time() - self._started_at,
            "total_generations": len(generation_stats),
        }

        if best_individual is not None:
            result["best_fitness"] = best_individual.fitness
            result["best_raw_fitness"] = best_individual.raw_fitness
            result["best_metrics"] = best_individual.metrics if best_individual.metrics else {}
            try:
                result["best_individual"] = best_individual.to_dict()
            except Exception:
                pass

        if generation_stats:
            fitness_history = []
            for s in generation_stats:
                if hasattr(s, 'best_fitness'):
                    fitness_history.append(s.best_fitness)
                elif isinstance(s, dict):
                    fitness_history.append(s.get('best_fitness'))
            result["fitness_history"] = fitness_history

        path = self.run_dir / "final_results.json"
        self._atomic_write_json(path, result)
        self._append_event("COMPLETED", {
            "best_fitness": result.get("best_fitness"),
            "total_generations": result.get("total_generations"),
        })
        # T1.3 — persist completion + best individual into experiments DB.
        db = self._get_db()
        if db is not None:
            try:
                db.record_run_completion(
                    self.run_id,
                    best_fitness=result.get("best_fitness"),
                    best_profit=(result.get("best_metrics") or {}).get("profit"),
                    elapsed_seconds=result.get("elapsed_seconds"),
                    status="completed",
                )
                if best_individual is not None:
                    bi = result.get("best_individual") or {}
                    db.record_strategy(
                        self.run_id,
                        strategy_id=f"{self.run_id}_best",
                        rank=1,
                        fitness=result.get("best_fitness"),
                        metrics=result.get("best_metrics") or {},
                        gene_dict=bi.get("strategy_gene") if isinstance(bi, dict) else None,
                    )
            except Exception as exc:
                logger.debug(f"[TRACKER] DB record completion failed: {exc}")
        logger.info(f"[TRACKER] Final results saved to {path}")

    # ── Internal helpers ──────────────────────────────────────────

    def _atomic_write_json(self, path: Path, data: Any):
        """Write JSON atomically via tmp file + rename."""
        tmp_path = path.with_suffix('.tmp')
        try:
            with open(tmp_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            os.replace(str(tmp_path), str(path))
        except Exception as e:
            logger.warning(f"[TRACKER] Failed to write {path}: {e}")
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

    def _atomic_write_yaml(self, path: Path, data: Any):
        """Write YAML atomically via tmp file + rename."""
        tmp_path = path.with_suffix('.tmp')
        try:
            with open(tmp_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            os.replace(str(tmp_path), str(path))
        except Exception as e:
            logger.warning(f"[TRACKER] Failed to write {path}: {e}")
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

    def _append_event(self, event_type: str, data: Dict[str, Any]):
        """Append a single event line to the JSONL log."""
        if not self._initialised and event_type != "STARTED":
            return
        event = {
            "ts": time.time(),
            "type": event_type,
            **data,
        }
        try:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            with open(self._event_path, 'a') as f:
                f.write(json.dumps(event, default=str) + '\n')
        except Exception as e:
            logger.debug(f"[TRACKER] Failed to append event: {e}")


def _detect_ga_type_from_config(config: Dict[str, Any]) -> str:
    """Mirror of ``cli._detect_ga_type`` to avoid a circular import."""
    if (config.get("generic_island_model") or {}).get("enabled"):
        return "generic_island"
    if (config.get("island_model") or {}).get("enabled"):
        return "island"
    return "standard"
