"""
ExperimentRegistry — the single source of truth for all GA experiments.

Every component reads/writes through this registry instead of scanning
the filesystem, parsing logs, or inspecting PID files.

Storage: JSON file with atomic writes (write-to-temp + rename).
Locking: ``fcntl.flock`` (advisory) to prevent concurrent corruption
         from scheduler, runners, and monitors writing at the same time.

Usage::

    from genetic_algorithm.orchestration.registry import ExperimentRegistry

    reg = ExperimentRegistry()                  # uses default path
    reg.register("E200_scalp_5m", config_path="config/presets/island.yaml",
                 tags=["batch1", "scalping"])
    reg.update("E200_scalp_5m", status="running", pid=12345)
    reg.update("E200_scalp_5m", generation=5, best_fitness=0.72)
    reg.complete("E200_scalp_5m", best_fitness=0.74, best_profit=32.1)

    for exp in reg.list(status="running"):
        print(exp["experiment_id"], exp["best_fitness"])
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_REGISTRY_PATH = Path("genetic_algorithm/data/registry.json")

_VALID_STATUSES = frozenset({"queued", "running", "completed", "failed", "cancelled"})


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class ExperimentRegistry:
    """Atomic, file-locked JSON registry of all GA experiments.

    The registry stores a flat dict of experiments keyed by *experiment_id*.
    It supports concurrent readers/writers via ``fcntl.flock`` advisory locks.

    File format::

        {
            "version": 1,
            "experiments": { "<id>": { ... }, ... }
        }
    """

    VERSION = 1

    def __init__(self, path: Optional[Path | str] = None) -> None:
        self.path = Path(path) if path else _DEFAULT_REGISTRY_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write({"version": self.VERSION, "experiments": {}})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(
        self,
        experiment_id: str,
        *,
        config_path: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        ga_type: str = "standard",
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Register a new experiment (status = *queued*)."""
        now = _now_iso()
        entry: Dict[str, Any] = {
            "experiment_id": experiment_id,
            "status": "queued",
            "config_path": config_path,
            "tags": list(tags or []),
            "ga_type": ga_type,
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "finished_at": None,
            "pid": None,
            "log_path": None,
            "data_dir": None,
            "checkpoint_dir": None,
            "generation": None,
            "generations_total": None,
            "best_fitness": None,
            "best_profit": None,
            "error": None,
        }
        if extra:
            entry.update(extra)

        data = self._read()
        if experiment_id in data["experiments"]:
            raise ValueError(f"Experiment '{experiment_id}' already registered")
        data["experiments"][experiment_id] = entry
        self._write(data)
        logger.info("Registered experiment %s", experiment_id)
        return deepcopy(entry)

    def start(
        self,
        experiment_id: str,
        *,
        pid: Optional[int] = None,
        log_path: Optional[str] = None,
        data_dir: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
        generations_total: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Mark an experiment as *running*."""
        updates: Dict[str, Any] = {
            "status": "running",
            "started_at": _now_iso(),
            "pid": pid or os.getpid(),
            "log_path": log_path,
            "data_dir": data_dir,
            "checkpoint_dir": checkpoint_dir,
            "generations_total": generations_total,
        }
        if extra:
            updates.update(extra)
        self._update(experiment_id, updates)

    def update(self, experiment_id: str, **fields: Any) -> None:
        """Update arbitrary fields on a running experiment.

        Commonly used for periodic generation/fitness updates::

            reg.update("E200", generation=5, best_fitness=0.71)
        """
        self._update(experiment_id, fields)

    def complete(self, experiment_id: str, **fields: Any) -> None:
        """Mark an experiment as *completed*."""
        fields["status"] = "completed"
        fields["finished_at"] = _now_iso()
        fields["pid"] = None
        self._update(experiment_id, fields)

    def fail(self, experiment_id: str, error: str = "", **fields: Any) -> None:
        """Mark an experiment as *failed*."""
        fields["status"] = "failed"
        fields["finished_at"] = _now_iso()
        fields["error"] = error
        fields["pid"] = None
        self._update(experiment_id, fields)

    def cancel(self, experiment_id: str) -> None:
        """Mark an experiment as *cancelled*."""
        self._update(experiment_id, {"status": "cancelled", "finished_at": _now_iso(), "pid": None})

    def get(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Return a single experiment entry, or ``None``."""
        data = self._read()
        entry = data["experiments"].get(experiment_id)
        return deepcopy(entry) if entry else None

    def list(
        self,
        *,
        status: Optional[str | Sequence[str]] = None,
        tags: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Return experiments matching the given filters.

        Parameters
        ----------
        status : filter by one or more statuses (e.g. "running" or ["running", "queued"])
        tags :   filter by experiments that have *all* of the given tags
        limit :  max results (most-recently-updated first)
        """
        data = self._read()
        results: List[Dict[str, Any]] = []

        if isinstance(status, str):
            status = [status]
        status_set = set(status) if status else None
        tag_set = set(tags) if tags else None

        for entry in data["experiments"].values():
            if status_set and entry.get("status") not in status_set:
                continue
            if tag_set and not tag_set.issubset(set(entry.get("tags", []))):
                continue
            results.append(deepcopy(entry))

        results.sort(key=lambda e: e.get("updated_at", ""), reverse=True)
        if limit:
            results = results[:limit]
        return results

    def remove(self, experiment_id: str) -> bool:
        """Remove an experiment from the registry. Returns True if found."""
        data = self._read()
        if experiment_id not in data["experiments"]:
            return False
        del data["experiments"][experiment_id]
        self._write(data)
        logger.info("Removed experiment %s from registry", experiment_id)
        return True

    def count(self, status: Optional[str] = None) -> int:
        """Count experiments, optionally filtered by status."""
        data = self._read()
        if status is None:
            return len(data["experiments"])
        return sum(1 for e in data["experiments"].values() if e.get("status") == status)

    def summary(self) -> Dict[str, int]:
        """Return a dict of status → count."""
        data = self._read()
        counts: Dict[str, int] = {}
        for entry in data["experiments"].values():
            s = entry.get("status", "unknown")
            counts[s] = counts.get(s, 0) + 1
        return counts

    # ------------------------------------------------------------------
    # Internals (locked read/write)
    # ------------------------------------------------------------------

    def _update(self, experiment_id: str, fields: Dict[str, Any]) -> None:
        data = self._read()
        if experiment_id not in data["experiments"]:
            raise KeyError(f"Experiment '{experiment_id}' not found in registry")
        # Validate status transitions
        if "status" in fields:
            new_status = fields["status"]
            if new_status not in _VALID_STATUSES:
                raise ValueError(f"Invalid status '{new_status}'; valid: {_VALID_STATUSES}")
        data["experiments"][experiment_id].update(fields)
        data["experiments"][experiment_id]["updated_at"] = _now_iso()
        self._write(data)

    def _read(self) -> Dict[str, Any]:
        """Read the registry file under a shared (read) lock."""
        try:
            with open(self.path, "r") as fh:
                fcntl.flock(fh, fcntl.LOCK_SH)
                try:
                    return json.load(fh)
                finally:
                    fcntl.flock(fh, fcntl.LOCK_UN)
        except (json.JSONDecodeError, FileNotFoundError):
            logger.warning("Registry file missing or corrupt; reinitializing")
            return {"version": self.VERSION, "experiments": {}}

    def _write(self, data: Dict[str, Any]) -> None:
        """Atomically write the registry (tmp + rename) under exclusive lock."""
        tmp_path = self.path.with_suffix(".tmp")
        with open(tmp_path, "w") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                json.dump(data, fh, indent=2, sort_keys=False)
                fh.write("\n")
                fh.flush()
                os.fsync(fh.fileno())
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)
        tmp_path.rename(self.path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
