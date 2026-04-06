"""
DataLifecycle — manages archival, compression, and cleanup of experiment data.

Replaces ad-hoc ``find … -delete`` patterns from shell scripts with
policy-driven data management.
"""

from __future__ import annotations

import logging
import shutil
import tarfile
from datetime import datetime, timedelta
from pathlib import Path

from genetic_algorithm.orchestration.registry import ExperimentRegistry

logger = logging.getLogger(__name__)

# Default data root
DATA_ROOT = Path("genetic_algorithm/data")


class DataLifecycle:
    """Policy-driven archival, compression, and cleanup."""

    def __init__(
        self,
        archive_after_days: int = 7,
        delete_after_days: int = 90,
        compress: bool = True,
        dry_run: bool = True,
    ) -> None:
        self.archive_after_days = archive_after_days
        self.delete_after_days = delete_after_days
        self.compress = compress
        self.dry_run = dry_run
        self.registry = ExperimentRegistry()

    def run(self) -> dict:
        """Execute lifecycle policy.  Returns summary counts."""
        now = datetime.utcnow()
        archived = 0
        deleted = 0

        for exp in self.registry.list(status="completed"):
            ended = exp.get("ended_at")
            if not ended:
                continue
            ended_dt = datetime.fromisoformat(ended)
            age = (now - ended_dt).days

            eid = exp["experiment_id"]
            data_dir = DATA_ROOT / eid

            if age >= self.delete_after_days:
                deleted += self._delete(eid, data_dir)
            elif age >= self.archive_after_days:
                archived += self._archive(eid, data_dir)

        summary = {"archived": archived, "deleted": deleted, "dry_run": self.dry_run}
        logger.info("Lifecycle run: %s", summary)
        return summary

    def _archive(self, eid: str, data_dir: Path) -> int:
        archive_path = DATA_ROOT / "archive" / f"{eid}.tar.gz"
        if archive_path.exists():
            return 0
        if not data_dir.exists():
            return 0

        if self.dry_run:
            logger.info("[DRY-RUN] Would archive %s → %s", data_dir, archive_path)
            return 1

        archive_path.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(data_dir, arcname=eid)
        shutil.rmtree(data_dir)
        logger.info("Archived %s → %s", eid, archive_path)
        return 1

    def _delete(self, eid: str, data_dir: Path) -> int:
        archive_path = DATA_ROOT / "archive" / f"{eid}.tar.gz"

        if self.dry_run:
            logger.info("[DRY-RUN] Would delete %s", eid)
            return 1

        if data_dir.exists():
            shutil.rmtree(data_dir)
        if archive_path.exists():
            archive_path.unlink()

        self.registry.remove(eid)
        logger.info("Deleted all data for %s", eid)
        return 1
