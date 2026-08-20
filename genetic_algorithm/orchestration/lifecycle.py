"""
DataLifecycle — manages archival, compression, and cleanup of experiment data.

Replaces ad-hoc ``find … -delete`` patterns from shell scripts with
policy-driven data management.

Usage::

    lifecycle = DataLifecycle(dry_run=True)
    report = lifecycle.report()      # disk usage summary
    result = lifecycle.run()         # archive/delete by policy
"""

from __future__ import annotations

import logging
import shutil
import tarfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

from genetic_algorithm.orchestration.attempt_state_v2 import AttemptStateStoreV2
from genetic_algorithm.orchestration.experiment_catalog_v2 import ExperimentCatalogV2
from genetic_algorithm.orchestration.runner_v2 import default_state_path

logger = logging.getLogger(__name__)

# Default data root
DATA_ROOT = Path("genetic_algorithm/data")

# Categories for disk usage reporting
_CATEGORIES = {
    "checkpoints": ["checkpoints", "checkpoint"],
    "cache": ["cache"],
    "logs": ["logs"],
    "hall_of_fame": ["hall_of_fame", "hof"],
    "runs": ["runs"],
    "archive": ["archive"],
    "corpus": ["corpus"],
}


class DataLifecycle:
    """Policy-driven archival, compression, and cleanup."""

    def __init__(
        self,
        data_root: str | Path = DATA_ROOT,
        archive_after_days: int = 7,
        delete_after_days: int = 90,
        compress: bool = True,
        dry_run: bool = True,
        state_path: str | Path | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.archive_after_days = archive_after_days
        self.delete_after_days = delete_after_days
        self.compress = compress
        self.dry_run = dry_run
        self.catalog = ExperimentCatalogV2(
            AttemptStateStoreV2(state_path or default_state_path())
        )
        # Test/legacy injection hook. Production code leaves this unset so
        # registry.json can never become authoritative again.
        self.registry = None

    def run(self) -> Dict[str, Any]:
        """Execute lifecycle policy.  Returns summary counts."""
        now = datetime.now(UTC)
        archived = 0
        deleted = 0
        skipped_canonical = 0

        if self.registry is not None:
            experiments = self.registry.list(status="completed")
        else:
            experiments = [
                record.compatibility_dict()
                for record in self.catalog.list_records(statuses=["completed"])
            ]
        for exp in experiments:
            if exp.get("source") == "CANONICAL_V2":
                skipped_canonical += 1
                continue
            ended = exp.get("finished_at") or exp.get("ended_at")
            if not ended:
                continue
            try:
                ended_dt = datetime.fromisoformat(ended)
            except (ValueError, TypeError):
                continue
            if ended_dt.utcoffset() is None:
                ended_dt = ended_dt.replace(tzinfo=UTC)
            age = (now - ended_dt.astimezone(UTC)).days

            eid = exp["experiment_id"]
            data_dir = self.data_root / eid

            if age >= self.delete_after_days:
                deleted += self._delete(eid, data_dir)
            elif age >= self.archive_after_days:
                archived += self._archive(eid, data_dir)

        summary = {
            "archived": archived,
            "deleted": deleted,
            "skipped_canonical": skipped_canonical,
            "dry_run": self.dry_run,
        }
        logger.info("Lifecycle run: %s", summary)
        return summary

    def report(self) -> Dict[str, Any]:
        """Generate a disk usage report by category.

        Returns a dict with per-category sizes, total, and experiment count.
        """
        categories: Dict[str, int] = {}
        total_bytes = 0
        file_count = 0

        for cat_name, dir_names in _CATEGORIES.items():
            cat_bytes = 0
            for dn in dir_names:
                cat_dir = self.data_root / dn
                if cat_dir.is_dir():
                    for f in cat_dir.rglob("*"):
                        if f.is_file():
                            size = f.stat().st_size
                            cat_bytes += size
                            file_count += 1
            categories[cat_name] = cat_bytes
            total_bytes += cat_bytes

        # Project logs are in scope only for the real default data root. A
        # caller-provided/test root must remain fully isolated.
        log_dir = DATA_ROOT.parent / "logs"
        log_bytes = 0
        if self.data_root.resolve() == DATA_ROOT.resolve() and log_dir.is_dir():
            for f in log_dir.rglob("*"):
                if f.is_file():
                    log_bytes += f.stat().st_size
                    file_count += 1
        categories["project_logs"] = log_bytes
        total_bytes += log_bytes

        if self.registry is not None:
            exp_counts = {
                status: self.registry.count(status=status)
                for status in ("queued", "running", "completed", "failed", "cancelled")
            }
        else:
            exp_counts = self.catalog.summary()

        return {
            "categories": categories,
            "total_bytes": total_bytes,
            "total_human": _human_bytes(total_bytes),
            "file_count": file_count,
            "experiments": exp_counts,
        }

    def cleanup_logs(
        self, max_age_days: int = 30, min_size_mb: float = 0,
    ) -> int:
        """Remove old log files.

        Returns count of files removed (or would-be-removed in dry_run).
        """
        cutoff = datetime.now(UTC) - timedelta(days=max_age_days)
        removed = 0

        log_dirs = [self.data_root / "logs"]
        project_logs = DATA_ROOT.parent / "logs"
        if self.data_root.resolve() == DATA_ROOT.resolve():
            log_dirs.append(project_logs)
        for log_dir in dict.fromkeys(log_dirs):
            if not log_dir.is_dir():
                continue
            for f in log_dir.glob("*.log"):
                if not f.is_file():
                    continue
                mtime = datetime.fromtimestamp(f.stat().st_mtime, tz=UTC)
                size_mb = f.stat().st_size / (1024 * 1024)
                if mtime < cutoff and size_mb >= min_size_mb:
                    if self.dry_run:
                        logger.info(
                            "[DRY-RUN] Would remove %s (%.1f MB, %d days old)",
                            f, size_mb, (datetime.now(UTC) - mtime).days,
                        )
                    else:
                        f.unlink()
                        logger.info("Removed %s", f)
                    removed += 1

        return removed

    def cleanup_cache(self, max_size_mb: float = 500) -> int:
        """Remove oldest cache entries if total exceeds max_size_mb.

        Returns count of files removed.
        """
        cache_dir = self.data_root / "cache"
        if not cache_dir.is_dir():
            return 0

        files = sorted(
            cache_dir.rglob("*"),
            key=lambda f: f.stat().st_mtime if f.is_file() else 0,
        )
        files = [f for f in files if f.is_file()]

        total = sum(f.stat().st_size for f in files)
        max_bytes = max_size_mb * 1024 * 1024
        removed = 0

        while total > max_bytes and files:
            oldest = files.pop(0)
            size = oldest.stat().st_size
            if self.dry_run:
                logger.info("[DRY-RUN] Would remove cache file %s", oldest)
            else:
                oldest.unlink()
                logger.info("Removed cache file %s", oldest)
            total -= size
            removed += 1

        return removed

    def _archive(self, eid: str, data_dir: Path) -> int:
        archive_path = self.data_root / "archive" / f"{eid}.tar.gz"
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
        archive_path = self.data_root / "archive" / f"{eid}.tar.gz"

        if self.dry_run:
            logger.info("[DRY-RUN] Would delete %s", eid)
            return 1

        if data_dir.exists():
            shutil.rmtree(data_dir)
        if archive_path.exists():
            archive_path.unlink()

        if self.registry is not None:
            self.registry.remove(eid)
        logger.info("Deleted all data for %s", eid)
        return 1


def _human_bytes(n: int) -> str:
    """Convert bytes to human-readable string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"
