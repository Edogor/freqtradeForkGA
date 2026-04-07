"""Tests for orchestration modules: scheduler, monitor, lifecycle."""

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from genetic_algorithm.orchestration.monitor import (
    ExperimentMetrics,
    ExperimentMonitor,
    detect_status,
    extract_metrics,
    _format_duration,
)
from genetic_algorithm.orchestration.lifecycle import DataLifecycle, _human_bytes
from genetic_algorithm.orchestration.scheduler import RunScheduler


# =====================================================================
# Helpers
# =====================================================================


@pytest.fixture
def tmp_registry(tmp_path):
    """Create a temporary ExperimentRegistry."""
    from genetic_algorithm.orchestration.registry import ExperimentRegistry

    reg = ExperimentRegistry(path=tmp_path / "registry.json")
    return reg


@pytest.fixture
def sample_log(tmp_path):
    """Create a sample GA log file."""
    log = tmp_path / "test_exp.log"
    log.write_text(
        "2024-01-01 00:00:00 [INFO] Starting GA run\n"
        "2024-01-01 00:01:00 [INFO] GENERATION 1/10\n"
        "2024-01-01 00:01:30 [STATS] Best: 0.5234 Avg: 0.3100\n"
        "2024-01-01 00:02:00 [INFO] diversity=0.7500\n"
        "2024-01-01 00:02:30 [INFO] GENERATION 2/10\n"
        "2024-01-01 00:03:00 [STATS] Best: 0.6123 Avg: 0.4200\n"
        "2024-01-01 00:03:30 [NEW BEST] Individual fitness=0.6123 profit=12.5%\n"
        "2024-01-01 00:04:00 [INFO] diversity=0.6800\n"
        "2024-01-01 00:04:30 [EVAL] Progress: 45/50\n"
    )
    return log


@pytest.fixture
def completed_log(tmp_path):
    """Create a log file that shows completion."""
    log = tmp_path / "done_exp.log"
    log.write_text(
        "2024-01-01 00:00:00 [INFO] Starting GA run\n"
        "2024-01-01 01:00:00 [INFO] GENERATION 10/10\n"
        "2024-01-01 01:00:30 [STATS] Best: 0.8500 Avg: 0.6000\n"
        "2024-01-01 01:01:00 [INFO] GA RUN COMPLETE\n"
    )
    return log


@pytest.fixture
def island_log(tmp_path):
    """Create a log file with island model format."""
    log = tmp_path / "island_exp.log"
    log.write_text(
        "2024-01-01 00:00:00 [INFO] Starting island model\n"
        "[SUMMARY] Gen 5/20 (elapsed 30m): island_0=0.45, island_1=0.62, island_2=0.58\n"
        "[island_0 ] best=0.45 avg=0.30\n"
        "[island_1 ] best=0.62 avg=0.41\n"
    )
    return log


# =====================================================================
# Monitor — extract_metrics
# =====================================================================


class TestExtractMetrics:
    def test_plain_ga_log(self, sample_log):
        m = extract_metrics(sample_log)
        assert m.generation == 2
        assert m.total_generations == 10
        assert m.best_fitness == pytest.approx(0.6123)
        assert m.avg_fitness == pytest.approx(0.4200)
        assert m.best_profit == pytest.approx(12.5)
        assert m.eval_progress == "45/50"

    def test_completed_log(self, completed_log):
        m = extract_metrics(completed_log)
        assert m.generation == 10
        assert m.total_generations == 10
        assert m.best_fitness == pytest.approx(0.8500)
        assert m.status == "DONE"

    def test_island_log(self, island_log):
        m = extract_metrics(island_log)
        assert m.generation == 5
        assert m.total_generations == 20
        # Best across islands
        assert m.best_fitness == pytest.approx(0.62)

    def test_nonexistent_log(self, tmp_path):
        m = extract_metrics(tmp_path / "nonexistent.log")
        assert m.generation is None
        assert m.best_fitness is None

    def test_empty_log(self, tmp_path):
        log = tmp_path / "empty.log"
        log.write_text("")
        m = extract_metrics(log)
        assert m.generation is None

    def test_diversity_extraction(self, sample_log):
        m = extract_metrics(sample_log)
        assert m.diversity == pytest.approx(0.68)

    def test_error_count(self, tmp_path):
        log = tmp_path / "errors.log"
        log.write_text(
            "[ERROR] boom\n[ERROR] crash\nTraceback (most recent call)\n"
        )
        m = extract_metrics(log)
        assert m.error_count == 3


# =====================================================================
# Monitor — detect_status
# =====================================================================


class TestDetectStatus:
    def test_completed_marker(self, completed_log):
        assert detect_status(completed_log) == "DONE"

    def test_running_process(self, sample_log):
        # Fake that our own PID is alive
        assert detect_status(sample_log, pid=os.getpid()) == "RUNNING"

    def test_dead_process(self, sample_log):
        # Use a PID that doesn't exist
        assert detect_status(sample_log, pid=99999999) in ("CRASHED", "STALE")

    def test_stale_log(self, tmp_path):
        log = tmp_path / "stale.log"
        log.write_text("old data\n")
        # Set mtime to 1 hour ago
        old_time = time.time() - 3600
        os.utime(log, (old_time, old_time))
        assert detect_status(log, stale_seconds=600) == "STALE"

    def test_nonexistent(self, tmp_path):
        assert detect_status(tmp_path / "nope.log") == "UNKNOWN"


# =====================================================================
# Monitor — ExperimentMonitor
# =====================================================================


class TestExperimentMonitor:
    def test_get_metrics_empty(self, tmp_registry):
        mon = ExperimentMonitor()
        mon.registry = tmp_registry
        metrics = mon.get_metrics()
        assert metrics == []

    def test_get_metrics_running(self, tmp_registry, sample_log):
        tmp_registry.register("exp1", config_path="test.yaml")
        started = datetime.utcnow().isoformat()
        tmp_registry.start("exp1", pid=os.getpid(), log_path=str(sample_log))

        mon = ExperimentMonitor()
        mon.registry = tmp_registry
        metrics = mon.get_metrics()

        assert len(metrics) == 1
        assert metrics[0].experiment_id == "exp1"
        assert metrics[0].status == "RUNNING"
        assert metrics[0].generation == 2

    def test_snapshot_no_crash(self, tmp_registry, capsys):
        mon = ExperimentMonitor()
        mon.registry = tmp_registry
        mon.snapshot()
        captured = capsys.readouterr()
        assert "GA Monitor" in captured.out


# =====================================================================
# Monitor — _format_duration
# =====================================================================


class TestFormatDuration:
    def test_seconds(self):
        assert _format_duration(90) == "00:01:30"

    def test_hours(self):
        assert _format_duration(7200) == "02:00:00"

    def test_days(self):
        assert _format_duration(90000) == "1d 01:00"

    def test_negative(self):
        assert _format_duration(-5) == "00:00:00"


# =====================================================================
# Scheduler
# =====================================================================


class TestRunScheduler:
    def test_init_defaults(self, tmp_registry):
        with patch(
            "genetic_algorithm.orchestration.scheduler.ExperimentRegistry",
            return_value=tmp_registry,
        ):
            s = RunScheduler()
            assert s.max_concurrent == 5
            assert s.persistent is False
            assert s._launched_count == 0

    def test_stats(self, tmp_registry):
        s = RunScheduler(max_concurrent=3)
        s.registry = tmp_registry
        stats = s.stats
        assert stats["max_concurrent"] == 3
        assert stats["active"] == 0
        assert stats["queued"] == 0

    def test_request_shutdown(self, tmp_registry):
        s = RunScheduler()
        s.registry = tmp_registry
        assert s._running is True
        # Signal handler sets _running = False
        s._handle_signal(2, None)
        assert s._running is False

    def test_check_memory(self, tmp_registry):
        s = RunScheduler()
        s.registry = tmp_registry
        # On Linux this should return True or False based on actual memory
        result = s._check_memory()
        assert isinstance(result, bool)

    def test_is_process_alive(self):
        assert RunScheduler._is_process_alive(os.getpid()) is True
        assert RunScheduler._is_process_alive(99999999) is False

    def test_recover_active_no_running(self, tmp_registry):
        s = RunScheduler()
        s.registry = tmp_registry
        assert s.recover_active() == 0

    def test_recover_active_with_stale(self, tmp_registry):
        tmp_registry.register("stale_exp", config_path="test.yaml")
        tmp_registry.start("stale_exp", pid=99999999)

        s = RunScheduler()
        s.registry = tmp_registry
        recovered = s.recover_active()
        assert recovered == 0  # process is dead
        assert s._failed_count == 1

        # Verify it was marked as failed
        exp = tmp_registry.get("stale_exp")
        assert exp["status"] == "failed"

    def test_recover_active_with_live(self, tmp_registry):
        tmp_registry.register("live_exp", config_path="test.yaml")
        tmp_registry.start("live_exp", pid=os.getpid())

        s = RunScheduler()
        s.registry = tmp_registry
        recovered = s.recover_active()
        assert recovered == 1
        assert "live_exp" in s._child_pids

    def test_reap_finished_no_children(self, tmp_registry):
        s = RunScheduler()
        s.registry = tmp_registry
        s._reap_finished()  # should not crash with empty dict

    def test_launch_next_respects_max(self, tmp_registry):
        s = RunScheduler(max_concurrent=0)
        s.registry = tmp_registry
        s._launch_next()  # should not launch anything
        assert len(s._child_pids) == 0


# =====================================================================
# Lifecycle
# =====================================================================


class TestDataLifecycle:
    def test_report_empty(self, tmp_path, tmp_registry):
        lc = DataLifecycle(data_root=tmp_path, dry_run=True)
        lc.registry = tmp_registry
        report = lc.report()
        assert report["total_bytes"] == 0
        assert report["file_count"] == 0
        assert "categories" in report

    def test_report_with_data(self, tmp_path, tmp_registry):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        (cache_dir / "entry1.pkl").write_bytes(b"x" * 1000)
        (cache_dir / "entry2.pkl").write_bytes(b"x" * 500)

        lc = DataLifecycle(data_root=tmp_path, dry_run=True)
        lc.registry = tmp_registry
        report = lc.report()
        assert report["categories"]["cache"] == 1500
        assert report["total_bytes"] >= 1500

    def test_run_dry_run(self, tmp_path, tmp_registry):
        # Register a completed experiment from 30 days ago
        tmp_registry.register("old_exp", config_path="test.yaml")
        tmp_registry.complete("old_exp")
        # Override finished_at to simulate an old experiment
        old_time = (datetime.utcnow() - timedelta(days=30)).isoformat()
        tmp_registry.update("old_exp", finished_at=old_time)

        # Create data dir
        data_dir = tmp_path / "old_exp"
        data_dir.mkdir()
        (data_dir / "result.json").write_text("{}")

        lc = DataLifecycle(
            data_root=tmp_path,
            archive_after_days=7,
            delete_after_days=90,
            dry_run=True,
        )
        lc.registry = tmp_registry
        result = lc.run()
        assert result["dry_run"] is True
        assert result["archived"] == 1
        # Verify file still exists (dry run)
        assert data_dir.exists()

    def test_archive_creates_tarball(self, tmp_path, tmp_registry):
        data_dir = tmp_path / "archive_me"
        data_dir.mkdir()
        (data_dir / "data.txt").write_text("hello")

        lc = DataLifecycle(data_root=tmp_path, dry_run=False)
        lc.registry = tmp_registry
        count = lc._archive("archive_me", data_dir)
        assert count == 1
        assert (tmp_path / "archive" / "archive_me.tar.gz").exists()
        assert not data_dir.exists()

    def test_delete_removes_data(self, tmp_path, tmp_registry):
        tmp_registry.register("del_exp", config_path="test.yaml")
        data_dir = tmp_path / "del_exp"
        data_dir.mkdir()
        (data_dir / "result.json").write_text("{}")

        lc = DataLifecycle(data_root=tmp_path, dry_run=False)
        lc.registry = tmp_registry
        count = lc._delete("del_exp", data_dir)
        assert count == 1
        assert not data_dir.exists()

    def test_cleanup_logs(self, tmp_path, tmp_registry):
        log_dir = tmp_path / "logs"
        log_dir.mkdir()

        old_log = log_dir / "old.log"
        old_log.write_text("old data")
        old_time = time.time() - 60 * 60 * 24 * 40  # 40 days ago
        os.utime(old_log, (old_time, old_time))

        new_log = log_dir / "new.log"
        new_log.write_text("new data")

        lc = DataLifecycle(data_root=tmp_path, dry_run=True)
        lc.registry = tmp_registry
        removed = lc.cleanup_logs(max_age_days=30)
        assert removed == 1

    def test_cleanup_cache(self, tmp_path, tmp_registry):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        # Create files that exceed 1KB total
        for i in range(5):
            (cache_dir / f"entry_{i}.pkl").write_bytes(b"x" * 500)

        lc = DataLifecycle(data_root=tmp_path, dry_run=False)
        lc.registry = tmp_registry
        removed = lc.cleanup_cache(max_size_mb=0.001)  # Very small limit
        assert removed > 0

    def test_human_bytes(self):
        assert _human_bytes(0) == "0.0 B"
        assert _human_bytes(1024) == "1.0 KB"
        assert _human_bytes(1024 * 1024) == "1.0 MB"
        assert _human_bytes(1024 ** 3) == "1.0 GB"
