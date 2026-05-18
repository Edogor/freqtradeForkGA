"""
RunScheduler — queue-based experiment scheduler.

Replaces ``ga_auto_queue_v2.sh`` with a Python daemon that manages
concurrent GA runs, respects slot limits, and updates the registry.

Usage::

    scheduler = RunScheduler(max_concurrent=5, persistent=True)
    scheduler.run()  # blocking daemon loop
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from genetic_algorithm.orchestration.registry import ExperimentRegistry

logger = logging.getLogger(__name__)

_PID_FILE = Path("genetic_algorithm/logs/scheduler.pid")
_MIN_FREE_MEMORY_MB = 1500


class RunScheduler:
    """Daemon that polls the registry for queued experiments and launches them."""

    def __init__(
        self,
        max_concurrent: int = 5,
        persistent: bool = False,
        poll_interval: int = 30,
        min_free_memory_mb: int = _MIN_FREE_MEMORY_MB,
    ) -> None:
        self.max_concurrent = max_concurrent
        self.persistent = persistent
        self.poll_interval = poll_interval
        self.min_free_memory_mb = min_free_memory_mb
        self.registry = ExperimentRegistry()
        self._running = True
        self._child_pids: dict[str, int] = {}  # experiment_id → pid
        self._launched_count = 0
        self._completed_count = 0
        self._failed_count = 0

    def run(self) -> None:
        """Main daemon loop.  Blocks until interrupted or no work left."""
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

        _PID_FILE.parent.mkdir(parents=True, exist_ok=True)
        _PID_FILE.write_text(str(os.getpid()))
        logger.info(
            "Scheduler started (pid=%d, max_concurrent=%d, persistent=%s)",
            os.getpid(), self.max_concurrent, self.persistent,
        )

        try:
            while self._running:
                self._reap_finished()
                self._launch_next()

                queued_count = self.registry.count(status="queued")
                running_count = len(self._child_pids)

                if not self.persistent and queued_count == 0 and running_count == 0:
                    logger.info("Queue empty, no running experiments — exiting.")
                    break

                time.sleep(self.poll_interval)
        finally:
            _PID_FILE.unlink(missing_ok=True)
            logger.info("Scheduler stopped.")

    def _launch_next(self) -> None:
        """Launch the next queued experiment if a slot is available."""
        running_count = len(self._child_pids)
        slots = self.max_concurrent - running_count
        if slots <= 0:
            return

        if not self._check_memory():
            logger.warning(
                "[SCHEDULER] Insufficient free memory (<%d MB); deferring launch",
                self.min_free_memory_mb,
            )
            return

        queued = self.registry.list(status="queued", limit=slots)
        for exp in queued:
            eid = exp["experiment_id"]
            config_path = exp.get("config_path")
            if not config_path or not Path(config_path).exists():
                self.registry.fail(eid, error=f"Config not found: {config_path}")
                continue

            tags_args = []
            for tag in exp.get("tags", []):
                tags_args.extend(["--tag", tag])

            cmd = [
                sys.executable, "-m", "genetic_algorithm", "run",
                config_path, "--name", eid, "--yes", "--no-monitor",
                *tags_args,
            ]
            logger.info("Launching %s: %s", eid, " ".join(cmd))

            log_path = Path(f"genetic_algorithm/logs/{eid}.log")
            log_path.parent.mkdir(parents=True, exist_ok=True)

            with open(log_path, "w") as log_fh:
                proc = subprocess.Popen(
                    cmd,
                    stdout=log_fh,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )

            self._child_pids[eid] = proc.pid
            self._launched_count += 1
            self.registry.start(
                eid,
                pid=proc.pid,
                log_path=str(log_path),
            )

    def _reap_finished(self) -> None:
        """Check child processes and update registry for finished ones."""
        finished = []
        for eid, pid in self._child_pids.items():
            try:
                result = os.waitpid(pid, os.WNOHANG)
                if result[0] != 0:
                    exit_code = os.WEXITSTATUS(result[1]) if os.WIFEXITED(result[1]) else -1
                    finished.append((eid, exit_code))
            except ChildProcessError:
                finished.append((eid, -1))

        for eid, exit_code in finished:
            del self._child_pids[eid]
            exp = self.registry.get(eid)
            if exp and exp["status"] == "running":
                if exit_code == 0:
                    logger.info("Experiment %s finished (exit=0)", eid)
                    self._completed_count += 1
                    # run command already calls registry.complete()
                else:
                    logger.warning("Experiment %s exited with code %d", eid, exit_code)
                    self._failed_count += 1
                    self.registry.fail(eid, error=f"Exit code: {exit_code}")

    def _handle_signal(self, signum, frame) -> None:
        logger.info("Received signal %d, shutting down...", signum)
        self._running = False

    @staticmethod
    def stop_daemon() -> None:
        """Send SIGTERM to the running scheduler daemon."""
        if not _PID_FILE.exists():
            print("No scheduler running (PID file not found).")
            return
        pid = int(_PID_FILE.read_text().strip())
        try:
            os.kill(pid, signal.SIGTERM)
            print(f"Sent SIGTERM to scheduler (pid={pid}).")
        except ProcessLookupError:
            print(f"Scheduler (pid={pid}) not running. Cleaning up PID file.")
            _PID_FILE.unlink(missing_ok=True)

    @property
    def stats(self) -> Dict[str, Any]:
        """Current scheduler statistics."""
        return {
            "active": len(self._child_pids),
            "max_concurrent": self.max_concurrent,
            "launched": self._launched_count,
            "completed": self._completed_count,
            "failed": self._failed_count,
            "queued": self.registry.count(status="queued"),
            "persistent": self.persistent,
        }

    def _check_memory(self) -> bool:
        """Check if there is sufficient free memory to launch a new run."""
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        kb = int(line.split()[1])
                        return kb // 1024 >= self.min_free_memory_mb
        except (OSError, ValueError, IndexError):
            pass
        # If we can't determine memory, allow launch
        return True

    def recover_active(self) -> int:
        """Recover tracking of runs still active in the registry.

        Call on scheduler restart to re-adopt running experiments
        launched by a previous scheduler instance.

        Returns the number of recovered processes.
        """
        running = self.registry.list(status="running")
        recovered = 0
        for entry in running:
            pid = entry.get("pid")
            exp_id = entry["experiment_id"]
            if pid and self._is_process_alive(pid):
                self._child_pids[exp_id] = pid
                recovered += 1
                logger.info(
                    "[SCHEDULER] Recovered active run: %s (PID %d)", exp_id, pid
                )
            else:
                self.registry.fail(exp_id, error="Process not found on recovery")
                self._failed_count += 1
                logger.warning(
                    "[SCHEDULER] Stale run on recovery: %s", exp_id
                )
        return recovered

    @staticmethod
    def _is_process_alive(pid: int) -> bool:
        """Check if a process is still running."""
        try:
            os.kill(pid, 0)
            return True
        except (ProcessLookupError, PermissionError):
            return False
