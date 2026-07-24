"""Operational daemon for the canonical SQLite-backed V2 attempt queue."""

from __future__ import annotations

import logging
import os
import signal
import time
from pathlib import Path
from typing import Any

from genetic_algorithm.orchestration.attempt_scheduler_v2 import AttemptSchedulerV2
from genetic_algorithm.orchestration.attempt_state_v2 import (
    AttemptLifecycleStatus,
    AttemptStateStoreV2,
)
from genetic_algorithm.orchestration.runner_v2 import default_state_path
from genetic_algorithm.orchestration.shadow_scheduler_v2 import ShadowSchedulerConfigV2


logger = logging.getLogger(__name__)

_PID_FILE = Path("genetic_algorithm/logs/scheduler.pid")
_MIN_FREE_MEMORY_MB = 1500


class RunScheduler:
    """Poll and execute immutable V2 attempts through one canonical executor."""

    _ACTIVE = {
        AttemptLifecycleStatus.CLAIMED,
        AttemptLifecycleStatus.RUNNING,
    }
    _TERMINAL = {
        AttemptLifecycleStatus.SUCCEEDED,
        AttemptLifecycleStatus.FAILED,
        AttemptLifecycleStatus.INTERRUPTED,
        AttemptLifecycleStatus.INVALID_RESULT,
    }

    def __init__(
        self,
        max_concurrent: int = 5,
        persistent: bool = False,
        poll_interval: float = 0.25,
        min_free_memory_mb: int = _MIN_FREE_MEMORY_MB,
        *,
        state_path: str | Path | None = None,
        lease_seconds: int = 30,
        heartbeat_interval_seconds: float = 5.0,
    ) -> None:
        self.max_concurrent = max_concurrent
        self.persistent = persistent
        self.poll_interval = poll_interval
        self.min_free_memory_mb = min_free_memory_mb
        self.state_store = AttemptStateStoreV2(state_path or default_state_path())
        self.config = ShadowSchedulerConfigV2(
            max_concurrent=max_concurrent,
            lease_seconds=lease_seconds,
            heartbeat_interval_seconds=heartbeat_interval_seconds,
            poll_interval_seconds=poll_interval,
        )
        self._running = True
        self._launched_count = 0
        self._completed_count = 0
        self._failed_count = 0

    def run(self) -> None:
        """Run until the queue drains, or keep polling in persistent mode."""

        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)
        _PID_FILE.parent.mkdir(parents=True, exist_ok=True)
        _PID_FILE.write_text(str(os.getpid()))
        logger.info(
            "V2 scheduler started (pid=%d, max_concurrent=%d, state=%s)",
            os.getpid(),
            self.max_concurrent,
            self.state_store.path,
        )
        seen_terminal: set[str] = set()
        try:
            with AttemptSchedulerV2(
                self.state_store,
                scheduler_id=f"operational-scheduler:{os.getpid()}",
                config=self.config,
            ) as scheduler:
                while self._running:
                    if not self._check_memory():
                        logger.warning(
                            "Insufficient free memory (<%d MB); deferring claims",
                            self.min_free_memory_mb,
                        )
                        time.sleep(self.poll_interval)
                        continue
                    tick = scheduler.run_once()
                    self._launched_count += len(tick.launched_attempt_ids)
                    for error in tick.future_errors:
                        logger.error(
                            "Attempt future failed outside its result contract: %s: %s",
                            error.attempt_id,
                            error.detail,
                        )
                    terminal = self.state_store.list_attempts(statuses=self._TERMINAL)
                    for state in terminal:
                        if state.attempt_id in seen_terminal:
                            continue
                        seen_terminal.add(state.attempt_id)
                        if state.status == AttemptLifecycleStatus.SUCCEEDED:
                            self._completed_count += 1
                        else:
                            self._failed_count += 1
                    if (
                        not self.persistent
                        and tick.queued_count == 0
                        and tick.active_count == 0
                        and not scheduler.has_pending_futures
                    ):
                        break
                    time.sleep(self.poll_interval)
        finally:
            _PID_FILE.unlink(missing_ok=True)
            logger.info("V2 scheduler stopped.")

    def _handle_signal(self, signum, frame) -> None:
        logger.info("Received signal %d, shutting down after active workers finish", signum)
        self._running = False

    @staticmethod
    def stop_daemon() -> None:
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
    def stats(self) -> dict[str, Any]:
        return {
            "active": self.state_store.count_attempts(statuses=self._ACTIVE),
            "max_concurrent": self.max_concurrent,
            "launched": self._launched_count,
            "completed": self._completed_count,
            "failed": self._failed_count,
            "queued": self.state_store.count_attempts(
                statuses={AttemptLifecycleStatus.QUEUED}
            ),
            "persistent": self.persistent,
            "state_path": str(self.state_store.path),
        }

    def _check_memory(self) -> bool:
        try:
            with Path("/proc/meminfo").open() as handle:
                for line in handle:
                    if line.startswith("MemAvailable:"):
                        kb = int(line.split()[1])
                        return kb // 1024 >= self.min_free_memory_mb
        except (OSError, ValueError, IndexError):
            pass
        return True
