"""One lease-aware scheduler for every canonical V2 worker kind."""

from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import UTC, datetime

from genetic_algorithm.orchestration.attempt_executor_v2 import (
    AttemptCommandV2,
    AttemptExecutorV2,
    AttemptReconcilerV2,
    LinuxProcessInspector,
    ProcessInspector,
)
from genetic_algorithm.orchestration.attempt_state_v2 import (
    AttemptLifecycleStatus,
    AttemptStateError,
    AttemptStateStoreV2,
    AttemptStateV2,
    WorkerKind,
)
from genetic_algorithm.orchestration.shadow_scheduler_v2 import (
    SchedulerFutureErrorV2,
    ShadowSchedulerConfigV2,
    ShadowSchedulerTickV2,
)


class AttemptSchedulerError(RuntimeError):
    """Raised when the canonical scheduler cannot safely progress."""


class AttemptSchedulerTimeout(AttemptSchedulerError):
    """Raised when a bounded scheduler run does not become idle."""


def _utc_now() -> datetime:
    return datetime.now(UTC)


class AttemptSchedulerV2:
    """Claim all supported worker kinds from one SQLite queue.

    Slot accounting includes claims/runs adopted from earlier scheduler
    instances.  The scheduler owns lifecycle state; children only consume
    immutable specs and commit result artifacts.
    """

    _ACTIVE = {
        AttemptLifecycleStatus.CLAIMED,
        AttemptLifecycleStatus.RUNNING,
    }
    _SUPPORTED = {
        WorkerKind.SHADOW_REPLAY,
        WorkerKind.STANDARD_EVOLUTION,
        WorkerKind.GENERIC_ISLAND_EVOLUTION,
    }

    def __init__(
        self,
        state_store: AttemptStateStoreV2,
        *,
        scheduler_id: str,
        config: ShadowSchedulerConfigV2 | None = None,
        process_inspector: ProcessInspector | None = None,
    ) -> None:
        if not scheduler_id:
            raise AttemptSchedulerError("scheduler_id cannot be empty")
        self.state_store = state_store
        self.scheduler_id = scheduler_id
        self.config = config or ShadowSchedulerConfigV2()
        self.process_inspector = process_inspector or LinuxProcessInspector()
        self.reconciler = AttemptReconcilerV2(
            state_store,
            process_inspector=self.process_inspector,
            actor=f"{scheduler_id}:reconciler",
        )
        self._pool = ThreadPoolExecutor(
            max_workers=self.config.max_concurrent,
            thread_name_prefix=f"{scheduler_id}-worker",
        )
        self._futures: dict[Future[AttemptStateV2], str] = {}
        self._worker_sequence = 0
        self._closed = False

    def __enter__(self) -> AttemptSchedulerV2:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close(wait=True)

    def close(self, *, wait: bool = True) -> None:
        if self._closed:
            return
        self._pool.shutdown(wait=wait, cancel_futures=False)
        self._closed = True

    @property
    def has_pending_futures(self) -> bool:
        return bool(self._futures)

    def run_once(self, *, observed_at: datetime | None = None) -> ShadowSchedulerTickV2:
        if self._closed:
            raise AttemptSchedulerError("scheduler is closed")
        now = observed_at or _utc_now()
        reconciliation = self.reconciler.reconcile_expired(as_of=now)
        completed, future_errors = self._reap_futures()
        active_before = self.state_store.count_attempts(statuses=self._ACTIVE)
        available = max(0, self.config.max_concurrent - active_before)
        launched: list[str] = []
        for _ in range(available):
            self._worker_sequence += 1
            worker_id = f"{self.scheduler_id}:worker-{self._worker_sequence}"
            claimed = self.state_store.claim_next(
                worker_id=worker_id,
                claimed_at=_utc_now(),
                lease_seconds=self.config.lease_seconds,
            )
            if claimed is None:
                break
            binding = claimed.worker_binding
            if binding is None or binding.worker_kind not in self._SUPPORTED:
                raise AttemptStateError("queue contains an unsupported worker binding")
            future = self._pool.submit(self._execute_claimed, claimed)
            self._futures[future] = claimed.attempt_id
            launched.append(claimed.attempt_id)

        active_after = self.state_store.count_attempts(statuses=self._ACTIVE)
        queued = self.state_store.count_attempts(
            statuses={AttemptLifecycleStatus.QUEUED}
        )
        return ShadowSchedulerTickV2(
            observed_at=now,
            queued_count=queued,
            active_count=active_after,
            available_slots=max(0, self.config.max_concurrent - active_after),
            launched_attempt_ids=launched,
            completed_attempt_ids=completed,
            future_errors=future_errors,
            reconciliation=reconciliation,
        )

    def run_until_idle(self, *, timeout_seconds: float = 60.0) -> list[ShadowSchedulerTickV2]:
        if timeout_seconds <= 0:
            raise AttemptSchedulerError("timeout_seconds must be positive")
        deadline = time.monotonic() + timeout_seconds
        ticks: list[ShadowSchedulerTickV2] = []
        while True:
            tick = self.run_once()
            ticks.append(tick)
            if tick.queued_count == 0 and tick.active_count == 0 and not self._futures:
                return ticks
            if time.monotonic() >= deadline:
                raise AttemptSchedulerTimeout(
                    "scheduler did not reach idle before timeout; "
                    f"queued={tick.queued_count}, active={tick.active_count}"
                )
            time.sleep(self.config.poll_interval_seconds)

    def _execute_claimed(self, claimed: AttemptStateV2) -> AttemptStateV2:
        binding = claimed.worker_binding
        if binding is None or binding.worker_kind not in self._SUPPORTED:
            raise AttemptStateError("claimed attempt lacks a supported worker binding")
        executor = AttemptExecutorV2(
            self.state_store,
            worker_id=str(claimed.claimed_by),
            lease_seconds=self.config.lease_seconds,
            heartbeat_interval_seconds=self.config.heartbeat_interval_seconds,
            max_execution_seconds=self.config.max_execution_seconds,
            process_inspector=self.process_inspector,
        )
        return executor.execute_attempt(
            claimed,
            AttemptCommandV2(
                argv=binding.argv,
                working_directory=binding.working_directory,
            ),
        )

    def _reap_futures(self) -> tuple[list[str], list[SchedulerFutureErrorV2]]:
        completed: list[str] = []
        errors: list[SchedulerFutureErrorV2] = []
        for future, attempt_id in list(self._futures.items()):
            if not future.done():
                continue
            del self._futures[future]
            try:
                future.result()
                completed.append(attempt_id)
            except Exception as exc:
                errors.append(
                    SchedulerFutureErrorV2(
                        attempt_id=attempt_id,
                        error_type=type(exc).__name__,
                        detail=str(exc)[:2000] or type(exc).__name__,
                    )
                )
        return sorted(completed), errors
