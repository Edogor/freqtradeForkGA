"""Lease-aware scheduler for immutable standard-evolution V2 workers."""

from __future__ import annotations

import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

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
    WorkerBindingV2,
    WorkerKind,
)
from genetic_algorithm.orchestration.evolution_worker_v2 import (
    PreparedEvolutionWorkerV2,
    evolution_worker_argv,
    load_evolution_worker,
)
from genetic_algorithm.orchestration.result_contract import AttemptManifestV2
from genetic_algorithm.orchestration.shadow_scheduler_v2 import (
    SchedulerFutureErrorV2,
    ShadowSchedulerConfigV2,
    ShadowSchedulerTickV2,
)


class EvolutionSchedulerError(RuntimeError):
    """Raised when the evolution scheduler cannot safely make progress."""


class EvolutionSchedulerTimeout(EvolutionSchedulerError):
    """Raised when run_until_idle reaches its explicit time budget."""


def _utc_now() -> datetime:
    return datetime.now(UTC)


def queue_prepared_evolution_attempt(
    state_store: AttemptStateStoreV2,
    *,
    manifest: AttemptManifestV2,
    prepared: PreparedEvolutionWorkerV2,
    bound_at: datetime,
    validated_at: datetime,
    queued_at: datetime,
    priority: int = 0,
    retry_number: int = 0,
    actor: str = "evolution-controller-v2",
    python_executable: str | Path = sys.executable,
    working_directory: str | Path,
) -> AttemptStateV2:
    """Register, bind, validate, and queue one immutable evolution attempt."""

    if not bound_at <= validated_at <= queued_at:
        raise EvolutionSchedulerError("bind/validate/queue timestamps are not monotonic")
    loaded = load_evolution_worker(
        prepared.spec_path,
        expected_spec_sha256=prepared.spec_file_sha256,
    )
    if loaded.manifest != manifest or prepared.spec.attempt_id != manifest.attempt_id:
        raise EvolutionSchedulerError("prepared evolution worker differs from manifest")
    worker_kind = WorkerKind(prepared.spec.worker_kind)
    if worker_kind not in {
        WorkerKind.STANDARD_EVOLUTION,
        WorkerKind.GENERIC_ISLAND_EVOLUTION,
    }:
        raise EvolutionSchedulerError("prepared worker is not an evolution worker")
    binding = WorkerBindingV2(
        worker_kind=worker_kind,
        argv=evolution_worker_argv(prepared, python_executable=python_executable),
        working_directory=str(Path(working_directory).resolve()),
        spec_path=str(prepared.spec_path.resolve()),
        spec_sha256=prepared.spec_file_sha256,
    )
    state = state_store.register(
        manifest,
        priority=priority,
        retry_number=retry_number,
        actor=actor,
    )
    state = state_store.bind_worker(
        manifest.attempt_id,
        binding,
        bound_at=bound_at,
        actor=actor,
    )
    if state.status == AttemptLifecycleStatus.DRAFT:
        state = state_store.transition(
            manifest.attempt_id,
            AttemptLifecycleStatus.VALIDATED,
            occurred_at=validated_at,
            actor=actor,
            reason="EVOLUTION_WORKER_INPUTS_VALID",
        )
    if state.status == AttemptLifecycleStatus.VALIDATED:
        state = state_store.transition(
            manifest.attempt_id,
            AttemptLifecycleStatus.QUEUED,
            occurred_at=queued_at,
            actor=actor,
            reason=f"QUEUED_FOR_{worker_kind.value}",
        )
    if state.status != AttemptLifecycleStatus.QUEUED:
        raise EvolutionSchedulerError(
            f"attempt cannot be queued from current state: {state.status.value}"
        )
    return state


class EvolutionSchedulerV2:
    """Run canonical evolution attempts without the legacy JSON registry."""

    _ACTIVE = {
        AttemptLifecycleStatus.CLAIMED,
        AttemptLifecycleStatus.RUNNING,
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
            raise EvolutionSchedulerError("scheduler_id cannot be empty")
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

    def __enter__(self) -> EvolutionSchedulerV2:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close(wait=True)

    def close(self, *, wait: bool = True) -> None:
        if self._closed:
            return
        self._pool.shutdown(wait=wait, cancel_futures=False)
        self._closed = True

    def queue_evolution(
        self,
        *,
        manifest: AttemptManifestV2,
        prepared: PreparedEvolutionWorkerV2,
        bound_at: datetime,
        validated_at: datetime,
        queued_at: datetime,
        priority: int = 0,
        retry_number: int = 0,
        working_directory: str | Path,
    ) -> AttemptStateV2:
        if self._closed:
            raise EvolutionSchedulerError("scheduler is closed")
        return queue_prepared_evolution_attempt(
            self.state_store,
            manifest=manifest,
            prepared=prepared,
            bound_at=bound_at,
            validated_at=validated_at,
            queued_at=queued_at,
            priority=priority,
            retry_number=retry_number,
            actor=self.scheduler_id,
            python_executable=self.config.python_executable,
            working_directory=working_directory,
        )

    def run_once(self, *, observed_at: datetime | None = None) -> ShadowSchedulerTickV2:
        if self._closed:
            raise EvolutionSchedulerError("scheduler is closed")
        now = observed_at or _utc_now()
        reconciliation = self.reconciler.reconcile_expired(as_of=now)
        completed, future_errors = self._reap_futures()
        active_before = self.state_store.count_attempts(
            statuses=self._ACTIVE,
            worker_kind=WorkerKind.STANDARD_EVOLUTION,
        )
        available = max(0, self.config.max_concurrent - active_before)
        launched: list[str] = []
        for _ in range(available):
            self._worker_sequence += 1
            worker_id = f"{self.scheduler_id}:worker-{self._worker_sequence}"
            claimed = self.state_store.claim_next(
                worker_id=worker_id,
                claimed_at=_utc_now(),
                lease_seconds=self.config.lease_seconds,
                worker_kind=WorkerKind.STANDARD_EVOLUTION,
            )
            if claimed is None:
                break
            future = self._pool.submit(self._execute_claimed, claimed)
            self._futures[future] = claimed.attempt_id
            launched.append(claimed.attempt_id)

        active_after = self.state_store.count_attempts(
            statuses=self._ACTIVE,
            worker_kind=WorkerKind.STANDARD_EVOLUTION,
        )
        queued = self.state_store.count_attempts(
            statuses={AttemptLifecycleStatus.QUEUED},
            worker_kind=WorkerKind.STANDARD_EVOLUTION,
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
            raise EvolutionSchedulerError("timeout_seconds must be positive")
        deadline = time.monotonic() + timeout_seconds
        ticks: list[ShadowSchedulerTickV2] = []
        while True:
            tick = self.run_once()
            ticks.append(tick)
            if tick.queued_count == 0 and tick.active_count == 0 and not self._futures:
                return ticks
            if time.monotonic() >= deadline:
                raise EvolutionSchedulerTimeout(
                    "scheduler did not reach idle before timeout; "
                    f"queued={tick.queued_count}, active={tick.active_count}"
                )
            time.sleep(self.config.poll_interval_seconds)

    def _execute_claimed(self, claimed: AttemptStateV2) -> AttemptStateV2:
        binding = claimed.worker_binding
        if binding is None or binding.worker_kind != WorkerKind.STANDARD_EVOLUTION:
            raise AttemptStateError("claimed evolution attempt lacks its worker binding")
        command = AttemptCommandV2(
            argv=binding.argv,
            working_directory=binding.working_directory,
        )
        executor = AttemptExecutorV2(
            self.state_store,
            worker_id=str(claimed.claimed_by),
            lease_seconds=self.config.lease_seconds,
            heartbeat_interval_seconds=self.config.heartbeat_interval_seconds,
            max_execution_seconds=self.config.max_execution_seconds,
            process_inspector=self.process_inspector,
        )
        return executor.execute_attempt(claimed, command)

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
