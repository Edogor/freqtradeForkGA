"""
Run Manager — Multi-run lifecycle management for the GA engine.

Supports starting, stopping, pausing, and resuming multiple evolution runs.
Each run executes in its own subprocess via multiprocessing.Process.
Communication with running evolutions happens through:
  - EventBus events (status updates, generation data)  — one-way out
  - threading.Event flags (stop, pause)                 — control signals
  - queue.Queue (strategy injection)                    — one-way in
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from genetic_algorithm.web.event_bus import Event, EventType, SubprocessEventRelay, get_event_bus
from genetic_algorithm.web.models.run import RunStatus, RunSummary


logger = logging.getLogger(__name__)


@dataclass
class RunHandle:
    """Metadata + IPC handles for a single evolution run."""

    run_id: str
    status: RunStatus = RunStatus.PENDING
    config: Dict[str, Any] = field(default_factory=dict)
    config_name: str = ""

    # Process management
    process: Optional[mp.Process] = None
    pid: Optional[int] = None

    # IPC — control signals (multiprocessing-safe)
    stop_event: Optional[mp.Event] = None
    pause_event: Optional[mp.Event] = None
    injection_queue: Optional[mp.Queue] = None
    event_relay: Optional[Any] = None  # SubprocessEventRelay
    attempt_id: Optional[str] = None
    state_path: Optional[str] = None

    # Timing
    started_at: Optional[float] = None
    finished_at: Optional[float] = None

    # Cached latest state (updated via events)
    current_generation: int = 0
    total_generations: int = 0
    best_fitness: Optional[float] = None
    best_profit: Optional[float] = None
    best_individual_id: Optional[str] = None
    generation_stats: List[Dict[str, Any]] = field(default_factory=list)

    def to_summary(self) -> RunSummary:
        elapsed = None
        if self.started_at:
            end = self.finished_at or time.time()
            elapsed = end - self.started_at
        return RunSummary(
            run_id=self.run_id,
            status=self.status,
            config_name=self.config_name,
            current_generation=self.current_generation,
            total_generations=self.total_generations,
            best_fitness=self.best_fitness,
            best_profit=self.best_profit,
            population_size=self.config.get("genetic_algorithm", {}).get("population_size", 0),
            started_at=self.started_at,
            elapsed_seconds=elapsed,
            pairs=self.config.get("backtesting", {}).get("pairs", []),
        )


class RunManager:
    """
    Manages the lifecycle of multiple GA evolution runs.

    Thread-safe — can be called from FastAPI request handlers concurrently.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._runs: Dict[str, RunHandle] = {}
        self.bus = get_event_bus()

        # Subscribe to events to keep RunHandle state current
        self.bus.subscribe(self._on_event)

    # ── Public API ─────────────────────────────────────────────────

    def start_run(
        self,
        config: Dict[str, Any],
        run_id: Optional[str] = None,
        resume_from: Optional[str] = None,
    ) -> RunHandle:
        """
        Launch a new evolution run in a subprocess.

        Args:
            config: Full GA config dict.
            run_id: Optional custom run identifier; auto-generated if omitted.
            resume_from: Optional checkpoint path to resume from.

        Returns:
            RunHandle for the new run.
        """
        if resume_from is not None:
            raise ValueError(
                "canonical V2 web runs do not support resume; no legacy run was started"
            )
        run_id = run_id or f"run_{uuid.uuid4().hex[:8]}"
        canonical_config = {
            key: value for key, value in config.items() if not key.startswith("_")
        }
        from genetic_algorithm.orchestration.runner_v2 import (
            default_state_path,
            prepare_and_queue_standard_attempt,
        )

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            prefix="ga-web-v2-",
            delete=False,
        ) as config_handle:
            yaml.safe_dump(canonical_config, config_handle, sort_keys=True)
            config_path = Path(config_handle.name)
        try:
            prepared = prepare_and_queue_standard_attempt(
                config_path,
                experiment_name=run_id,
            )
        finally:
            config_path.unlink(missing_ok=True)
        state_path = default_state_path()

        stop_event = mp.Event()
        pause_event = mp.Event()
        injection_queue = mp.Queue(maxsize=100)

        handle = RunHandle(
            run_id=run_id,
            status=RunStatus.PENDING,
            config=config,
            config_name=config.get("_config_name", "custom"),
            stop_event=stop_event,
            pause_event=pause_event,
            injection_queue=injection_queue,
            total_generations=config.get("genetic_algorithm", {}).get("generations", 0),
            attempt_id=prepared.attempt_id,
            state_path=str(state_path),
        )

        with self._lock:
            self._runs[run_id] = handle

        # Publish creation event
        self.bus.publish(Event(
            type=EventType.RUN_CREATED,
            run_id=run_id,
            data={"config_name": handle.config_name},
        ))

        # Create event relay for subprocess → parent bridging
        relay = SubprocessEventRelay()
        relay.start()
        handle.event_relay = relay

        # Start the subprocess
        process = mp.Process(
            target=_run_canonical_attempt_worker,
            args=(run_id, prepared.attempt_id, str(state_path), relay.queue),
            name=f"ga-{run_id}",
            daemon=True,
        )
        process.start()

        with self._lock:
            handle.process = process
            handle.pid = process.pid
            handle.started_at = time.time()
            handle.status = RunStatus.RUNNING

        logger.info("Started run %s (PID %s)", run_id, process.pid)
        return handle

    def stop_run(self, run_id: str) -> bool:
        """Signal a run to stop gracefully (saves checkpoint, then exits)."""
        with self._lock:
            handle = self._runs.get(run_id)
            if not handle or handle.status not in (RunStatus.RUNNING, RunStatus.PAUSED):
                return False
            if handle.attempt_id is not None:
                logger.warning(
                    "Canonical V2 run %s cannot be stopped without a fenced cancellation contract",
                    run_id,
                )
                return False
            handle.status = RunStatus.STOPPING
            # Clear pause so the loop can reach the stop check
            if handle.pause_event:
                handle.pause_event.clear()
            if handle.stop_event:
                handle.stop_event.set()

        self.bus.publish(Event(type=EventType.RUN_STOPPED, run_id=run_id))
        logger.info("Stop signal sent to run %s", run_id)
        return True

    def pause_run(self, run_id: str) -> bool:
        """Pause a running evolution (blocks at top of generation loop)."""
        with self._lock:
            handle = self._runs.get(run_id)
            if not handle or handle.status != RunStatus.RUNNING:
                return False
            if handle.attempt_id is not None:
                return False
            if handle.pause_event:
                handle.pause_event.set()
            handle.status = RunStatus.PAUSED

        self.bus.publish(Event(type=EventType.RUN_PAUSED, run_id=run_id))
        logger.info("Pause signal sent to run %s", run_id)
        return True

    def resume_run(self, run_id: str) -> bool:
        """Resume a paused evolution."""
        with self._lock:
            handle = self._runs.get(run_id)
            if not handle or handle.status != RunStatus.PAUSED:
                return False
            if handle.attempt_id is not None:
                return False
            if handle.pause_event:
                handle.pause_event.clear()
            handle.status = RunStatus.RUNNING

        self.bus.publish(Event(type=EventType.RUN_RESUMED, run_id=run_id))
        logger.info("Resume signal sent to run %s", run_id)
        return True

    def inject_strategy(self, run_id: str, strategy_gene_dict: dict) -> bool:
        """Inject a strategy into a running evolution's population."""
        with self._lock:
            handle = self._runs.get(run_id)
            if not handle or handle.status not in (RunStatus.RUNNING, RunStatus.PAUSED):
                return False
            if handle.attempt_id is not None:
                return False
            if handle.injection_queue:
                try:
                    handle.injection_queue.put_nowait(strategy_gene_dict)
                except queue.Full:
                    logger.warning("Injection queue full for run %s", run_id)
                    return False

        self.bus.publish(Event(
            type=EventType.STRATEGY_INJECTED,
            run_id=run_id,
            data={"strategy_id": strategy_gene_dict.get("individual_id", "?")},
        ))
        return True

    def save_checkpoint(self, run_id: str) -> bool:
        """Request an immediate checkpoint save (next generation boundary)."""
        # We reuse the injection queue with a sentinel message
        with self._lock:
            handle = self._runs.get(run_id)
            if not handle or handle.status not in (RunStatus.RUNNING, RunStatus.PAUSED):
                return False
            if handle.attempt_id is not None:
                return False
            if handle.injection_queue:
                try:
                    handle.injection_queue.put_nowait({"_command": "checkpoint"})
                except queue.Full:
                    return False
        self.bus.publish(Event(
            type=EventType.CHECKPOINT_SAVED,
            run_id=run_id,
            data={"requested": True},
        ))
        return True

    def list_runs(self) -> List[RunSummary]:
        """Return summaries of all known runs (active + past)."""
        self._reap_finished()
        with self._lock:
            return [h.to_summary() for h in self._runs.values()]

    def get_run(self, run_id: str) -> Optional[RunHandle]:
        with self._lock:
            return self._runs.get(run_id)

    def get_run_ids(self) -> List[str]:
        with self._lock:
            return list(self._runs.keys())

    # ── Internal ───────────────────────────────────────────────────

    def _on_event(self, event: Event) -> None:
        """Update RunHandle state from incoming events."""
        with self._lock:
            handle = self._runs.get(event.run_id)
            if not handle:
                return

            if event.type == EventType.GENERATION_END:
                data = event.data
                handle.current_generation = data.get("generation", handle.current_generation)
                # Stats are now flattened at the top-level of data
                handle.generation_stats.append(data.get("_stats", data))
                # Prevent unbounded memory growth for long-running evolutions
                if len(handle.generation_stats) > 500:
                    handle.generation_stats = handle.generation_stats[-250:]
                best = data.get("best_individual", {})
                if best:
                    handle.best_individual_id = best.get("id")
                    metrics = best.get("metrics", {})
                    handle.best_profit = metrics.get("profit")
                bf = data.get("best_fitness")
                if bf is not None:
                    handle.best_fitness = bf

            elif event.type == EventType.NEW_BEST:
                ind = event.data.get("individual", {})
                handle.best_individual_id = ind.get("id")
                f = ind.get("fitness")
                if f is not None:
                    handle.best_fitness = f
                m = ind.get("metrics", {})
                if "profit" in m:
                    handle.best_profit = m["profit"]

            elif event.type == EventType.EVOLUTION_COMPLETE:
                handle.status = RunStatus.COMPLETED
                handle.finished_at = time.time()

            elif event.type == EventType.RUN_ERROR:
                handle.status = RunStatus.FAILED
                handle.finished_at = time.time()

    def _reap_finished(self) -> None:
        """Check for subprocess exits and update status."""
        with self._lock:
            for handle in self._runs.values():
                if handle.process and not handle.process.is_alive():
                    if handle.status in (RunStatus.RUNNING, RunStatus.PAUSED, RunStatus.STOPPING):
                        exit_code = handle.process.exitcode
                        if exit_code == 0 or handle.status == RunStatus.STOPPING:
                            handle.status = RunStatus.COMPLETED
                        else:
                            handle.status = RunStatus.FAILED
                        handle.finished_at = time.time()
                        # Stop the event relay thread
                        if handle.event_relay:
                            handle.event_relay.stop()
                            handle.event_relay = None


# ═══════════════════════════════════════════════════════════════════
# Subprocess worker
# ═══════════════════════════════════════════════════════════════════

def _run_canonical_attempt_worker(
    run_id: str,
    attempt_id: str,
    state_path: str,
    relay_queue=None,
) -> None:
    """Claim the web run's exact immutable attempt through the V2 executor."""

    if relay_queue is not None:
        get_event_bus().attach_relay_queue(relay_queue)
    try:
        from genetic_algorithm.orchestration.attempt_state_v2 import (
            AttemptLifecycleStatus,
            AttemptStateStoreV2,
        )
        from genetic_algorithm.orchestration.runner_v2 import execute_queued_attempt

        terminal = execute_queued_attempt(
            AttemptStateStoreV2(state_path),
            attempt_id,
            worker_id=f"web-runner:{run_id}",
        )
        if terminal.status != AttemptLifecycleStatus.SUCCEEDED:
            raise RuntimeError(
                f"attempt {attempt_id} ended as {terminal.status.value}: "
                f"{terminal.error_code or 'NO_ERROR_CODE'}"
            )
        get_event_bus().publish(
            Event(
                type=EventType.EVOLUTION_COMPLETE,
                run_id=run_id,
                data={
                    "attempt_id": attempt_id,
                    "artifact_root": terminal.artifact_root,
                },
            )
        )
    except Exception as exc:
        logger.exception("Evolution run %s failed", run_id)
        get_event_bus().publish(
            Event(
                type=EventType.RUN_ERROR,
                run_id=run_id,
                data={"attempt_id": attempt_id, "error": str(exc)},
            )
        )
        raise
