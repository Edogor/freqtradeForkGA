"""T3.9 — Crash-safe evaluation backends.

This module provides two **process-free** alternatives to the
``ProcessPoolExecutor``-based ``ParallelEvaluator`` in ``parallel.py``:

* :class:`ThreadPoolEvaluator` — runs strategy evaluations in an
  in-process thread pool.  Pandas / NumPy releases the GIL for most
  array work so backtests get 2-3× speedup on a multi-core machine.
  No fork, no pickling, no ``BrokenProcessPool`` errors.
* :class:`SequentialEvaluator` — a single-thread, in-process loop.
  Slowest but bulletproof.  Used as a fall-back when the user wants
  zero parallel-evaluation surface.

Both implement the same minimal interface as ``ParallelEvaluator``::

    evaluator.num_workers     -> int
    evaluator.evaluate_batch(individuals, progress_callback=None)
        -> ParallelEvaluationResult
    evaluator.shutdown()

so they can be slotted in via ``parallel_evaluation.backend`` without
changing the call sites in ``core/evolution.py`` or
``engine/generation.py``.

Worker-crash motivation (user-reported):
  "wir sollten keine workers nutzen da das sehr oft zu problemen
   geführt hat".  Threads avoid the entire OS-process failure mode
   that produces orphaned PIDs, ``BrokenProcessPool``, and zombie
   reapers — at the price of weaker timeout semantics (we can't kill
   a stuck thread, only mark the future as failed and let it finish).
"""
from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import (
    ThreadPoolExecutor,
    Future,
    FIRST_COMPLETED,
    TimeoutError as FuturesTimeoutError,
    as_completed,
    wait,
)
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from genetic_algorithm.evaluation.parallel import ParallelEvaluationResult

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────


def _resolve_num_workers(
    config: Dict[str, Any], num_workers: Optional[int]
) -> int:
    parallel_config = config.get("parallel_evaluation", {}) or {}
    if num_workers is not None:
        return max(1, int(num_workers))
    if parallel_config.get("num_workers"):
        return max(1, int(parallel_config["num_workers"]))
    cpu = os.cpu_count() or 2
    return max(1, cpu - 1)


def _safe_evaluate(
    evaluator,
    individual,
    nsga2_mode: bool,
    objectives_config: list,
    nsga2_min_trades: int,
) -> Dict[str, Any]:
    """Evaluate one individual, never raising; return a result dict
    compatible with the ParallelEvaluator output schema."""
    try:
        from genetic_algorithm.core.nsga2 import extract_objectives_from_metrics
    except Exception:
        extract_objectives_from_metrics = None  # type: ignore

    try:
        fitness, metrics = evaluator.evaluate(individual.strategy_gene)
        result = {
            "fitness": fitness,
            "metrics": metrics,
            "success": True,
        }
        if nsga2_mode and objectives_config and extract_objectives_from_metrics:
            try:
                result["objectives"] = extract_objectives_from_metrics(
                    metrics, objectives_config, min_trades=nsga2_min_trades
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug(f"[T3.9] objective extraction failed: {exc}")
        return result
    except Exception as exc:
        return {
            "fitness": 0.0,
            "metrics": {"error": str(exc)},
            "success": False,
            "error": str(exc),
        }


# ── Sequential evaluator ──────────────────────────────────────────────


class SequentialEvaluator:
    """Zero-parallelism, zero-risk fallback evaluator.

    Runs all evaluations in the calling thread.  Implements just enough
    of the ``ParallelEvaluator`` surface that ``core/evolution.py`` and
    ``engine/generation.py`` can swap it in transparently.
    """

    def __init__(self, config: Dict[str, Any], num_workers: Optional[int] = None):
        from genetic_algorithm.evaluation.fitness import FitnessEvaluator

        self.config = config
        # Ignore any worker hint — by definition we use 1.
        self.num_workers = 1
        self.backtest_timeout = (config.get("parallel_evaluation", {}) or {}).get(
            "backtest_timeout", 120
        )
        self.nsga2_mode = (
            config.get("genetic_algorithm", {}).get("mode") == "nsga2"
        )
        self.objectives_config = config.get("nsga2", {}).get("objectives", []) or []
        self.nsga2_min_trades = config.get("nsga2", {}).get("min_trades", 0)
        self._evaluator = FitnessEvaluator(config)
        logger.info("[T3.9] SequentialEvaluator initialised (1 worker, no pool)")

    # API ----------------------------------------------------------------

    def evaluate_batch(
        self,
        individuals: List[Any],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> ParallelEvaluationResult:
        if not individuals:
            return ParallelEvaluationResult(
                successful=0, failed=0, total_time=0.0, speedup_estimate=1.0
            )
        start = time.time()
        successful = failed = 0
        for i, ind in enumerate(individuals):
            r = _safe_evaluate(
                self._evaluator,
                ind,
                self.nsga2_mode,
                self.objectives_config,
                self.nsga2_min_trades,
            )
            if r["success"]:
                ind.set_fitness(r["fitness"], r["metrics"])
                if self.nsga2_mode and "objectives" in r:
                    ind.set_objectives(r["objectives"], r["metrics"])
                successful += 1
            else:
                ind.set_fitness(0.0, r.get("metrics", {}))
                failed += 1
            if progress_callback:
                try:
                    progress_callback(i + 1, len(individuals))
                except Exception:
                    pass
        total = time.time() - start
        return ParallelEvaluationResult(
            successful=successful, failed=failed, total_time=total, speedup_estimate=1.0
        )

    def shutdown(self) -> None:  # pragma: no cover - trivial
        return

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.shutdown()


# ── Thread-pool evaluator ─────────────────────────────────────────────


class ThreadPoolEvaluator:
    """In-process thread-pool evaluator (crash-safe alternative).

    Uses ``concurrent.futures.ThreadPoolExecutor``.  Crash-safe because:
      * No worker process to die: a stuck or slow strategy can't kill
        the pool.
      * Timeout: ``future.result(timeout=...)`` returns ``TimeoutError``
        and we mark the strategy as failed; the thread itself keeps
        running, but the GA continues.  On the next batch the pool is
        still healthy.
      * Shared FitnessEvaluator: cache hits across the batch.

    Speedup is real but smaller than the process-pool: 2-3× on 4-6
    cores, vs 4-5× for processes.  The trade-off — reliability over
    raw throughput — is the explicit goal of T3.9.
    """

    def __init__(self, config: Dict[str, Any], num_workers: Optional[int] = None):
        from genetic_algorithm.evaluation.fitness import FitnessEvaluator

        self.config = config
        self.num_workers = _resolve_num_workers(config, num_workers)
        parallel_config = config.get("parallel_evaluation", {}) or {}
        self.backtest_timeout = parallel_config.get("backtest_timeout", 120)
        self.nsga2_mode = (
            config.get("genetic_algorithm", {}).get("mode") == "nsga2"
        )
        self.objectives_config = config.get("nsga2", {}).get("objectives", []) or []
        self.nsga2_min_trades = config.get("nsga2", {}).get("min_trades", 0)

        # A single shared evaluator — backtest cache works across the batch.
        self._evaluator = FitnessEvaluator(config)
        # NOTE: We deliberately do NOT serialise calls to the shared
        # FitnessEvaluator.  Pandas/NumPy backtests are read-mostly on
        # the cached candle data; the only mutable state is the
        # evaluation cache which tolerates benign races (worst case:
        # the same strategy is evaluated twice).  Serialising here
        # would defeat the entire point of T3.9 parallelism.
        self._executor: Optional[ThreadPoolExecutor] = None
        self._batch_count = 0
        logger.info(
            f"[T3.9] ThreadPoolEvaluator initialised "
            f"({self.num_workers} threads, timeout={self.backtest_timeout}s)"
        )

    # Internal --------------------------------------------------------------

    def _get_executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self.num_workers,
                thread_name_prefix="ga-eval",
            )
        return self._executor

    def _worker_call(self, individual) -> Dict[str, Any]:
        return _safe_evaluate(
            self._evaluator,
            individual,
            self.nsga2_mode,
            self.objectives_config,
            self.nsga2_min_trades,
        )

    # API ----------------------------------------------------------------

    def evaluate_batch(
        self,
        individuals: List[Any],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> ParallelEvaluationResult:
        if not individuals:
            return ParallelEvaluationResult(
                successful=0, failed=0, total_time=0.0, speedup_estimate=1.0
            )
        start = time.time()
        self._batch_count += 1
        executor = self._get_executor()
        submission_time: Dict[Future, float] = {}
        fut_to_idx: Dict[Future, int] = {}
        for i, ind in enumerate(individuals):
            f = executor.submit(self._worker_call, ind)
            submission_time[f] = time.time()
            fut_to_idx[f] = i

        successful = failed = timed_out = 0
        per_task_timeout = self.backtest_timeout if self.backtest_timeout else None
        pending = set(fut_to_idx)
        completed_count = 0

        def _record(fut: Future, timed_out_local: bool = False):
            nonlocal successful, failed, timed_out, completed_count
            idx = fut_to_idx[fut]
            ind = individuals[idx]
            if timed_out_local:
                timed_out += 1
                failed += 1
                ind.set_fitness(
                    0.0, {"error": f"timeout after {per_task_timeout}s"}
                )
            else:
                try:
                    r = fut.result(timeout=0)
                except Exception as exc:  # pragma: no cover - defensive
                    failed += 1
                    ind.set_fitness(0.0, {"error": str(exc)})
                else:
                    if r["success"]:
                        ind.set_fitness(r["fitness"], r["metrics"])
                        if self.nsga2_mode and "objectives" in r:
                            ind.set_objectives(r["objectives"], r["metrics"])
                        successful += 1
                    else:
                        ind.set_fitness(0.0, r.get("metrics", {}))
                        failed += 1
            completed_count += 1
            if progress_callback:
                try:
                    progress_callback(completed_count, len(individuals))
                except Exception:
                    pass

        while pending:
            if per_task_timeout is None:
                # No per-task timeout — just wait for everything.
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for fut in done:
                    _record(fut)
                    pending.discard(fut)
                continue
            now = time.time()
            # Earliest deadline among pending futures.
            next_deadline = min(submission_time[f] + per_task_timeout for f in pending)
            wait_for = max(0.0, next_deadline - now)
            done, _ = wait(pending, timeout=wait_for, return_when=FIRST_COMPLETED)
            if done:
                for fut in done:
                    _record(fut)
                    pending.discard(fut)
            else:
                # Nothing completed before next deadline → expire stale ones.
                now = time.time()
                for fut in list(pending):
                    if submission_time[fut] + per_task_timeout <= now:
                        _record(fut, timed_out_local=True)
                        pending.discard(fut)

        total = time.time() - start
        if timed_out:
            logger.warning(
                f"[T3.9] ThreadPool batch #{self._batch_count}: "
                f"{timed_out}/{len(individuals)} strategies timed out"
            )
        # Speedup estimate: rough — assume linear scaling for the part
        # that ran, ignoring contended GIL portions.
        speedup_estimate = float(min(self.num_workers, max(1, len(individuals))))
        return ParallelEvaluationResult(
            successful=successful,
            failed=failed,
            total_time=total,
            speedup_estimate=speedup_estimate,
        )

    def shutdown(self) -> None:
        if self._executor is not None:
            try:
                self._executor.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                # Python < 3.9 lacks cancel_futures kwarg
                self._executor.shutdown(wait=False)
            self._executor = None
            logger.info("[T3.9] ThreadPoolEvaluator shut down")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.shutdown()

    def __del__(self):  # pragma: no cover - destructor
        try:
            self.shutdown()
        except Exception:
            pass


# ── Backend factory ───────────────────────────────────────────────────


def select_evaluator_backend(
    config: Dict[str, Any], num_workers: Optional[int] = None
):
    """Pick the right evaluator class based on ``parallel_evaluation.backend``.

    Backend values:
      * ``'process'`` (default)  — the existing ProcessPoolExecutor-based
        ``ParallelEvaluator``.  Best raw throughput but vulnerable to
        worker crashes (BrokenProcessPool).
      * ``'thread'``             — ThreadPoolEvaluator (this module).
        Crash-safe; lower speedup but resilient.
      * ``'sequential'``         — SequentialEvaluator.  No parallelism
        at all.  Safest option for debugging or low-RAM machines.

    Returns ``None`` when ``parallel_evaluation.enabled`` is false so
    callers can fall back to single-threaded inline evaluation.
    """
    pe = config.get("parallel_evaluation", {}) or {}
    if not pe.get("enabled", False):
        return None
    backend = (pe.get("backend") or "process").lower().strip()
    if backend == "thread":
        return ThreadPoolEvaluator(config, num_workers=num_workers)
    if backend == "sequential":
        return SequentialEvaluator(config, num_workers=num_workers)
    # Default: existing process-pool implementation
    from genetic_algorithm.evaluation.parallel import ParallelEvaluator

    return ParallelEvaluator(config, num_workers=num_workers)
