"""
ExperimentMonitor — terminal dashboard for running GA experiments.

Replaces ``ga_monitor_v2.sh`` with a Python implementation that displays the
authoritative SQLite lifecycle plus non-authoritative log progress. Economic
metrics are read only from canonical structured catalog evidence.

Usage::

    monitor = ExperimentMonitor()
    monitor.run()               # continuous refresh
    monitor.snapshot()          # single status print
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import List, Optional

from genetic_algorithm.orchestration.attempt_state_v2 import AttemptStateStoreV2
from genetic_algorithm.orchestration.experiment_catalog_v2 import ExperimentCatalogV2
from genetic_algorithm.orchestration.runner_v2 import default_state_path


logger = logging.getLogger(__name__)

# Number of tail lines to scan for metrics
_TAIL_LINES = 300

# ──────────────────────────────────────────────────────────────────
# Metrics extraction
# ──────────────────────────────────────────────────────────────────

# Compiled regex patterns for log parsing
_RE_GENERATION_SUMMARY = re.compile(r"\[SUMMARY\] Gen (\d+)/(\d+)")
_RE_GENERATION_PLAIN = re.compile(r"GENERATION (\d+)/(\d+)")
_RE_EVAL_PROGRESS = re.compile(r"\[EVAL\] Progress: (\d+)/(\d+)")
_RE_ERRORS = re.compile(r"\[ERROR\]|\bTraceback\b")


@dataclass
class ExperimentMetrics:
    """Metrics extracted from a single experiment's log."""

    experiment_id: str = ""
    status: str = "UNKNOWN"
    generation: Optional[int] = None
    total_generations: Optional[int] = None
    best_fitness: Optional[float] = None
    avg_fitness: Optional[float] = None
    diversity: Optional[float] = None
    best_profit: Optional[float] = None
    eval_progress: Optional[str] = None
    error_count: int = 0
    elapsed: Optional[str] = None
    eta: Optional[str] = None
    pid: Optional[int] = None
    log_path: Optional[str] = None
    metric_basis: Optional[str] = None


def extract_metrics(log_path: Path, tail_lines: int = _TAIL_LINES) -> ExperimentMetrics:
    """Extract non-authoritative progress diagnostics from a log tail.

    Fitness, return, risk, diversity and lifecycle are deliberately excluded.
    """
    m = ExperimentMetrics()

    if not log_path.exists():
        return m

    try:
        with open(log_path) as f:
            lines = f.readlines()
    except OSError:
        return m

    tail = lines[-tail_lines:] if len(lines) > tail_lines else lines
    buf = "".join(tail)

    # Generation progress
    match = None
    for match in _RE_GENERATION_SUMMARY.finditer(buf):
        pass
    if match:
        m.generation = int(match.group(1))
        m.total_generations = int(match.group(2))
    else:
        match = None
        for match in _RE_GENERATION_PLAIN.finditer(buf):
            pass
        if match:
            m.generation = int(match.group(1))
            m.total_generations = int(match.group(2))

    # Eval progress
    match = None
    for match in _RE_EVAL_PROGRESS.finditer(buf):
        pass
    if match:
        m.eval_progress = f"{match.group(1)}/{match.group(2)}"

    # Error count
    m.error_count = len(_RE_ERRORS.findall(buf))

    return m


# ──────────────────────────────────────────────────────────────────
# Monitor
# ──────────────────────────────────────────────────────────────────


class ExperimentMonitor:
    """Terminal dashboard for GA experiments.

    Uses the SQLite experiment catalog for state and parses log files
    for real-time metrics like generation, fitness, profit, ETA.
    """

    def __init__(
        self,
        experiment_id: Optional[str] = None,
        show_completed: bool = False,
        *,
        state_path: str | Path | None = None,
        include_legacy: bool = False,
        statuses: List[str] | None = None,
        tags: List[str] | None = None,
    ) -> None:
        self.experiment_id = experiment_id
        self.show_completed = show_completed
        self.include_legacy = include_legacy
        self.statuses = list(statuses) if statuses is not None else None
        self.tags = list(tags or [])
        self.catalog = ExperimentCatalogV2(AttemptStateStoreV2(state_path or default_state_path()))

    def run(self, interval: int = 10) -> None:
        """Blocking refresh loop printing status to stdout."""
        try:
            while True:
                os.system("clear")
                self._print_dashboard()
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\nMonitor stopped.")

    def snapshot(self) -> None:
        """Print a single status snapshot."""
        self._print_dashboard()

    def get_metrics(self) -> List[ExperimentMetrics]:
        """Extract metrics for all relevant experiments."""
        if self.experiment_id:
            record = self.catalog.get_record(
                self.experiment_id,
                include_legacy=self.include_legacy,
            )
            experiments = [record.compatibility_dict()] if record else []
        else:
            statuses = self.statuses or ["running"]
            if self.statuses is None and self.show_completed:
                statuses.extend(["completed", "failed"])
            experiments = [
                record.compatibility_dict()
                for record in self.catalog.list_records(
                    statuses=statuses,
                    tags=self.tags,
                    include_legacy=self.include_legacy,
                )
            ]

        results = []
        for exp in experiments:
            log_path = exp.get("log_path")
            if not log_path:
                # Minimal entry from registry only
                m = ExperimentMetrics(
                    experiment_id=exp["experiment_id"],
                    status=exp["status"].upper(),
                    pid=exp.get("pid"),
                )
                results.append(m)
                continue

            lp = Path(log_path)
            m = extract_metrics(lp)
            m.experiment_id = exp["experiment_id"]
            m.pid = exp.get("pid")
            m.log_path = log_path

            # SQLite lifecycle state is authoritative. Log parsing only adds
            # non-lifecycle progress metrics.
            reg_status = exp.get("status", "")
            m.status = reg_status.upper() if reg_status else "UNKNOWN"

            # Elapsed time
            started = exp.get("started_at")
            if started:
                try:
                    start_dt = datetime.fromisoformat(started)
                    if start_dt.utcoffset() is None:
                        start_dt = start_dt.replace(tzinfo=UTC)
                    elapsed = datetime.now(UTC) - start_dt.astimezone(UTC)
                    m.elapsed = _format_duration(elapsed.total_seconds())

                    # ETA calculation
                    if (
                        m.status == "RUNNING"
                        and m.generation
                        and m.total_generations
                        and m.generation > 0
                    ):
                        secs = elapsed.total_seconds()
                        per_gen = secs / m.generation
                        remaining = (m.total_generations - m.generation) * per_gen
                        m.eta = _format_duration(remaining)
                except (ValueError, TypeError):
                    pass

            # Only canonical catalog evidence may populate economic fields.
            # Legacy imports remain visible as lifecycle history but cannot
            # masquerade as verified candidate metrics.
            if exp.get("source") == "CANONICAL_V2":
                m.best_fitness = exp.get("best_fitness")
                m.best_profit = exp.get("best_profit")
                m.metric_basis = exp.get("best_net_return_basis")
            if m.generation is None and exp.get("generation"):
                m.generation = exp["generation"]
            if m.total_generations is None and exp.get("generations_total"):
                m.total_generations = exp["generations_total"]

            results.append(m)

        return results

    def _print_dashboard(self) -> None:
        """Render the dashboard to stdout."""
        metrics = self.get_metrics()

        now = datetime.now().strftime("%H:%M:%S")
        running = sum(1 for m in metrics if m.status == "RUNNING")
        queued = self.catalog.summary(include_legacy=False).get("queued", 0)

        print(f"{'=' * 80}")
        print(f"  GA Monitor  |  {now}  |  {running} running  |  {queued} queued")
        print(f"{'=' * 80}")

        if not metrics:
            print("  No experiments found.")
            print()
            return

        # Header
        print(
            f"  {'Experiment':<25} {'Status':>8} {'Gen':>10} "
            f"{'Best':>8} {'Avg':>8} {'Profit':>8} "
            f"{'Elapsed':>8} {'ETA':>8} {'Err':>4}"
        )
        print(f"  {'-' * 78}")

        for m in metrics:
            gen_str = f"{m.generation}/{m.total_generations}" if m.generation else "-"
            best = f"{m.best_fitness:.4f}" if m.best_fitness is not None else "-"
            avg = f"{m.avg_fitness:.4f}" if m.avg_fitness is not None else "-"
            profit = f"{m.best_profit:+.1f}%" if m.best_profit is not None else "-"
            elapsed = m.elapsed or "-"
            eta = m.eta or "-"
            errors = str(m.error_count) if m.error_count else "-"

            # Truncate long experiment names
            eid = m.experiment_id[:25]

            print(
                f"  {eid:<25} {m.status:>8} {gen_str:>10} "
                f"{best:>8} {avg:>8} {profit:>8} "
                f"{elapsed:>8} {eta:>8} {errors:>4}"
            )

        print()


def _format_duration(seconds: float) -> str:
    """Format seconds as HH:MM:SS or Xd HH:MM."""
    seconds = max(0, seconds)
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    if hours >= 24:
        days = hours // 24
        hours = hours % 24
        return f"{days}d {hours:02d}:{minutes:02d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
