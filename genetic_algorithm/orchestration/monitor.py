"""
ExperimentMonitor — terminal dashboard for running GA experiments.

Replaces ``ga_monitor_v2.sh`` with a Python implementation that
parses log files for metrics and displays a dashboard.

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
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from genetic_algorithm.orchestration.registry import ExperimentRegistry

logger = logging.getLogger(__name__)

# Number of tail lines to scan for metrics
_TAIL_LINES = 300

# ──────────────────────────────────────────────────────────────────
# Metrics extraction
# ──────────────────────────────────────────────────────────────────

# Compiled regex patterns for log parsing
_RE_GENERATION_SUMMARY = re.compile(r"\[SUMMARY\] Gen (\d+)/(\d+)")
_RE_GENERATION_PLAIN = re.compile(r"GENERATION (\d+)/(\d+)")
_RE_BEST_STATS = re.compile(r"\[STATS\] Best: ([0-9.]+)")
_RE_BEST_ISLAND = re.compile(r"island_[^=]+=([0-9.]+)")
_RE_BEST_NEW = re.compile(r"\[NEW BEST\].*fitness.?([0-9.]+)")
_RE_AVG_STATS = re.compile(r"\[STATS\].*Avg: ([0-9.]+)")
_RE_DIVERSITY = re.compile(r"diversity[=: ]+([0-9.]+)", re.IGNORECASE)
_RE_PROFIT = re.compile(r"profit=([0-9.+-]+)%?")
_RE_EVAL_PROGRESS = re.compile(r"\[EVAL\] Progress: (\d+)/(\d+)")
_RE_ERRORS = re.compile(r"\[ERROR\]|\bTraceback\b")
_RE_COMPLETE = re.compile(r"GA RUN COMPLETE|Evolution complete")


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


def extract_metrics(log_path: Path, tail_lines: int = _TAIL_LINES) -> ExperimentMetrics:
    """Extract GA metrics from the tail of a log file.

    Returns an ExperimentMetrics with whatever data could be parsed.
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

    # Best fitness - try multiple sources
    match = None
    for match in _RE_BEST_STATS.finditer(buf):
        pass
    if match:
        m.best_fitness = float(match.group(1))
    else:
        # Try island summary (max across islands)
        island_vals = _RE_BEST_ISLAND.findall(buf)
        if island_vals:
            m.best_fitness = max(float(v) for v in island_vals)
        else:
            match = None
            for match in _RE_BEST_NEW.finditer(buf):
                pass
            if match:
                m.best_fitness = float(match.group(1))

    # Average fitness
    match = None
    for match in _RE_AVG_STATS.finditer(buf):
        pass
    if match:
        m.avg_fitness = float(match.group(1))

    # Diversity
    match = None
    for match in _RE_DIVERSITY.finditer(buf):
        pass
    if match:
        m.diversity = float(match.group(1))

    # Best profit
    profits = _RE_PROFIT.findall(buf)
    if profits:
        m.best_profit = max(float(p) for p in profits)

    # Eval progress
    match = None
    for match in _RE_EVAL_PROGRESS.finditer(buf):
        pass
    if match:
        m.eval_progress = f"{match.group(1)}/{match.group(2)}"

    # Error count
    m.error_count = len(_RE_ERRORS.findall(buf))

    # Completion check
    if _RE_COMPLETE.search(buf):
        m.status = "DONE"

    return m


def detect_status(
    log_path: Path, pid: Optional[int] = None, stale_seconds: int = 600
) -> str:
    """Determine experiment status: RUNNING, DONE, CRASHED, STALE."""
    if not log_path.exists():
        return "UNKNOWN"

    # Check for completion marker
    try:
        with open(log_path) as f:
            lines = f.readlines()
        tail = "".join(lines[-30:])
        if _RE_COMPLETE.search(tail):
            return "DONE"
    except OSError:
        return "UNKNOWN"

    # Check if process is alive
    if pid:
        try:
            os.kill(pid, 0)
            return "RUNNING"
        except (ProcessLookupError, PermissionError):
            pass

    # Check log freshness
    try:
        mtime = log_path.stat().st_mtime
        age = time.time() - mtime
        if age > stale_seconds:
            return "STALE"
    except OSError:
        pass

    return "CRASHED" if pid else "UNKNOWN"


# ──────────────────────────────────────────────────────────────────
# Monitor
# ──────────────────────────────────────────────────────────────────


class ExperimentMonitor:
    """Terminal dashboard for GA experiments.

    Uses the experiment registry for state and parses log files
    for real-time metrics like generation, fitness, profit, ETA.
    """

    def __init__(
        self,
        experiment_id: Optional[str] = None,
        show_completed: bool = False,
    ) -> None:
        self.experiment_id = experiment_id
        self.show_completed = show_completed
        self.registry = ExperimentRegistry()

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
            experiments = [self.registry.get(self.experiment_id)]
            experiments = [e for e in experiments if e]
        else:
            statuses = ["running"]
            if self.show_completed:
                statuses.extend(["completed", "failed"])
            experiments = self.registry.list(status=statuses)

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

            # Override status from registry if available, or detect from log
            reg_status = exp.get("status", "")
            if reg_status in ("completed", "failed", "cancelled"):
                m.status = reg_status.upper()
            else:
                m.status = detect_status(lp, pid=exp.get("pid"))

            # Elapsed time
            started = exp.get("started_at")
            if started:
                try:
                    start_dt = datetime.fromisoformat(started)
                    elapsed = datetime.utcnow() - start_dt.replace(tzinfo=None)
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

            # Use registry fitness if log didn't have it
            if m.best_fitness is None and exp.get("best_fitness"):
                m.best_fitness = exp["best_fitness"]
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
        queued = self.registry.count(status="queued")

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
