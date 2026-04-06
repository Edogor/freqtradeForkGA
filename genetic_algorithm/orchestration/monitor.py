"""
Monitor — thin wrapper for terminal/web monitoring of running experiments.

This delegates to existing monitoring code (terminal_monitor, web_dashboard)
and the experiment registry.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

from genetic_algorithm.orchestration.registry import ExperimentRegistry

logger = logging.getLogger(__name__)


class ExperimentMonitor:
    """Watch one or all running experiments — refresh loop."""

    def __init__(self, experiment_id: Optional[str] = None) -> None:
        self.experiment_id = experiment_id
        self.registry = ExperimentRegistry()

    def run(self, interval: int = 5) -> None:
        """Blocking refresh loop printing status to stdout."""
        try:
            while True:
                self._print_status()
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\nMonitor stopped.")

    def _print_status(self) -> None:
        if self.experiment_id:
            exp = self.registry.get(self.experiment_id)
            if not exp:
                print(f"Experiment {self.experiment_id} not found in registry.")
                return
            _print_single(exp)
        else:
            running = self.registry.list(status="running")
            if not running:
                print("No running experiments.")
                return
            print(f"\n{'='*60}")
            print(f"  {len(running)} running experiment(s)")
            print(f"{'='*60}")
            for exp in running:
                _print_single(exp)
            print()


def _print_single(exp: dict) -> None:
    eid = exp["experiment_id"]
    status = exp["status"]
    started = exp.get("started_at", "?")
    tags = ", ".join(exp.get("tags", []))
    pid = exp.get("pid", "?")
    print(f"  [{status:>10}]  {eid}  pid={pid}  started={started}  tags={tags}")
