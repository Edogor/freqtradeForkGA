#!/usr/bin/env python3
"""
SIS Comparison Monitor
Reads ctrl and exp log files, shows side-by-side fitness comparison.
Usage: python -m genetic_algorithm.intelligence.sis_monitor
       python -m genetic_algorithm.intelligence.sis_monitor --once
       python -m genetic_algorithm.intelligence.sis_monitor --ctrl-log path/to/ctrl.log --exp-log path/to/exp.log
"""

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_CTRL_LOG = "genetic_algorithm/logs/sis_comparison/ctrl.log"
DEFAULT_EXP_LOG = "genetic_algorithm/logs/sis_comparison/exp.log"
DEFAULT_SIS_LOG = "genetic_algorithm/logs/sis_comparison/sis_events.jsonl"
DEFAULT_INTERVAL = 30

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class GenRecord:
    gen: int
    best: Optional[float] = None
    mean: Optional[float] = None


@dataclass
class RunState:
    name: str
    records: Dict[int, GenRecord] = field(default_factory=dict)

    def upsert(self, gen: int, best: Optional[float] = None, mean: Optional[float] = None) -> None:
        if gen not in self.records:
            self.records[gen] = GenRecord(gen=gen)
        rec = self.records[gen]
        if best is not None:
            rec.best = best
        if mean is not None:
            rec.mean = mean

    def latest_gen(self) -> int:
        return max(self.records.keys(), default=0)

    def sorted_records(self) -> List[GenRecord]:
        return [self.records[g] for g in sorted(self.records.keys())]


# ---------------------------------------------------------------------------
# Log parsers
# ---------------------------------------------------------------------------

# Regex patterns tried in order (case-insensitive)
_PATTERNS: List[Tuple[str, str]] = [
    # "Gen 3 | best=0.712 | mean=0.634"  or  "Gen 3/15 | best=0.712 mean=0.634"
    (
        "full",
        r"[Gg]en(?:eration)?\s+(\d+)[^\d|]*(?:best[_\s=:]+(\d+\.\d+))[^\d]*(?:mean[_\s=:]+(\d+\.\d+))",
    ),
    # "Generation 3  best_fitness=0.712  mean_fitness=0.634"
    (
        "underscore",
        r"[Gg]en(?:eration)?\s+(\d+)[^\d]*best_fitness[=:\s]+(\d+\.\d+)[^\d]*mean_fitness[=:\s]+(\d+\.\d+)",
    ),
    # single best only — fallback
    (
        "best_only",
        r"[Gg]en(?:eration)?\s+(\d+)[^\d]*(?:best[_\s=:]+|Best\s+fitness[=:\s]+)(\d+\.\d+)",
    ),
]

_RE_BEST_STANDALONE = re.compile(r"best[_\s=:]+(\d+\.\d+)", re.IGNORECASE)
_RE_MEAN_STANDALONE = re.compile(r"mean[_\s=:]+(\d+\.\d+)", re.IGNORECASE)
_RE_GEN_NUMBER = re.compile(r"[Gg]en(?:eration)?\s+(\d+)")


def _parse_log_file(path: Path) -> RunState:
    """Parse a GA log file and return a RunState with per-generation records."""
    name = path.stem
    state = RunState(name=name)

    if not path.exists():
        return state

    compiled = [(name, re.compile(pat, re.IGNORECASE)) for name, pat in _PATTERNS]

    try:
        with path.open("r", errors="replace") as fh:
            for line in fh:
                _try_parse_line(line, state, compiled)
    except OSError:
        pass

    return state


def _try_parse_line(
    line: str,
    state: RunState,
    compiled: List[Tuple[str, re.Pattern]],
) -> None:
    """Attempt to extract generation/fitness info from a single log line."""

    for pat_name, pat in compiled:
        m = pat.search(line)
        if not m:
            continue
        gen = int(m.group(1))
        if pat_name in ("full", "underscore"):
            best = float(m.group(2)) if m.group(2) else None
            mean = float(m.group(3)) if m.group(3) else None
            state.upsert(gen, best=best, mean=mean)
        else:  # best_only
            best = float(m.group(2)) if m.group(2) else None
            state.upsert(gen, best=best)
        return

    # Last-resort: line mentions a gen number + standalone best/mean
    gm = _RE_GEN_NUMBER.search(line)
    if not gm:
        return
    gen = int(gm.group(1))
    bm = _RE_BEST_STANDALONE.search(line)
    mm = _RE_MEAN_STANDALONE.search(line)
    if bm or mm:
        state.upsert(
            gen,
            best=float(bm.group(1)) if bm else None,
            mean=float(mm.group(1)) if mm else None,
        )


# ---------------------------------------------------------------------------
# SIS events parser
# ---------------------------------------------------------------------------


def _parse_sis_events(path: Path) -> Dict[int, List[dict]]:
    """Parse JSONL SIS events file.  Returns dict[gen -> list[event]]."""
    events_by_gen: Dict[int, List[dict]] = defaultdict(list)
    if not path.exists():
        return events_by_gen
    try:
        with path.open("r", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                gen = ev.get("gen") or ev.get("generation") or ev.get("g")
                if gen is not None:
                    events_by_gen[int(gen)].append(ev)
    except OSError:
        pass
    return events_by_gen


def _archetype_coverage(events_by_gen: Dict[int, List[dict]]) -> Dict[str, int]:
    """Count how many immigrants/seeds came from each archetype."""
    counts: Dict[str, int] = defaultdict(int)
    for events in events_by_gen.values():
        for ev in events:
            arch = ev.get("archetype") or ev.get("archetype_id")
            if arch is not None:
                counts[str(arch)] += 1
    return dict(counts)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_COL_W = 10  # width of numeric columns


def _fmt(val: Optional[float]) -> str:
    if val is None:
        return " " * _COL_W
    return f"{val:>{_COL_W}.4f}"


def _sis_summary(gens_events: Dict[int, List[dict]], gen: int) -> str:
    """One-line summary of SIS events for a generation."""
    evs = gens_events.get(gen, [])
    if not evs:
        return ""
    types: Dict[str, int] = defaultdict(int)
    for ev in evs:
        t = ev.get("type") or ev.get("event") or "event"
        types[str(t)] += 1
    parts = [f"{v}×{k}" for k, v in sorted(types.items())]
    return ", ".join(parts)


def _render_table(
    ctrl: RunState,
    exp: RunState,
    sis_events: Dict[int, List[dict]],
) -> str:
    """Build the comparison table string."""
    lines: List[str] = []

    # Header
    ctrl_label = f"CTRL ({ctrl.name})"
    exp_label = f"EXP  ({exp.name})"
    lines.append("")
    lines.append("=" * 80)
    lines.append("  SIS COMPARISON MONITOR")
    lines.append("=" * 80)
    lines.append(
        f"  {'Gen':>5}  {'Ctrl Best':>10}  {'Ctrl Mean':>10}  "
        f"{'Exp Best':>10}  {'Exp Mean':>10}  {'Delta Best':>10}  SIS Events"
    )
    lines.append("-" * 80)

    all_gens = sorted(
        set(ctrl.records.keys()) | set(exp.records.keys())
    )

    if not all_gens:
        lines.append("  (no data yet — waiting for log entries)")
    else:
        for g in all_gens:
            cr = ctrl.records.get(g)
            er = exp.records.get(g)
            c_best = cr.best if cr else None
            c_mean = cr.mean if cr else None
            e_best = er.best if er else None
            e_mean = er.mean if er else None

            delta = None
            if c_best is not None and e_best is not None:
                delta = e_best - c_best
            delta_str = (
                f"{delta:>+{_COL_W}.4f}" if delta is not None else " " * _COL_W
            )

            sis_str = _sis_summary(sis_events, g)
            lines.append(
                f"  {g:>5}  {_fmt(c_best)}  {_fmt(c_mean)}  "
                f"{_fmt(e_best)}  {_fmt(e_mean)}  {delta_str}  {sis_str}"
            )

    lines.append("-" * 80)

    # Summary row
    c_all_best = [r.best for r in ctrl.records.values() if r.best is not None]
    e_all_best = [r.best for r in exp.records.values() if r.best is not None]
    if c_all_best or e_all_best:
        c_peak = max(c_all_best) if c_all_best else None
        e_peak = max(e_all_best) if e_all_best else None
        delta_peak = None
        if c_peak is not None and e_peak is not None:
            delta_peak = e_peak - c_peak
        delta_str = (
            f"{delta_peak:>+{_COL_W}.4f}" if delta_peak is not None else " " * _COL_W
        )
        lines.append(
            f"  {'PEAK':>5}  {_fmt(c_peak)}  {'':>10}  "
            f"{_fmt(e_peak)}  {'':>10}  {delta_str}"
        )

    lines.append("")

    # Archetype coverage
    arch_cov = _archetype_coverage(sis_events)
    if arch_cov:
        lines.append("  Archetype coverage (SIS immigrants/seeds):")
        for arch, cnt in sorted(arch_cov.items(), key=lambda x: -x[1]):
            lines.append(f"    Archetype {arch:>4}: {cnt:>4} events")
        lines.append("")

    # Status
    ctrl_latest = ctrl.latest_gen()
    exp_latest = exp.latest_gen()
    total_sis = sum(len(v) for v in sis_events.values())
    lines.append(
        f"  Status: CTRL gen={ctrl_latest}  EXP gen={exp_latest}"
        f"  SIS total events={total_sis}"
    )
    lines.append("=" * 80)
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Monitor SIS vs CTRL GA comparison runs side by side.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--ctrl-log",
        default=DEFAULT_CTRL_LOG,
        help=f"Path to CTRL log file (default: {DEFAULT_CTRL_LOG})",
    )
    p.add_argument(
        "--exp-log",
        default=DEFAULT_EXP_LOG,
        help=f"Path to EXP log file (default: {DEFAULT_EXP_LOG})",
    )
    p.add_argument(
        "--sis-log",
        default=DEFAULT_SIS_LOG,
        help=f"Path to SIS events JSONL file (default: {DEFAULT_SIS_LOG})",
    )
    p.add_argument(
        "--interval",
        type=int,
        default=DEFAULT_INTERVAL,
        help=f"Polling interval in seconds (default: {DEFAULT_INTERVAL})",
    )
    p.add_argument(
        "--once",
        action="store_true",
        help="Print once and exit instead of polling continuously",
    )
    return p


def _run(args: argparse.Namespace) -> None:
    ctrl_path = Path(args.ctrl_log)
    exp_path = Path(args.exp_log)
    sis_path = Path(args.sis_log)

    while True:
        ctrl_state = _parse_log_file(ctrl_path)
        exp_state = _parse_log_file(exp_path)
        sis_events = _parse_sis_events(sis_path)

        table = _render_table(ctrl_state, exp_state, sis_events)
        # Clear screen for live updates (skip when --once so output is pipe-friendly)
        if not args.once:
            print("\033[H\033[J", end="")
        print(table)
        sys.stdout.flush()

        if args.once:
            break

        time.sleep(args.interval)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        _run(args)
    except KeyboardInterrupt:
        print("\nMonitor stopped.")


if __name__ == "__main__":
    main()
