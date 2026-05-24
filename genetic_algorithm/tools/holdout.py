"""T1.2 — True Out-of-Time Holdout (Locked-Box) tooling.

Provides three things on top of the legacy ``holdout_test`` block in
``genetic_algorithm/core/evolution.py``:

1. **Lockbox enforcement** — given a ``holdout.lock_start`` date, this
   module ensures the *training* ``backtesting.timerange`` never extends
   into or past that date.  If it does, we clip it (with a loud warning)
   so that the GA can never inadvertently train on what is meant to be
   future hold-out data.
2. **Holdout runner** — a standalone scorer that takes a HoF JSON file,
   runs every entry through :class:`DirectBacktester` on the holdout
   timerange and writes a structured JSON report.  Uses pinned strategy
   code (T1.5) whenever available, and falls back to regenerating from
   the gene if not.
3. **CLI integration** — ``python -m genetic_algorithm holdout {check,run}``
   for ad-hoc inspection and ex-post backtests.

The runner integration is intentionally non-invasive: the only
side-effect during a normal GA run is a single ``enforce_lockbox(config)``
call from the runner, which mutates the in-memory config to clip the
training timerange.
"""
from __future__ import annotations

import csv
import json
import logging
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Default report location is alongside the HoF for easy discovery.
DEFAULT_REPORT_NAME = "holdout_report.json"

_DATE_FORMATS = ("%Y%m%d", "%Y-%m-%d")


# ── Date / timerange helpers ──────────────────────────────────────────


def parse_date(value: str) -> datetime:
    """Parse ``YYYYMMDD`` or ``YYYY-MM-DD`` into a datetime."""
    if not value:
        raise ValueError("Empty date string")
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unrecognised date format: {value!r}")


def parse_timerange(timerange: str) -> Tuple[Optional[datetime], Optional[datetime]]:
    """Parse a freqtrade-style ``START-END`` timerange.

    Either side may be empty (``"-20250101"`` or ``"20240101-"``).
    Returns ``(start, end)`` where each side is ``None`` when omitted.
    """
    if not timerange or "-" not in timerange:
        return None, None
    left, _, right = timerange.partition("-")
    start = parse_date(left) if left else None
    end = parse_date(right) if right else None
    return start, end


def format_date(dt: datetime) -> str:
    return dt.strftime("%Y%m%d")


def format_timerange(start: Optional[datetime], end: Optional[datetime]) -> str:
    a = format_date(start) if start else ""
    b = format_date(end) if end else ""
    return f"{a}-{b}"


# ── Lockbox enforcement (training-side guarantee) ─────────────────────


@dataclass
class LockboxResult:
    """Result of enforcing the holdout lockbox on a config."""

    applied: bool
    lock_start: Optional[datetime]
    original_timerange: Optional[str]
    clipped_timerange: Optional[str]
    holdout_timerange: Optional[str]
    note: str = ""


def enforce_lockbox(config: Dict[str, Any]) -> LockboxResult:
    """Mutate ``config['backtesting']['timerange']`` so it stops at ``lock_start``.

    Reads ``config['holdout']['lock_start']`` (string ``YYYYMMDD``).  When
    enabled and the current training timerange extends to or past that
    date, the timerange is clipped and a warning is logged.  The
    ``LockboxResult.holdout_timerange`` is the suggested
    ``lock_start-original_end`` range for the post-run holdout scorer.

    Always returns; never raises, but logs warnings on inconsistencies.
    """
    holdout_cfg = config.get("holdout") or {}
    if not holdout_cfg.get("enabled"):
        return LockboxResult(False, None, None, None, None,
                             note="holdout.enabled is false")
    raw_lock = holdout_cfg.get("lock_start")
    if not raw_lock:
        logger.warning("[HOLDOUT] holdout.enabled=true but lock_start missing")
        return LockboxResult(False, None, None, None, None,
                             note="holdout.lock_start missing")

    try:
        lock_start = parse_date(str(raw_lock))
    except ValueError as exc:
        logger.warning(f"[HOLDOUT] Invalid lock_start={raw_lock!r}: {exc}")
        return LockboxResult(False, None, None, None, None, note=str(exc))

    bt = config.setdefault("backtesting", {})
    original_tr = bt.get("timerange") or ""
    start, end = parse_timerange(original_tr) if original_tr else (None, None)

    holdout_end = end  # open-ended holdout if no original end
    holdout_tr = format_timerange(lock_start, holdout_end)

    # Decide whether clipping is needed.
    if end is None or end > lock_start:
        new_end = lock_start
        clipped_tr = format_timerange(start, new_end)
        bt["timerange"] = clipped_tr
        logger.warning(
            f"[HOLDOUT] Lockbox active: clipped training timerange "
            f"{original_tr!r} → {clipped_tr!r}; holdout block reserved {holdout_tr!r}"
        )
        return LockboxResult(True, lock_start, original_tr, clipped_tr, holdout_tr,
                             note="clipped")
    # Training already ends at or before lock_start — nothing to clip.
    return LockboxResult(True, lock_start, original_tr, original_tr, holdout_tr,
                         note="training already inside lockbox")


def check_lockbox(config: Dict[str, Any]) -> Dict[str, Any]:
    """Read-only equivalent of :func:`enforce_lockbox` used by the CLI."""
    cfg_copy = {"holdout": dict(config.get("holdout") or {}),
                "backtesting": dict(config.get("backtesting") or {})}
    result = enforce_lockbox(cfg_copy)
    return {
        "enabled": result.applied,
        "lock_start": format_date(result.lock_start) if result.lock_start else None,
        "original_training_timerange": result.original_timerange,
        "clipped_training_timerange": result.clipped_timerange,
        "holdout_timerange": result.holdout_timerange,
        "note": result.note,
    }


# ── Holdout runner (ex-post scorer) ───────────────────────────────────


@dataclass
class HoldoutScore:
    """Per-strategy holdout result."""

    entry_id: str
    fitness: Optional[float] = None
    profit_train: Optional[float] = None
    profit_holdout: Optional[float] = None
    sharpe_holdout: Optional[float] = None
    max_drawdown_holdout: Optional[float] = None
    trades_holdout: Optional[int] = None
    winrate_holdout: Optional[float] = None
    overfit_ratio: Optional[float] = None
    code_was_regenerated: bool = False
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None or k == "error"}


@dataclass
class HoldoutReport:
    """Aggregate report written next to the HoF."""

    hof_file: str
    holdout_timerange: str
    scores: List[HoldoutScore] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    generated_at: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hof_file": self.hof_file,
            "holdout_timerange": self.holdout_timerange,
            "generated_at": self.generated_at,
            "summary": self.summary,
            "scores": [s.to_dict() for s in self.scores],
        }


def _load_hof_entries(hof_file: Path) -> List[Dict[str, Any]]:
    with open(hof_file, "r") as f:
        data = json.load(f)
    return list(data.get("entries", []))


def _backtest_entry_on_holdout(entry: Dict[str, Any], holdout_timerange: str,
                               config: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """Backtest a single HoF entry on ``holdout_timerange``.

    Returns ``(metrics_dict, code_was_regenerated)``.
    """
    from genetic_algorithm.evaluation.direct_backtester import DirectBacktester

    # Prefer pinned code (T1.5); fall back to regeneration from gene.
    strategy_code = entry.get("strategy_code")
    code_was_regenerated = False
    if not strategy_code:
        from genetic_algorithm.genome.codegen import StrategyGenerator
        gene = entry.get("strategy_gene") or {}
        gen = StrategyGenerator(config)
        strategy_code = gen.generate_strategy_code(gene)
        code_was_regenerated = True

    backtester = DirectBacktester(config)
    result = backtester.backtest(
        strategy_code=strategy_code,
        strategy_name=f"holdout_{entry.get('id', 'unknown')[:16]}",
        timerange_override=holdout_timerange,
    )
    return {
        "profit": getattr(result, "profit_percent", 0.0),
        "sharpe": getattr(result, "sharpe_ratio", 0.0),
        "max_drawdown": getattr(result, "max_drawdown", 0.0),
        "trades": getattr(result, "total_trades", 0),
        "winrate": getattr(result, "win_rate", 0.0),
        "success": getattr(result, "success", True),
        "error": getattr(result, "error_message", None),
    }, code_was_regenerated


def run_holdout(
    hof_file: Path,
    holdout_timerange: str,
    config: Dict[str, Any],
    top_n: int = 10,
) -> HoldoutReport:
    """Score the top-N HoF entries on the holdout timerange.

    The strategy code is taken from the pinned snapshot (T1.5) when
    present.  Resulting per-strategy scores include the overfit ratio
    ``(train_profit - holdout_profit) / max(|train_profit|, 1e-6)``.
    """
    import time
    hof_file = Path(hof_file)
    entries = _load_hof_entries(hof_file)
    entries = entries[:top_n]
    report = HoldoutReport(
        hof_file=str(hof_file.resolve()),
        holdout_timerange=holdout_timerange,
        generated_at=time.time(),
    )
    for entry in entries:
        score = HoldoutScore(entry_id=entry.get("id", "unknown"),
                             fitness=entry.get("fitness"),
                             profit_train=(entry.get("metrics") or {}).get("profit"))
        try:
            metrics, regen = _backtest_entry_on_holdout(entry, holdout_timerange, config)
            score.code_was_regenerated = regen
            if not metrics.get("success", True):
                score.error = metrics.get("error") or "backtest reported failure"
            else:
                score.profit_holdout = metrics["profit"]
                score.sharpe_holdout = metrics["sharpe"]
                score.max_drawdown_holdout = metrics["max_drawdown"]
                score.trades_holdout = metrics["trades"]
                score.winrate_holdout = metrics["winrate"]
                if score.profit_train is not None:
                    denom = max(abs(score.profit_train), 1e-6)
                    score.overfit_ratio = (score.profit_train - score.profit_holdout) / denom
        except Exception as exc:
            logger.warning(f"[HOLDOUT] Failed for entry {score.entry_id}: {exc}")
            score.error = str(exc)
        report.scores.append(score)

    profitable = [s for s in report.scores
                  if s.profit_holdout is not None and s.profit_holdout > 0]
    overfit_ratios = [s.overfit_ratio for s in report.scores
                      if s.overfit_ratio is not None]
    report.summary = {
        "total": len(report.scores),
        "profitable_on_holdout": len(profitable),
        "median_overfit_ratio": (
            statistics.median(overfit_ratios) if overfit_ratios else None
        ),
        "median_holdout_profit": (
            statistics.median([s.profit_holdout for s in profitable])
            if profitable else None
        ),
    }
    return report


def write_report(report: HoldoutReport, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report.to_dict(), f, indent=2, default=str)
    return path


def write_report_csv(report: HoldoutReport, path: Path) -> Path:
    """Companion CSV for spreadsheet review."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["entry_id", "fitness", "profit_train", "profit_holdout",
              "sharpe_holdout", "max_drawdown_holdout", "trades_holdout",
              "winrate_holdout", "overfit_ratio", "code_was_regenerated", "error"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in report.scores:
            row = {k: getattr(s, k, None) for k in fields}
            w.writerow(row)
    return path


def default_report_path(hof_file: Path) -> Path:
    """Return ``<hof_dir>/holdout_report.json`` for a HoF file."""
    return Path(hof_file).parent / DEFAULT_REPORT_NAME
