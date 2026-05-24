"""T1.1 — Hall of Fame Replay Tool

Re-runs strategies stored in a Hall-of-Fame file against arbitrary backtest
settings.  Two usage modes:

* **replay** — one fee/slippage setting; produces a diff report comparing the
  original metrics (as recorded in the HoF entry) with the freshly measured
  metrics.  Detects code/indicator drift.
* **stress** (T1.6) — grid over ``fee × slippage × max_open_trades``; emits a
  CSV heatmap per entry plus an aggregated robustness score.

The tool is designed to be importable (``replay_entry``, ``stress_entry``,
``replay_hof_file``) and CLI-callable (see ``cli.py``).

Both modes load *pinned* strategy code when available (T1.5); otherwise the
``StrategyGenerator`` is used to regenerate code from the gene dictionary —
in that case a ``code_was_regenerated`` flag is set on every result so the
user knows the replay is *not* immune to drift.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import yaml

logger = logging.getLogger(__name__)


# --- Data classes ----------------------------------------------------------

@dataclass
class ReplayResult:
    """Outcome of replaying a single HoF entry under one cost setting."""
    entry_id: str
    fitness_original: float
    profit_original: float
    sharpe_original: float
    drawdown_original: float
    # Replayed metrics
    success: bool
    profit_replay: float = 0.0
    sharpe_replay: float = 0.0
    drawdown_replay: float = 0.0
    trades_replay: int = 0
    winrate_replay: float = 0.0
    # Diff (replay - original)
    profit_diff: float = 0.0
    sharpe_diff: float = 0.0
    drawdown_diff: float = 0.0
    # Settings used
    fee: float = 0.0
    slippage: float = 0.0
    max_open_trades: int = 0
    timerange: str = ""
    # Provenance
    code_was_regenerated: bool = False
    code_hash_original: Optional[str] = None
    code_hash_replay: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StressReport:
    """Stress-sweep report for one HoF entry."""
    entry_id: str
    grid_results: List[ReplayResult] = field(default_factory=list)
    # Aggregated stats
    profit_min: float = 0.0
    profit_median: float = 0.0
    profit_max: float = 0.0
    survival_rate: float = 0.0  # share of grid points with profit > 0
    robustness_score: float = 0.0  # see ``_compute_robustness`` below

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["grid_results"] = [r.to_dict() for r in self.grid_results]
        return d


# --- Helpers ---------------------------------------------------------------

def _load_hof(hof_path: Path) -> Dict[str, Any]:
    """Load a HoF JSON file, raising a clear error on failure."""
    if not hof_path.exists():
        raise FileNotFoundError(f"Hall-of-Fame file not found: {hof_path}")
    with open(hof_path, "r", encoding="utf-8") as fh:
        try:
            return json.load(fh)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid HoF JSON: {exc}") from exc


def _select_entries(
    hof_data: Dict[str, Any],
    *,
    entry_ids: Optional[Sequence[str]] = None,
    top_n: Optional[int] = None,
    min_fitness: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Filter HoF entries by ID list, top-N (already sorted by fitness desc),
    or minimum fitness threshold.  Returns the raw dicts from the JSON."""
    entries = list(hof_data.get("entries", []))
    if entry_ids:
        wanted = set(entry_ids)
        entries = [e for e in entries if e.get("id") in wanted]
    if min_fitness is not None:
        entries = [e for e in entries if (e.get("fitness") or 0) >= min_fitness]
    if top_n is not None:
        entries = entries[:top_n]
    return entries


def _build_replay_config(
    base_config: Dict[str, Any],
    *,
    fee: Optional[float] = None,
    slippage: Optional[float] = None,
    max_open_trades: Optional[int] = None,
    timerange: Optional[str] = None,
    pairs: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Clone ``base_config`` and apply the replay overrides under ``backtesting``.

    Disables cache so cost variations actually run (the cache key only
    fingerprints the strategy code, not the cost knobs)."""
    import copy
    cfg = copy.deepcopy(base_config)
    bt = cfg.setdefault("backtesting", {})
    if fee is not None:
        bt["fee"] = fee
    if slippage is not None:
        bt["slippage"] = slippage
    if max_open_trades is not None:
        bt["max_open_trades"] = max_open_trades
    if timerange is not None:
        bt["timerange"] = timerange
    if pairs is not None:
        bt["pairs"] = pairs
    # Force cache-off so all grid points hit the backtester even if codes match.
    bt["enable_cache"] = False
    return cfg


def _resolve_strategy_code(
    entry: Dict[str, Any],
    config: Dict[str, Any],
) -> tuple[str, bool, Optional[str]]:
    """Return ``(strategy_code, was_regenerated, original_code_hash)``.

    Prefers the pinned code (T1.5).  Falls back to regenerating via
    ``StrategyGenerator`` when no code was pinned (legacy HoF files)."""
    pinned = entry.get("strategy_code")
    original_hash = entry.get("code_hash")
    if pinned:
        return pinned, False, original_hash

    # Fallback: regenerate from gene dict.
    from genetic_algorithm.genome.codegen import StrategyGenerator
    from genetic_algorithm.core.strategy_gene import StrategyGene
    gen = StrategyGenerator(config)
    gene = StrategyGene.from_dict(entry["strategy_gene"])
    code = gen.generate_strategy_code(gene)
    return code, True, original_hash


def _bt_result_to_metrics(result: Any) -> Dict[str, float]:
    """Extract a compact metric dict from a ``BacktestResult``."""
    return {
        "profit": float(getattr(result, "profit_percent", 0.0) or 0.0),
        "sharpe": float(getattr(result, "sharpe_ratio", 0.0) or 0.0),
        "drawdown": float(getattr(result, "max_drawdown", 0.0) or 0.0),
        "trades": int(getattr(result, "total_trades", 0) or 0),
        "winrate": float(getattr(result, "win_rate", 0.0) or 0.0),
    }


# --- Public API ------------------------------------------------------------

def replay_entry(
    entry: Dict[str, Any],
    base_config: Dict[str, Any],
    backtester: Any,
    *,
    fee: Optional[float] = None,
    slippage: Optional[float] = None,
    max_open_trades: Optional[int] = None,
    timerange: Optional[str] = None,
    pairs: Optional[List[str]] = None,
) -> ReplayResult:
    """Replay a single HoF entry under the given cost setting."""
    import hashlib

    entry_id = entry.get("id", "<unknown>")
    orig_metrics = entry.get("metrics", {}) or {}
    fit_orig = float(entry.get("fitness", 0) or 0)
    p_orig = float(orig_metrics.get("profit", 0) or 0)
    s_orig = float(orig_metrics.get("sharpe_ratio", 0) or 0)
    d_orig = float(orig_metrics.get("max_drawdown", 0) or 0)

    try:
        code, regenerated, original_hash = _resolve_strategy_code(entry, base_config)
        # Replayed hash (post-regen if applicable)
        replay_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()
    except Exception as exc:
        return ReplayResult(
            entry_id=entry_id,
            fitness_original=fit_orig,
            profit_original=p_orig, sharpe_original=s_orig, drawdown_original=d_orig,
            success=False,
            fee=fee or 0, slippage=slippage or 0,
            max_open_trades=max_open_trades or 0, timerange=timerange or "",
            error=f"code resolution failed: {exc}",
        )

    # Build replay config, mutate the backtester to use it.
    cfg = _build_replay_config(
        base_config,
        fee=fee, slippage=slippage, max_open_trades=max_open_trades,
        timerange=timerange, pairs=pairs,
    )
    # The DirectBacktester reads ``self.backtest_config`` on every call, so we
    # temporarily override it.  No instance is mutated permanently when the
    # caller passes a fresh backtester per replay session.
    prev_config = backtester.config
    prev_bt_config = backtester.backtest_config
    backtester.config = cfg
    backtester.backtest_config = cfg.get("backtesting", {})
    try:
        result = backtester.backtest_strategy(
            strategy_code=code,
            strategy_name=f"replay_{entry_id}_{int(time.time())}",
            timerange_override=timerange,
            pairs_override=pairs,
            strategy_max_open_trades=max_open_trades,
        )
    except Exception as exc:
        backtester.config = prev_config
        backtester.backtest_config = prev_bt_config
        return ReplayResult(
            entry_id=entry_id,
            fitness_original=fit_orig,
            profit_original=p_orig, sharpe_original=s_orig, drawdown_original=d_orig,
            success=False,
            fee=fee or 0, slippage=slippage or 0,
            max_open_trades=max_open_trades or 0, timerange=timerange or "",
            code_was_regenerated=regenerated,
            code_hash_original=original_hash,
            code_hash_replay=replay_hash,
            error=f"backtest crashed: {exc}",
        )
    finally:
        backtester.config = prev_config
        backtester.backtest_config = prev_bt_config

    m = _bt_result_to_metrics(result)
    return ReplayResult(
        entry_id=entry_id,
        fitness_original=fit_orig,
        profit_original=p_orig, sharpe_original=s_orig, drawdown_original=d_orig,
        success=bool(getattr(result, "success", False)),
        profit_replay=m["profit"], sharpe_replay=m["sharpe"],
        drawdown_replay=m["drawdown"], trades_replay=m["trades"],
        winrate_replay=m["winrate"],
        profit_diff=m["profit"] - p_orig,
        sharpe_diff=m["sharpe"] - s_orig,
        drawdown_diff=m["drawdown"] - d_orig,
        fee=fee if fee is not None else float(base_config.get("backtesting", {}).get("fee", 0)),
        slippage=slippage if slippage is not None else float(base_config.get("backtesting", {}).get("slippage", 0)),
        max_open_trades=max_open_trades if max_open_trades is not None else int(base_config.get("backtesting", {}).get("max_open_trades", 0)),
        timerange=timerange or base_config.get("backtesting", {}).get("timerange", ""),
        code_was_regenerated=regenerated,
        code_hash_original=original_hash,
        code_hash_replay=replay_hash,
        error=getattr(result, "error_message", None),
    )


def stress_entry(
    entry: Dict[str, Any],
    base_config: Dict[str, Any],
    backtester: Any,
    *,
    fees: Sequence[float],
    slippages: Sequence[float],
    max_open_trades_grid: Optional[Sequence[int]] = None,
    timerange: Optional[str] = None,
    pairs: Optional[List[str]] = None,
) -> StressReport:
    """Run the cost-stress grid for one HoF entry (T1.6)."""
    motgrid: Sequence[int] = max_open_trades_grid or (None,)  # type: ignore[assignment]
    rep = StressReport(entry_id=entry.get("id", "<unknown>"))
    for fee in fees:
        for slip in slippages:
            for mot in motgrid:
                res = replay_entry(
                    entry, base_config, backtester,
                    fee=fee, slippage=slip,
                    max_open_trades=mot if mot is not None else None,
                    timerange=timerange, pairs=pairs,
                )
                rep.grid_results.append(res)

    # Aggregated stats: ignore failed points.
    profits = [r.profit_replay for r in rep.grid_results if r.success]
    if profits:
        rep.profit_min = min(profits)
        rep.profit_max = max(profits)
        rep.profit_median = sorted(profits)[len(profits) // 2]
        rep.survival_rate = sum(1 for p in profits if p > 0) / len(profits)
        rep.robustness_score = _compute_robustness(profits)
    return rep


def _compute_robustness(profits: List[float]) -> float:
    """Robustness = ``profit_median`` scaled by ``1 - CV`` (clamped to >=0).

    Encourages strategies whose profit stays roughly the same across the
    cost grid, not just whose median is highest."""
    if not profits:
        return 0.0
    n = len(profits)
    mean = sum(profits) / n
    if mean == 0:
        return 0.0
    var = sum((p - mean) ** 2 for p in profits) / n
    std = var ** 0.5
    cv = abs(std / mean) if mean else 1.0
    median = sorted(profits)[n // 2]
    return float(median * max(0.0, 1.0 - cv))


def replay_hof_file(
    hof_path: Path,
    base_config: Dict[str, Any],
    *,
    fee: Optional[float] = None,
    slippage: Optional[float] = None,
    timerange: Optional[str] = None,
    pairs: Optional[List[str]] = None,
    entry_ids: Optional[Sequence[str]] = None,
    top_n: Optional[int] = None,
    min_fitness: Optional[float] = None,
    backtester: Any = None,
) -> List[ReplayResult]:
    """Replay an entire HoF file (or a subset).

    Returns one ``ReplayResult`` per inspected entry.  Caller decides how to
    render / persist them (CLI renders a table; tests use the list directly).
    """
    hof_data = _load_hof(hof_path)
    entries = _select_entries(
        hof_data, entry_ids=entry_ids, top_n=top_n, min_fitness=min_fitness,
    )
    if not entries:
        logger.warning("No HoF entries matched the filters.")
        return []

    if backtester is None:
        from genetic_algorithm.evaluation.direct_backtester import DirectBacktester
        backtester = DirectBacktester(base_config)

    results: List[ReplayResult] = []
    for e in entries:
        r = replay_entry(
            e, base_config, backtester,
            fee=fee, slippage=slippage,
            timerange=timerange, pairs=pairs,
        )
        results.append(r)
    return results


def stress_hof_file(
    hof_path: Path,
    base_config: Dict[str, Any],
    *,
    fees: Sequence[float],
    slippages: Sequence[float],
    max_open_trades_grid: Optional[Sequence[int]] = None,
    entry_ids: Optional[Sequence[str]] = None,
    top_n: Optional[int] = None,
    min_fitness: Optional[float] = None,
    timerange: Optional[str] = None,
    pairs: Optional[List[str]] = None,
    backtester: Any = None,
) -> List[StressReport]:
    """Cost-stress sweep across (selected) entries in a HoF file (T1.6)."""
    hof_data = _load_hof(hof_path)
    entries = _select_entries(
        hof_data, entry_ids=entry_ids, top_n=top_n, min_fitness=min_fitness,
    )
    if not entries:
        return []
    if backtester is None:
        from genetic_algorithm.evaluation.direct_backtester import DirectBacktester
        backtester = DirectBacktester(base_config)
    return [
        stress_entry(
            e, base_config, backtester,
            fees=fees, slippages=slippages,
            max_open_trades_grid=max_open_trades_grid,
            timerange=timerange, pairs=pairs,
        )
        for e in entries
    ]


# --- Rendering / persistence helpers --------------------------------------

def write_replay_csv(results: Iterable[ReplayResult], path: Path) -> None:
    """Persist replay results as CSV for downstream analysis."""
    rows = [r.to_dict() for r in results]
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_stress_csv(reports: Iterable[StressReport], path: Path) -> None:
    """Persist stress-sweep grid points as a single flat CSV."""
    flat: List[Dict[str, Any]] = []
    for rep in reports:
        for gp in rep.grid_results:
            row = gp.to_dict()
            row["robustness_score"] = rep.robustness_score
            row["survival_rate"] = rep.survival_rate
            flat.append(row)
    if not flat:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(flat[0].keys()))
        writer.writeheader()
        writer.writerows(flat)


def render_replay_table(results: List[ReplayResult]) -> str:
    """Return a human-readable replay-result table (Markdown-ish)."""
    if not results:
        return "No replay results."
    header = (
        f"{'entry':<18}{'orig_profit':>12}{'replay_profit':>14}{'diff':>10}"
        f"{'sharpe_diff':>12}{'dd_diff':>10}{'trades':>8}  status"
    )
    lines = [header, "-" * len(header)]
    for r in results:
        status = "OK" if r.success else (r.error or "FAIL")[:30]
        if r.code_was_regenerated:
            status = "regen " + status
        lines.append(
            f"{r.entry_id:<18}{r.profit_original:>12.2f}{r.profit_replay:>14.2f}"
            f"{r.profit_diff:>+10.2f}{r.sharpe_diff:>+12.2f}{r.drawdown_diff:>+10.2f}"
            f"{r.trades_replay:>8}  {status}"
        )
    return "\n".join(lines)


def render_stress_table(reports: List[StressReport]) -> str:
    """Return a per-entry robustness summary table."""
    if not reports:
        return "No stress reports."
    header = f"{'entry':<18}{'profit_min':>12}{'profit_med':>12}{'profit_max':>12}{'survival':>10}{'robust':>10}  grid_points"
    lines = [header, "-" * len(header)]
    for r in reports:
        lines.append(
            f"{r.entry_id:<18}{r.profit_min:>12.2f}{r.profit_median:>12.2f}"
            f"{r.profit_max:>12.2f}{r.survival_rate:>10.2f}{r.robustness_score:>10.2f}"
            f"  {len(r.grid_results)}"
        )
    return "\n".join(lines)


def list_hof_entries(hof_path: Path, limit: int = 10) -> str:
    """Render a short ``hof list`` table for the CLI."""
    data = _load_hof(hof_path)
    entries = data.get("entries", [])[:limit]
    if not entries:
        return f"No entries in {hof_path}"
    lines = [f"{'id':<18}{'fitness':>10}{'profit':>10}{'sharpe':>8}{'dd':>8}{'gen':>5}  pinned  run_id"]
    lines.append("-" * len(lines[0]))
    for e in entries:
        m = e.get("metrics", {}) or {}
        pinned = "yes" if e.get("strategy_code") else " no"
        lines.append(
            f"{(e.get('id') or '')[:18]:<18}"
            f"{(e.get('fitness') or 0):>10.4f}"
            f"{(m.get('profit') or 0):>10.2f}"
            f"{(m.get('sharpe_ratio') or 0):>8.2f}"
            f"{(m.get('max_drawdown') or 0):>8.2f}"
            f"{(e.get('generation_found') or 0):>5}"
            f"  {pinned}    {(e.get('run_id') or '')[:24]}"
        )
    return "\n".join(lines)


def show_hof_entry(hof_path: Path, entry_id: str) -> str:
    """Render the full detail view for a single HoF entry."""
    data = _load_hof(hof_path)
    for e in data.get("entries", []):
        if e.get("id") == entry_id:
            # Hide the code body unless asked for; mention if present.
            code = e.get("strategy_code")
            preview = {k: v for k, v in e.items() if k != "strategy_code"}
            preview["strategy_code"] = f"<{len(code)} chars, pinned>" if code else None
            return json.dumps(preview, indent=2, default=str)
    return f"Entry not found: {entry_id}"


def load_base_config(config_arg: str) -> Dict[str, Any]:
    """Load a base config for the replay backtester.

    ``config_arg`` may be a path or a preset name (resolved like ``cli run``).
    """
    from genetic_algorithm.config.schema import load_config
    p = Path(config_arg)
    if not p.exists():
        preset_path = Path("genetic_algorithm/config/presets") / f"{config_arg}.yaml"
        if preset_path.exists():
            p = preset_path
        else:
            raise FileNotFoundError(f"Config not found: {config_arg}")
    return load_config(p)
