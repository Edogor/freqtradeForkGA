"""T2.2 — Tail-risk metric primitives.

Provides four downside-aware risk measures we currently don't track in
the fitness function:

* **CVaR-95** — Conditional Value-at-Risk at the 95% confidence level
  (a.k.a. Expected Shortfall).  Average loss on the worst 5% of trades.
* **Recovery Factor** — ``net_profit / max_drawdown``.  How many
  "drawdown widths" of profit the strategy produces.
* **Ulcer Index** — RMS of percentage drawdowns at every point of the
  equity curve.  Cares about how *long* you stayed underwater, not just
  how deep.
* **Maximum Adverse Excursion (MAE)** — for a *trade*, the worst
  paper-loss before the trade closed.  Aggregated per-strategy as the
  median MAE across all trades.

These are pure-math helpers with no GA dependencies.  They are wired
into ``FitnessEvaluator`` via :func:`compute_tail_risk_metrics`, which
takes a per-trade profit list (and optional equity curve) and returns
a flat dict ready to be merged into the metrics dict consumed by the
fitness function.

Design constraint: **never raise**.  Bad / empty inputs return safe
defaults (``0.0`` / ``None``) so the GA hot path is unaffected.
"""
from __future__ import annotations

import logging
import math
from typing import Iterable, List, Optional, Sequence

logger = logging.getLogger(__name__)


# ── Primitives ────────────────────────────────────────────────────────


def cvar(returns: Sequence[float], confidence: float = 0.95) -> float:
    """Conditional VaR at ``confidence`` level (expected shortfall).

    Negative number for losses (``-5.0`` ⇒ on the worst 5%, you lose 5
    units on average).  Returns ``0.0`` when there's no data.
    """
    n = len(returns)
    if n == 0:
        return 0.0
    if not (0.0 < confidence < 1.0):
        return 0.0
    losses = sorted(returns)
    # Tail size — at least one observation.  Use a small epsilon to
    # avoid floating-point overshoot (e.g. ``100*0.05`` is 5.0…7 not 5.0).
    tail = max(1, int(math.ceil(n * (1.0 - confidence) - 1e-9)))
    worst = losses[:tail]
    return float(sum(worst) / len(worst))


def recovery_factor(net_profit: float, max_drawdown: float) -> float:
    """``net_profit / max_drawdown``.

    ``max_drawdown`` is interpreted as a positive magnitude (drawdown
    of 15% ⇒ ``0.15`` *or* ``15.0`` — caller's choice; we never look at
    sign).  Returns ``0.0`` when the drawdown is zero or non-finite.
    """
    if not math.isfinite(net_profit) or not math.isfinite(max_drawdown):
        return 0.0
    dd = abs(max_drawdown)
    if dd <= 1e-12:
        # No drawdown observed: if profit positive return a high but
        # finite number so the metric is still ordering-meaningful; if
        # zero/negative, return 0.
        return float(net_profit) * 100.0 if net_profit > 0 else 0.0
    return float(net_profit / dd)


def ulcer_index(equity_curve: Sequence[float]) -> float:
    """Root-mean-square of percentage drawdowns along the equity curve.

    ``equity_curve`` is the running balance over time (any unit, any
    scaling).  Returns ``0.0`` for curves with fewer than two points
    or a non-positive starting balance.
    """
    n = len(equity_curve)
    if n < 2:
        return 0.0
    peak = -math.inf
    squared = 0.0
    count = 0
    for v in equity_curve:
        if v <= 0:
            # Skip non-positive / NaN entries; can't form a percentage.
            continue
        if v > peak:
            peak = v
        if peak > 0:
            dd_pct = (v - peak) / peak * 100.0  # negative or zero
            squared += dd_pct * dd_pct
            count += 1
    if count == 0:
        return 0.0
    return math.sqrt(squared / count)


def max_adverse_excursion(per_trade_mae: Sequence[float]) -> float:
    """Median magnitude of trade-level adverse excursions.

    ``per_trade_mae`` is the list of per-trade MAE values (each as a
    *negative* percentage or absolute drawdown reached during the
    trade).  We return the absolute median so larger = worse.  Empty
    input → ``0.0``.
    """
    vals = [abs(x) for x in per_trade_mae if math.isfinite(x)]
    if not vals:
        return 0.0
    vals.sort()
    mid = len(vals) // 2
    if len(vals) % 2:
        return float(vals[mid])
    return float((vals[mid - 1] + vals[mid]) / 2.0)


# ── High-level helper for FitnessEvaluator integration ────────────────


def compute_tail_risk_metrics(
    trade_returns: Optional[Iterable[float]] = None,
    *,
    equity_curve: Optional[Iterable[float]] = None,
    per_trade_mae: Optional[Iterable[float]] = None,
    net_profit: Optional[float] = None,
    max_drawdown: Optional[float] = None,
    cvar_confidence: float = 0.95,
) -> dict:
    """Compute the full tail-risk metric bundle.

    Any input that is ``None`` or empty simply yields a default for
    the corresponding metric.  All keys are prefixed ``tail_*`` so they
    can't accidentally collide with existing metric names.

    Returned keys:
      * ``tail_cvar_95`` — float, negative for typical losses.
      * ``tail_recovery_factor`` — float.
      * ``tail_ulcer_index`` — float, RMS percentage drawdown.
      * ``tail_mae_median`` — float, absolute median trade-MAE.
    """
    trade_returns = list(trade_returns) if trade_returns is not None else []
    equity_curve = list(equity_curve) if equity_curve is not None else []
    per_trade_mae = list(per_trade_mae) if per_trade_mae is not None else []

    try:
        cvar_val = cvar(trade_returns, confidence=cvar_confidence)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug(f"[TAIL] cvar failed: {exc}")
        cvar_val = 0.0

    try:
        rf_val = recovery_factor(
            net_profit if net_profit is not None else sum(trade_returns),
            max_drawdown if max_drawdown is not None else 0.0,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug(f"[TAIL] recovery_factor failed: {exc}")
        rf_val = 0.0

    try:
        ulcer_val = ulcer_index(equity_curve)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug(f"[TAIL] ulcer_index failed: {exc}")
        ulcer_val = 0.0

    try:
        mae_val = max_adverse_excursion(per_trade_mae)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug(f"[TAIL] mae failed: {exc}")
        mae_val = 0.0

    return {
        "tail_cvar_95": cvar_val,
        "tail_recovery_factor": rf_val,
        "tail_ulcer_index": ulcer_val,
        "tail_mae_median": mae_val,
    }


def normalised_tail_score(metrics: dict) -> dict:
    """Map raw tail metrics to 0-1 ranges suitable for fitness blending.

    Currently a soft heuristic; callers can override with their own
    normalisation.  Returns four keys ``tail_*_norm``.
    """
    cvar95 = metrics.get("tail_cvar_95", 0.0) or 0.0
    rf = metrics.get("tail_recovery_factor", 0.0) or 0.0
    ulcer = metrics.get("tail_ulcer_index", 0.0) or 0.0
    mae = metrics.get("tail_mae_median", 0.0) or 0.0

    # Tail CVaR is negative; map -50% → 0, 0% → 1, clamp.
    cvar_norm = max(0.0, min(1.0, 1.0 + cvar95 / 50.0))
    # Recovery factor: 0 → 0, 5+ → 1 (linear).
    rf_norm = max(0.0, min(1.0, rf / 5.0))
    # Ulcer: 0 → 1, 20+ → 0 (linear, lower is better).
    ulcer_norm = max(0.0, min(1.0, 1.0 - ulcer / 20.0))
    # MAE median: 0 → 1, 20+ → 0 (linear, lower is better).
    mae_norm = max(0.0, min(1.0, 1.0 - mae / 20.0))
    return {
        "tail_cvar_95_norm": cvar_norm,
        "tail_recovery_factor_norm": rf_norm,
        "tail_ulcer_index_norm": ulcer_norm,
        "tail_mae_median_norm": mae_norm,
    }
