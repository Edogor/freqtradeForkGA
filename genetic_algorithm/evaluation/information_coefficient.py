"""T2.5 — Information Coefficient (IC) primitives and fitness penalty.

The Information Coefficient is the Spearman rank correlation between
an *indicator value* and the *forward return* over a fixed horizon.
High |IC| means the indicator carries genuine predictive information;
near-zero IC indicates the indicator is essentially noise.

We expose:

* :func:`spearman_ic`       — pure-Python rank correlation
* :func:`compute_ic_table`  — bulk: indicator_name -> IC
* :func:`median_abs_ic`     — summary statistic used by the penalty
* :func:`low_ic_penalty`    — multiplicative penalty in ``[0, 1]``

The current ``FitnessEvaluator`` integration is *passive*: if the
upstream evaluator stores ``median_ic_magnitude`` (and/or a full
``indicator_ics`` dict) in the metrics, the penalty is applied during
:py:meth:`FitnessEvaluator._apply_penalties`.  When the key is absent
the penalty is a no-op, so this module is safe to land without a
strategy-generator change.

Design constraint: **never raise**.  Bad input returns ``0.0``.
"""
from __future__ import annotations

import logging
import math
from typing import Dict, Iterable, List, Sequence, Tuple

logger = logging.getLogger(__name__)


# ── Rank helpers ──────────────────────────────────────────────────────


def _ranks(values: Sequence[float]) -> List[float]:
    """Return fractional (average-tie-broken) ranks of ``values``."""
    n = len(values)
    indexed = sorted(((v, i) for i, v in enumerate(values)), key=lambda x: x[0])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        # Group all ties together
        while j + 1 < n and indexed[j + 1][0] == indexed[i][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-based
        for k in range(i, j + 1):
            _, orig_idx = indexed[k]
            ranks[orig_idx] = avg_rank
        i = j + 1
    return ranks


# ── Core IC ───────────────────────────────────────────────────────────


def spearman_ic(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation between two equal-length sequences.

    Returns ``0.0`` on bad / degenerate inputs (length mismatch,
    constant series, < 2 observations).
    """
    n = len(x)
    if n < 2 or n != len(y):
        return 0.0

    # Filter pairs where either side is NaN/Inf to avoid poisoning ranks
    pairs = [
        (float(a), float(b))
        for a, b in zip(x, y)
        if isinstance(a, (int, float)) and isinstance(b, (int, float))
        and math.isfinite(a) and math.isfinite(b)
    ]
    if len(pairs) < 2:
        return 0.0

    xs, ys = zip(*pairs)
    if all(v == xs[0] for v in xs) or all(v == ys[0] for v in ys):
        return 0.0  # zero variance → undefined

    rx = _ranks(xs)
    ry = _ranks(ys)
    n = len(rx)
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n
    num = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    denom_x = math.sqrt(sum((r - mean_rx) ** 2 for r in rx))
    denom_y = math.sqrt(sum((r - mean_ry) ** 2 for r in ry))
    if denom_x == 0 or denom_y == 0:
        return 0.0
    return float(num / (denom_x * denom_y))


# ── Bulk helpers ──────────────────────────────────────────────────────


def compute_ic_table(
    indicators: Dict[str, Sequence[float]],
    forward_returns: Sequence[float],
) -> Dict[str, float]:
    """Compute IC for each indicator vs the same ``forward_returns``.

    Returns ``{indicator_name: IC}``.  Indicators that fail return 0.0.
    """
    table: Dict[str, float] = {}
    if not indicators or not forward_returns:
        return table
    for name, series in indicators.items():
        try:
            table[name] = spearman_ic(series, forward_returns)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug(f"[IC] spearman_ic({name}) failed: {exc}")
            table[name] = 0.0
    return table


def median_abs_ic(ic_table: Dict[str, float]) -> float:
    """Median magnitude of ICs in ``ic_table``.  Empty → ``0.0``."""
    if not ic_table:
        return 0.0
    vals = sorted(abs(v) for v in ic_table.values() if isinstance(v, (int, float))
                  and math.isfinite(v))
    if not vals:
        return 0.0
    mid = len(vals) // 2
    if len(vals) % 2:
        return float(vals[mid])
    return float((vals[mid - 1] + vals[mid]) / 2.0)


# ── Penalty ───────────────────────────────────────────────────────────


def low_ic_penalty(
    median_ic: float,
    threshold: float = 0.02,
    max_penalty: float = 0.30,
) -> float:
    """Multiplicative fitness penalty when ``median_ic`` < ``threshold``.

    Returns ``1.0`` (no penalty) for ``median_ic >= threshold``.
    Returns ``1.0 - max_penalty`` when median_ic == 0.
    Smooth linear ramp in between.

    Result is always in ``[1.0 - max_penalty, 1.0]``.
    """
    if not math.isfinite(median_ic) or threshold <= 0:
        return 1.0
    if median_ic >= threshold:
        return 1.0
    # Linear: 0 → 1-max_penalty, threshold → 1.
    severity = 1.0 - (median_ic / threshold)
    severity = max(0.0, min(1.0, severity))
    return 1.0 - max_penalty * severity


def ic_penalty_settings(config: dict) -> Tuple[bool, float, float]:
    """Read IC-penalty knobs from config.

    Looks at ``fitness.ic_penalty``:
        enabled: bool (default False)
        threshold: float (default 0.02)
        max_penalty: float (default 0.30)

    Returns ``(enabled, threshold, max_penalty)``.
    """
    if not isinstance(config, dict):
        return (False, 0.02, 0.30)
    fitness = config.get('fitness') or {}
    block = fitness.get('ic_penalty') if isinstance(fitness, dict) else None
    if not isinstance(block, dict):
        return (False, 0.02, 0.30)
    return (
        bool(block.get('enabled', False)),
        float(block.get('threshold', 0.02)),
        float(block.get('max_penalty', 0.30)),
    )
