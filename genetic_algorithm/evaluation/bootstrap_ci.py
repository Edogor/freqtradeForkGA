"""T2.6 — Bootstrap confidence-interval estimator for trade returns.

When enabled in the config (``fitness.use_bootstrap_ci: true``),
``FitnessEvaluator`` replaces the raw ``profit`` metric with the lower
``ci`` percentile of a bootstrap distribution of mean trade return
multiplied by the trade count.  This is a "conservative profit"
estimate that strongly punishes strategies whose profit hinges on a
handful of lucky trades.

The estimator is intentionally dependency-free (uses ``random``,
``statistics``) so it works on the server and in unit tests without
NumPy on the hot path.

Public API:

    bootstrap_lower_ci(trade_returns, n_iter=1000, ci=0.05, seed=None)
        Returns a float: the lower-ci percentile of bootstrap means
        of *cumulative* trade return.  Empty / single-trade inputs
        return 0.0 (we can't bootstrap a single point).
"""
from __future__ import annotations

import logging
import math
import random
from typing import Optional, Sequence

logger = logging.getLogger(__name__)


def bootstrap_lower_ci(
    trade_returns: Sequence[float],
    n_iter: int = 1000,
    ci: float = 0.05,
    seed: Optional[int] = None,
) -> float:
    """Lower-percentile of bootstrap distribution of total trade return.

    Args:
        trade_returns: list of per-trade returns (any unit, typically
            ratios or percentages).  Must contain at least 2 entries
            for a meaningful bootstrap.
        n_iter: number of bootstrap resamples (with replacement).
        ci: lower percentile in [0, 1].  ``0.05`` = "the 5th
            percentile of bootstrap totals".
        seed: optional RNG seed for reproducibility.

    Returns:
        Lower-CI estimate of total return (sum of all trade returns
        on a resampled sample of the same length).  Returns ``0.0``
        on bad inputs — never raises.
    """
    n = len(trade_returns)
    if n < 2 or n_iter < 1:
        return 0.0
    if not (0.0 < ci < 1.0):
        return 0.0

    # Sanitize: drop NaN / Inf, return 0 if too few finite trades.
    finite = [float(x) for x in trade_returns
              if isinstance(x, (int, float)) and math.isfinite(x)]
    if len(finite) < 2:
        return 0.0

    rng = random.Random(seed)
    n_finite = len(finite)
    totals = []
    for _ in range(n_iter):
        # Resample n_finite trades with replacement and sum.
        resampled_total = 0.0
        for _ in range(n_finite):
            resampled_total += finite[rng.randrange(n_finite)]
        totals.append(resampled_total)

    totals.sort()
    # Lower-ci percentile via nearest-rank method.
    rank = max(0, min(len(totals) - 1, int(math.floor(ci * len(totals)))))
    return float(totals[rank])


def is_bootstrap_ci_enabled(config: dict) -> bool:
    """Return True iff the config opts in to bootstrap-CI profit
    replacement.  Looks at both ``fitness.use_bootstrap_ci`` and the
    legacy ``bootstrap_ci.enabled`` for forward compatibility.
    """
    if not isinstance(config, dict):
        return False
    fitness = config.get('fitness') or {}
    if isinstance(fitness, dict) and bool(fitness.get('use_bootstrap_ci', False)):
        return True
    bci = config.get('bootstrap_ci') or {}
    if isinstance(bci, dict) and bool(bci.get('enabled', False)):
        return True
    return False


def bootstrap_ci_settings(config: dict) -> dict:
    """Extract bootstrap CI tuning knobs from the config.

    Returns dict with ``n_iter``, ``ci``, ``seed`` (None when not set).
    Falls back to safe defaults.
    """
    settings = {}
    if isinstance(config, dict):
        fitness = config.get('fitness') or {}
        bci = config.get('bootstrap_ci') or {}
        # fitness block takes priority over the legacy block
        merged = {}
        if isinstance(bci, dict):
            merged.update(bci)
        if isinstance(fitness, dict):
            merged.update({
                k.replace('bootstrap_', ''): v
                for k, v in fitness.items()
                if k.startswith('bootstrap_')
            })
        settings = merged

    return {
        'n_iter': int(settings.get('n_iter', 1000)),
        'ci': float(settings.get('ci', 0.05)),
        'seed': settings.get('seed'),  # may be None
    }
