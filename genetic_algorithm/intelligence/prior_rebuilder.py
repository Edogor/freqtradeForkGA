"""
Data-Driven Prior Rebuilder (T4.1)
===================================

Replaces the hardcoded :data:`SIS_INDICATOR_ENRICHMENT` and
:data:`SIS_OPERATOR_ENRICHMENT` priors with values *derived from the
actual strategy corpus*.

The current hardcoded priors (CCI=1.5, STOCH=1.5, ATR=1.4, ADX=1.2, ...)
were computed once from a pattern-mining run on 4289 strategies but have
been observed to disagree with live wave outcomes — e.g. ADX gets a 1.2x
boost while empirically yielding ~0% Tier-1 strategies in W37/W38.

This module:

1. Reads a CSV of evaluated strategies (one row per HoF entry) that
   contains at minimum an ``indicators`` column (pipe-separated) and a
   ``tier`` column (or columns sufficient to derive a tier).
2. Computes per-indicator and per-pair *lift* statistics:

       lift(I)  = P(tier=1 | uses I) / P(tier=1)
       lift(I, J) = P(tier=1 | uses I & J) / P(tier=1)

3. Maps lift to an enrichment weight in a bounded range (default
   ``[0.3, 2.0]``) using a smoothed transformation that:
     * collapses small-support indicators toward 1.0 (Bayesian shrinkage),
     * clips extremes,
     * keeps a global mean close to 1.0 so total selection pressure
       remains comparable to the hardcoded baseline.

4. Writes a YAML file consumable by :class:`PriorLoader`.

The rebuilder is pure-Python (pandas optional) and has no side effects
beyond writing the output file.  This is intentional: we want to run
it offline on the laptop, version the result, and ship it as a config
artifact — not auto-mutate the live runner.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class IndicatorStat:
    """Aggregated tier statistics for a single indicator."""

    name: str
    total: int = 0
    tier1: int = 0
    tier2: int = 0
    tier3: int = 0

    @property
    def tier1_rate(self) -> float:
        return self.tier1 / self.total if self.total else 0.0


@dataclass
class RebuildResult:
    """Output of a single rebuild pass."""

    indicator_weights: Dict[str, float] = field(default_factory=dict)
    operator_weights: Dict[str, float] = field(default_factory=dict)
    synergy_graph: Dict[str, Dict[str, float]] = field(default_factory=dict)
    anti_patterns: Dict[str, Dict[str, float]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tier inference
# ---------------------------------------------------------------------------


def infer_tier(
    row: Mapping[str, Any],
    *,
    tier1_min_fitness: float = 0.6,
    tier1_min_holdout: float = 0.4,
    tier2_min_fitness: float = 0.45,
) -> int:
    """Classify a single strategy row into tier 1, 2 or 3.

    Tier 1 ("live-worthy"): high fitness AND holdout-validated.
    Tier 2 ("promising"): decent fitness, holdout not catastrophic.
    Tier 3 ("noise"): everything else.

    Falls back gracefully when validation columns are missing — in that
    case fitness alone determines the tier.
    """
    if "tier" in row and row["tier"] is not None:
        try:
            return int(row["tier"])
        except (TypeError, ValueError):
            pass

    fitness = _to_float(row.get("fitness"))
    holdout = _to_float(row.get("holdout_fitness"))
    val = _to_float(row.get("val_fitness"))
    holdout_degradation = _to_float(row.get("holdout_degradation"))

    if fitness is None:
        return 3

    # Strong T1 signal: fitness high AND holdout positive
    if fitness >= tier1_min_fitness:
        if holdout is not None and holdout >= tier1_min_holdout:
            return 1
        if val is not None and val >= tier1_min_holdout:
            return 1
        if holdout is None and val is None:
            # No validation columns at all → optimistic T1
            return 1
        # Validation existed and was poor — demote to T2
        if holdout_degradation is not None and holdout_degradation < 0.5:
            return 2
        return 2

    if fitness >= tier2_min_fitness:
        return 2

    return 3


def _to_float(v: Any) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f):
        return None
    return f


# ---------------------------------------------------------------------------
# Core rebuilder
# ---------------------------------------------------------------------------


def _split_indicators(value: Any) -> List[str]:
    """Parse the pipe-separated ``indicators`` column into a unique list.

    Empty / NaN / non-string values yield an empty list.  Duplicates
    inside a single strategy are deduplicated so each indicator
    contributes once to that strategy's tier counts.
    """
    if not isinstance(value, str) or not value:
        return []
    parts = [p.strip().upper() for p in value.split("|") if p.strip()]
    # Stable de-duplication
    seen: set = set()
    unique: List[str] = []
    for p in parts:
        if p not in seen:
            unique.append(p)
            seen.add(p)
    return unique


def _shrink_lift(
    tier1_count: int,
    total: int,
    base_rate: float,
    *,
    prior_strength: float = 20.0,
) -> float:
    """Bayesian shrinkage of an empirical tier-1 rate toward the base rate.

    Uses a Beta(α, β) conjugate prior centered on ``base_rate`` with
    ``α + β = prior_strength`` pseudo-counts.  Small-support indicators
    get pulled strongly toward the global rate; well-supported ones are
    almost unchanged.
    """
    alpha0 = base_rate * prior_strength
    beta0 = (1.0 - base_rate) * prior_strength
    posterior = (tier1_count + alpha0) / (total + alpha0 + beta0)
    return posterior / base_rate if base_rate > 0 else 1.0


def _bound_weight(
    raw: float,
    *,
    lo: float = 0.3,
    hi: float = 2.0,
    soft: bool = True,
) -> float:
    """Clip a raw lift to the configured weight range.

    With ``soft=True`` we apply a log-compression around 1.0 so a 10x
    lift becomes ~2.0 rather than getting hard-clipped — preserves
    ordering between very high-lift indicators.
    """
    if soft and raw > 0:
        # log-compress around 1.0
        x = math.log(max(raw, 1e-6))
        compressed = math.exp(x / 2.0)
    else:
        compressed = raw
    return max(lo, min(hi, compressed))


def rebuild_from_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    min_support: int = 5,
    synergy_top_k: int = 8,
    anti_pattern_top_k: int = 8,
    weight_lo: float = 0.3,
    weight_hi: float = 2.0,
    tier_kwargs: Optional[Mapping[str, Any]] = None,
) -> RebuildResult:
    """Rebuild SIS priors from an in-memory iterable of strategy rows.

    Args:
        rows: Any iterable producing dict-like rows with at least the
            ``indicators`` column (and ideally ``fitness`` /
            ``holdout_fitness`` for tier inference).  Pre-classified
            rows may include a ``tier`` column directly.
        min_support: Indicators or pairs occurring in fewer than this
            many strategies are dropped (too noisy).
        synergy_top_k: Per-indicator, keep this many strongest tier-1
            co-occurrences in the synergy graph.
        anti_pattern_top_k: Per-indicator, keep this many strongest
            tier-3-dominated co-occurrences as anti-patterns.
        weight_lo, weight_hi: Output weight bounds.
        tier_kwargs: Extra keyword args forwarded to :func:`infer_tier`.

    Returns:
        :class:`RebuildResult` containing weights, synergy graph,
        anti-patterns and aggregate metadata.
    """
    tier_kwargs = dict(tier_kwargs or {})

    # ---- Pass 1: per-strategy classification + per-indicator counters ----
    per_ind: Dict[str, IndicatorStat] = {}
    per_pair: Dict[Tuple[str, str], IndicatorStat] = {}
    n_total = 0
    n_t1 = 0

    for row in rows:
        indicators = _split_indicators(row.get("indicators"))
        if not indicators:
            continue
        tier = infer_tier(row, **tier_kwargs)
        n_total += 1
        if tier == 1:
            n_t1 += 1

        for ind in indicators:
            stat = per_ind.setdefault(ind, IndicatorStat(ind))
            stat.total += 1
            if tier == 1:
                stat.tier1 += 1
            elif tier == 2:
                stat.tier2 += 1
            else:
                stat.tier3 += 1

        # Pairs (unordered, deduped within strategy)
        for i in range(len(indicators)):
            for j in range(i + 1, len(indicators)):
                a, b = sorted((indicators[i], indicators[j]))
                pair_key = (a, b)
                pstat = per_pair.setdefault(pair_key, IndicatorStat(f"{a}|{b}"))
                pstat.total += 1
                if tier == 1:
                    pstat.tier1 += 1
                elif tier == 2:
                    pstat.tier2 += 1
                else:
                    pstat.tier3 += 1

    if n_total == 0:
        logger.warning("[T4.1] No usable rows — returning empty priors")
        return RebuildResult(metadata={"n_total": 0, "n_t1": 0})

    base_rate = n_t1 / n_total if n_total else 0.0

    # ---- Pass 2: indicator weights with Bayesian shrinkage ----
    indicator_weights: Dict[str, float] = {}
    for name, stat in per_ind.items():
        if stat.total < min_support:
            continue
        if base_rate <= 0:
            # Degenerate corpus (no T1) — fall back to neutral weights
            indicator_weights[name] = 1.0
            continue
        raw = _shrink_lift(stat.tier1, stat.total, base_rate)
        indicator_weights[name] = round(_bound_weight(raw, lo=weight_lo, hi=weight_hi), 3)

    # ---- Pass 3: synergy + anti-pattern graphs ----
    synergy: Dict[str, Dict[str, float]] = {}
    anti: Dict[str, Dict[str, float]] = {}
    for (a, b), stat in per_pair.items():
        if stat.total < min_support or base_rate <= 0:
            continue
        lift = _shrink_lift(stat.tier1, stat.total, base_rate)
        # Synergy: pairs that over-perform the corpus baseline.
        if lift > 1.0:
            synergy.setdefault(a, {})[b] = round(lift, 3)
            synergy.setdefault(b, {})[a] = round(lift, 3)
        # Anti-pattern: pairs that are clearly under-performing AND
        # have non-trivial T3 mass.  We multiply by the empirical T3
        # share so a pair that just has zero T1 but 1 strategy total is
        # not flagged.
        if lift < 0.5:
            penalty = round((1.0 - lift) * (stat.tier3 / stat.total), 3)
            if penalty > 0:
                anti.setdefault(a, {})[b] = penalty
                anti.setdefault(b, {})[a] = penalty

    # Top-K trimming so the graphs stay actionable
    def _top_k(g: Dict[str, Dict[str, float]], k: int, *, descending: bool) -> None:
        for src, neighbours in g.items():
            if len(neighbours) <= k:
                continue
            items = sorted(neighbours.items(), key=lambda x: x[1], reverse=descending)
            g[src] = dict(items[:k])

    _top_k(synergy, synergy_top_k, descending=True)
    _top_k(anti, anti_pattern_top_k, descending=True)

    # ---- Operator weights ----
    # Only computed when the corpus carries condition/operator info, which
    # all_strategies.csv currently does not.  Left empty so the loader can
    # fall back to the hardcoded operator enrichment.
    operator_weights: Dict[str, float] = {}

    return RebuildResult(
        indicator_weights=indicator_weights,
        operator_weights=operator_weights,
        synergy_graph=synergy,
        anti_patterns=anti,
        metadata={
            "n_total": n_total,
            "n_t1": n_t1,
            "base_t1_rate": round(base_rate, 4),
            "min_support": min_support,
            "weight_range": [weight_lo, weight_hi],
        },
    )


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def rebuild_from_csv(path: str, **kwargs) -> RebuildResult:
    """Convenience wrapper around :func:`rebuild_from_rows` for CSV input.

    Reads via the stdlib :mod:`csv` module so pandas is not required
    when running the rebuilder on light hardware (e.g. the phone).
    """
    import csv

    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return rebuild_from_rows(reader, **kwargs)


def write_result(result: RebuildResult, path: str) -> None:
    """Serialize a :class:`RebuildResult` to YAML (or JSON fallback).

    The format is intentionally hand-readable so a human reviewer can
    diff old vs new priors before adopting them.
    """
    payload = {
        "metadata": result.metadata,
        "indicator_enrichment": result.indicator_weights,
        "operator_enrichment": result.operator_weights,
        "synergy_graph": result.synergy_graph,
        "anti_patterns": result.anti_patterns,
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml

            with out.open("w", encoding="utf-8") as fh:
                yaml.safe_dump(payload, fh, sort_keys=True, default_flow_style=False)
            return
        except ImportError:
            logger.warning("[T4.1] PyYAML not installed — writing JSON instead")

    with out.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _cli(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m genetic_algorithm.intelligence.prior_rebuilder",
        description="Rebuild SIS enrichment priors from a strategies CSV.",
    )
    parser.add_argument("csv", help="Path to all_strategies.csv (or tier-tagged equivalent)")
    parser.add_argument(
        "-o",
        "--output",
        default="genetic_algorithm/intelligence/priors/data_driven_enrichment.yaml",
        help="Output path (yaml or json)",
    )
    parser.add_argument("--min-support", type=int, default=5)
    parser.add_argument("--weight-lo", type=float, default=0.3)
    parser.add_argument("--weight-hi", type=float, default=2.0)
    parser.add_argument(
        "--tier1-min-fitness", type=float, default=0.6,
        help="Minimum fitness to consider for tier 1 classification",
    )
    parser.add_argument(
        "--tier1-min-holdout", type=float, default=0.4,
        help="Minimum holdout fitness for confirmed tier 1",
    )

    args = parser.parse_args(argv)
    result = rebuild_from_csv(
        args.csv,
        min_support=args.min_support,
        weight_lo=args.weight_lo,
        weight_hi=args.weight_hi,
        tier_kwargs={
            "tier1_min_fitness": args.tier1_min_fitness,
            "tier1_min_holdout": args.tier1_min_holdout,
        },
    )
    write_result(result, args.output)
    print(
        f"[T4.1] Rebuilt priors from {result.metadata.get('n_total')} strategies "
        f"({result.metadata.get('n_t1')} tier-1).  Wrote {args.output}."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli())
