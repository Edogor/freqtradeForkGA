"""T3.5 — Genome-Distance Niching.

A richer genotype distance that captures parameter-level differences and
condition-graph structure, complementing the legacy structural distance
in ``engine/population.calculate_strategy_distance``.

The legacy distance only considers
  * the *set* of indicator *types*
  * counts of entry/exit conditions
  * timeframe, stoploss magnitude, trailing-stop flag

so two strategies that share the same indicator types but use wildly
different parameters or conditions are reported as identical.  This
hurts fitness sharing: the GA can't distinguish near-duplicates from
true variants and prematurely collapses to a single niche.

This module provides :func:`calculate_genome_distance` which returns a
value in ``[0, 1]`` and weighs:

1. Indicator *signature* multiset distance — type + bucketed parameters
2. Per-type parameter distance for matched indicators
3. Condition signature multiset distance — (indicator, operator,
   bucketed threshold) tuples across entry + exit
4. Continuous risk-parameter distance — stoploss, trailing offsets,
   ROI-curve integral
5. Categorical/structural flags — timeframe equality, trailing on,
   regime on

The function is pure-Python (no NumPy needed) and tolerant of legacy
gene shapes — missing optional attributes are treated as their defaults.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Weights — sum to 1.0
W_INDICATOR_SIG = 0.30
W_INDICATOR_PARAMS = 0.20
W_CONDITION_SIG = 0.25
W_RISK_PARAMS = 0.15
W_FLAGS = 0.10


# ── Helpers ───────────────────────────────────────────────────────────


def _bucket(value: float, scale: float = 1.0, buckets: int = 10) -> int:
    """Round a continuous value into one of ``buckets`` discrete bins.

    Used to coarsen indicator parameters so that "period=14" and
    "period=15" collapse to the same signature, while "period=14" and
    "period=50" do not.
    """
    if value is None:
        return -1
    try:
        v = float(value)
    except (TypeError, ValueError):
        return hash(str(value)) % 997
    if scale <= 0:
        return 0
    return int(round((v / scale) * buckets))


def _indicator_signature(ind) -> Tuple:
    """Coarse signature of an indicator used for multiset matching.

    ``(type, frozenset of (param_name, bucketed_value)) — timeframe``
    """
    params: Dict[str, Any] = getattr(ind, "parameters", {}) or {}
    bucketed = []
    for k in sorted(params.keys()):
        v = params[k]
        if isinstance(v, bool):
            bucketed.append((k, "bool", bool(v)))
        elif isinstance(v, (int, float)):
            # Period-like params are typically in [2, 200] → scale 10
            bucketed.append((k, "num", _bucket(float(v), scale=10.0, buckets=2)))
        else:
            bucketed.append((k, "cat", str(v)))
    tf = getattr(ind, "timeframe", None)
    return (getattr(ind, "type", "?"), tuple(bucketed), tf)


def _condition_signature(cond) -> Tuple:
    """Signature for a ConditionGene used in multiset matching."""
    threshold = getattr(cond, "threshold", 0.0)
    return (
        getattr(cond, "indicator", "?"),
        getattr(cond, "operator", "?"),
        _bucket(float(threshold or 0.0), scale=10.0, buckets=2),
        getattr(cond, "logic", "AND"),
        int(getattr(cond, "lookback", 0) or 0) // 5,  # coarse
    )


def _multiset_jaccard(a: Iterable, b: Iterable) -> float:
    """Multiset Jaccard distance in [0, 1].

    ``1 - |A ∩ B| / |A ∪ B|`` with multiset semantics.  Returns 0 if
    both collections are empty (identical), 1 if disjoint.
    """
    ca, cb = Counter(a), Counter(b)
    if not ca and not cb:
        return 0.0
    intersection = sum((ca & cb).values())
    union = sum((ca | cb).values())
    if union == 0:
        return 0.0
    return 1.0 - (intersection / union)


def _param_vector_distance(p1: Dict[str, Any], p2: Dict[str, Any]) -> float:
    """Normalised distance between two parameter dicts in [0, 1].

    For each shared numeric key, compute ``|a-b| / max(|a|, |b|, 1)``;
    average over the union of keys; missing keys count as full
    distance.
    """
    keys = set(p1.keys()) | set(p2.keys())
    if not keys:
        return 0.0
    total = 0.0
    for k in keys:
        if k not in p1 or k not in p2:
            total += 1.0
            continue
        v1, v2 = p1[k], p2[k]
        if isinstance(v1, bool) or isinstance(v2, bool):
            total += 0.0 if v1 == v2 else 1.0
            continue
        try:
            a, b = float(v1), float(v2)
        except (TypeError, ValueError):
            total += 0.0 if v1 == v2 else 1.0
            continue
        denom = max(abs(a), abs(b), 1.0)
        total += min(1.0, abs(a - b) / denom)
    return total / len(keys)


def _indicator_param_distance(inds1: List, inds2: List) -> float:
    """Average per-type parameter distance for indicator types shared
    between the two strategies.  Falls back to 0.5 when none overlap
    so that identical-type-set strategies still register some distance
    (and disjoint sets fall back to the indicator-signature term)."""
    if not inds1 or not inds2:
        return 0.0 if (not inds1 and not inds2) else 1.0
    by_type1: Dict[str, List[Dict]] = {}
    by_type2: Dict[str, List[Dict]] = {}
    for i in inds1:
        by_type1.setdefault(getattr(i, "type", "?"), []).append(
            getattr(i, "parameters", {}) or {}
        )
    for i in inds2:
        by_type2.setdefault(getattr(i, "type", "?"), []).append(
            getattr(i, "parameters", {}) or {}
        )
    shared = set(by_type1) & set(by_type2)
    if not shared:
        return 1.0
    per_type = []
    for t in shared:
        # Greedy 1-1 matching: for each parameter dict in the smaller
        # list, find the closest in the other.
        a_list = by_type1[t]
        b_list = by_type2[t]
        if len(a_list) > len(b_list):
            a_list, b_list = b_list, a_list
        dists = []
        used = set()
        for pa in a_list:
            best = 1.0
            best_idx = -1
            for j, pb in enumerate(b_list):
                if j in used:
                    continue
                d = _param_vector_distance(pa, pb)
                if d < best:
                    best, best_idx = d, j
            if best_idx >= 0:
                used.add(best_idx)
            dists.append(best)
        # Penalise extra unmatched indicators in the longer list
        leftover = len(b_list) - len(a_list)
        per_type.append((sum(dists) + leftover) / max(len(a_list) + leftover, 1))
    return sum(per_type) / len(per_type)


def _risk_parameter_distance(g1, g2) -> float:
    """Distance over stoploss / trailing offsets / ROI curve."""
    comps = []
    s1 = float(getattr(g1, "stoploss", -0.1) or -0.1)
    s2 = float(getattr(g2, "stoploss", -0.1) or -0.1)
    denom = max(abs(s1), abs(s2), 0.05)
    comps.append(min(1.0, abs(s1 - s2) / denom))

    for attr in ("trailing_stop_positive", "trailing_stop_positive_offset"):
        a = float(getattr(g1, attr, 0.0) or 0.0)
        b = float(getattr(g2, attr, 0.0) or 0.0)
        denom = max(abs(a), abs(b), 0.05)
        comps.append(min(1.0, abs(a - b) / denom))

    # ROI curve integral (sum of value * duration_weight): proxy for
    # how aggressive the take-profit ladder is.
    def roi_signature(g) -> List[Tuple[int, float]]:
        roi = getattr(g, "minimal_roi", {}) or {}
        pts: List[Tuple[int, float]] = []
        for k, v in roi.items():
            try:
                pts.append((int(k), float(v)))
            except (TypeError, ValueError):
                continue
        return sorted(pts)

    r1, r2 = roi_signature(g1), roi_signature(g2)
    if r1 and r2:
        # Sample at each unique time point and compare values.
        times = sorted(set(t for t, _ in r1) | set(t for t, _ in r2))

        def at(times_pts: List[Tuple[int, float]], t: int) -> float:
            last = 0.0
            for tt, vv in times_pts:
                if tt <= t:
                    last = vv
                else:
                    break
            return last

        diffs = []
        for t in times:
            a, b = at(r1, t), at(r2, t)
            denom = max(abs(a), abs(b), 0.01)
            diffs.append(min(1.0, abs(a - b) / denom))
        comps.append(sum(diffs) / len(diffs) if diffs else 0.0)
    elif r1 or r2:
        comps.append(1.0)
    else:
        comps.append(0.0)
    return sum(comps) / len(comps) if comps else 0.0


def _flag_distance(g1, g2) -> float:
    flags = []
    flags.append(0.0 if getattr(g1, "timeframe", None) == getattr(g2, "timeframe", None) else 1.0)
    flags.append(0.0 if bool(getattr(g1, "trailing_stop", False)) == bool(getattr(g2, "trailing_stop", False)) else 1.0)
    # Regime flag (optional)
    r1 = getattr(g1, "regime", None)
    r2 = getattr(g2, "regime", None)
    e1 = bool(getattr(r1, "enabled", False)) if r1 else False
    e2 = bool(getattr(r2, "enabled", False)) if r2 else False
    flags.append(0.0 if e1 == e2 else 1.0)
    return sum(flags) / len(flags) if flags else 0.0


# ── Public API ────────────────────────────────────────────────────────


def calculate_genome_distance(ind1, ind2) -> float:
    """Compute the v2 genome distance between two evaluated or
    un-evaluated individuals.  Returns a value in ``[0, 1]``.

    Inputs are :class:`Individual` instances; only ``ind.strategy_gene``
    is inspected, so this is independent of whether the individuals
    have been backtested yet.
    """
    g1 = getattr(ind1, "strategy_gene", ind1)
    g2 = getattr(ind2, "strategy_gene", ind2)
    if g1 is None or g2 is None:
        return 1.0

    inds1 = list(getattr(g1, "indicators", []) or [])
    inds2 = list(getattr(g2, "indicators", []) or [])

    # 1. Indicator signature multiset
    sig1 = [_indicator_signature(i) for i in inds1]
    sig2 = [_indicator_signature(i) for i in inds2]
    d_ind_sig = _multiset_jaccard(sig1, sig2)

    # 2. Per-type parameter distance
    d_ind_params = _indicator_param_distance(inds1, inds2)

    # 3. Condition signature multiset (entry + exit unioned)
    cs1 = [_condition_signature(c) for c in (getattr(g1, "entry_conditions", []) or [])]
    cs1 += [_condition_signature(c) for c in (getattr(g1, "exit_conditions", []) or [])]
    cs2 = [_condition_signature(c) for c in (getattr(g2, "entry_conditions", []) or [])]
    cs2 += [_condition_signature(c) for c in (getattr(g2, "exit_conditions", []) or [])]
    d_cond_sig = _multiset_jaccard(cs1, cs2)

    # 4. Risk parameter continuous distance
    d_risk = _risk_parameter_distance(g1, g2)

    # 5. Flag distance
    d_flags = _flag_distance(g1, g2)

    total = (
        W_INDICATOR_SIG * d_ind_sig
        + W_INDICATOR_PARAMS * d_ind_params
        + W_CONDITION_SIG * d_cond_sig
        + W_RISK_PARAMS * d_risk
        + W_FLAGS * d_flags
    )
    # Numeric safety
    if math.isnan(total) or math.isinf(total):
        return 1.0
    return max(0.0, min(1.0, total))


def distance_breakdown(ind1, ind2) -> Dict[str, float]:
    """Return the per-component contributions for diagnostics/tests."""
    g1 = getattr(ind1, "strategy_gene", ind1)
    g2 = getattr(ind2, "strategy_gene", ind2)
    inds1 = list(getattr(g1, "indicators", []) or [])
    inds2 = list(getattr(g2, "indicators", []) or [])
    sig1 = [_indicator_signature(i) for i in inds1]
    sig2 = [_indicator_signature(i) for i in inds2]
    cs1 = [_condition_signature(c) for c in (getattr(g1, "entry_conditions", []) or [])]
    cs1 += [_condition_signature(c) for c in (getattr(g1, "exit_conditions", []) or [])]
    cs2 = [_condition_signature(c) for c in (getattr(g2, "entry_conditions", []) or [])]
    cs2 += [_condition_signature(c) for c in (getattr(g2, "exit_conditions", []) or [])]
    return {
        "indicator_signature": _multiset_jaccard(sig1, sig2),
        "indicator_params": _indicator_param_distance(inds1, inds2),
        "condition_signature": _multiset_jaccard(cs1, cs2),
        "risk_params": _risk_parameter_distance(g1, g2),
        "flags": _flag_distance(g1, g2),
        "total": calculate_genome_distance(ind1, ind2),
    }
