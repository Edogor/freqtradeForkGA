"""T4.3 — Anti-pattern penalty hook.

Counterpart to the synergy graph: a small registry of indicator-pair
combinations that empirically under-perform.  Two consumption modes:

1. **Soft fitness penalty** (recommended): post-evaluation, multiply
   raw fitness by ``anti_pattern_multiplier(indicators)``.  Default
   per-edge penalty is 0.95 so the prior nudges ranking without
   killing exploration.

2. **Veto check** for mutation/crossover that wants to *introduce*
   an anti-pattern pair.  ``would_introduce_anti_pattern`` returns
   the offending pair (or None) so callers can re-roll.

Both helpers accept the bare anti-pattern dict (as produced by
``prior_rebuilder.rebuild_from_rows``) so the module has no
dependency on SISIntegrator and can be unit-tested in isolation.

The penalty multiplier shape::

    mult = max(per_edge_floor, 1 - sum(weight_of_each_anti_edge) * scale)

is bounded in ``[per_edge_floor, 1.0]`` so a strategy stuffed with
anti-pairs cannot have its fitness reduced below the floor (default
0.6).  This keeps the ranking signal monotonic but stops a single
penalty layer from wiping out otherwise interesting candidates.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple


AntiPatternGraph = Dict[str, Dict[str, float]]


def _normalise_indicators(indicators: Iterable[str]) -> List[str]:
    seen: List[str] = []
    for name in indicators:
        if not isinstance(name, str):
            continue
        upper = name.strip().upper()
        if not upper:
            continue
        if upper in seen:
            continue
        seen.append(upper)
    return seen


def matched_anti_pattern_pairs(
    indicators: Iterable[str],
    anti_patterns: Optional[AntiPatternGraph],
) -> List[Tuple[str, str, float]]:
    """Return the anti-pattern pairs present in ``indicators``.

    Each tuple is ``(indicator_a, indicator_b, penalty_weight)`` with
    ``a < b`` so each pair appears at most once.  Empty when the graph
    is missing or no pairs match.
    """
    if not anti_patterns:
        return []
    inds = _normalise_indicators(indicators)
    if len(inds) < 2:
        return []
    matches: List[Tuple[str, str, float]] = []
    seen_pairs: set = set()
    for i, a in enumerate(inds):
        partners = anti_patterns.get(a)
        if not partners:
            continue
        for b in inds[i + 1 :]:
            if b not in partners:
                continue
            key = (a, b) if a < b else (b, a)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            weight = float(partners.get(b, 0.0))
            matches.append((key[0], key[1], weight))
    return matches


def anti_pattern_multiplier(
    indicators: Iterable[str],
    anti_patterns: Optional[AntiPatternGraph],
    *,
    scale: float = 0.5,
    floor: float = 0.6,
) -> float:
    """Soft fitness multiplier in ``[floor, 1.0]``.

    Each matched anti-pattern pair contributes ``weight * scale`` to a
    cumulative penalty.  Tuned so a typical penalty weight of 0.5
    yields multiplier 0.75 for one edge, 0.5 with two edges, then
    clipped at the floor.  A strategy with no anti-pairs (or no
    anti-pattern graph) gets multiplier 1.0 — zero-cost path.
    """
    matches = matched_anti_pattern_pairs(indicators, anti_patterns)
    if not matches:
        return 1.0
    total = sum(weight for _, _, weight in matches)
    return max(floor, 1.0 - total * scale)


def would_introduce_anti_pattern(
    current_indicators: Iterable[str],
    new_indicator: str,
    anti_patterns: Optional[AntiPatternGraph],
) -> Optional[Tuple[str, float]]:
    """Return ``(partner, weight)`` if adding ``new_indicator`` would
    create an anti-pattern with any current indicator.  Highest-weight
    conflict wins so callers see the most painful pair first.
    """
    if not anti_patterns:
        return None
    target = new_indicator.strip().upper() if isinstance(new_indicator, str) else ""
    if not target:
        return None
    partners = anti_patterns.get(target)
    if not partners:
        return None
    current = set(_normalise_indicators(current_indicators))
    if not current:
        return None
    conflict: Optional[Tuple[str, float]] = None
    for existing in current:
        if existing in partners:
            weight = float(partners[existing])
            if conflict is None or weight > conflict[1]:
                conflict = (existing, weight)
    return conflict


def summarise_anti_patterns(
    indicators: Iterable[str],
    anti_patterns: Optional[AntiPatternGraph],
    *,
    scale: float = 0.5,
    floor: float = 0.6,
) -> Dict[str, object]:
    """Return a diagnostics-friendly record describing the penalty.

    Useful for logging / failure-mode reports (T4.5) so we can audit
    *why* a strategy's fitness was discounted.
    """
    pairs = matched_anti_pattern_pairs(indicators, anti_patterns)
    multiplier = anti_pattern_multiplier(
        indicators, anti_patterns, scale=scale, floor=floor
    )
    return {
        "multiplier": multiplier,
        "pairs": [
            {"a": a, "b": b, "weight": w} for a, b, w in pairs
        ],
        "n_pairs": len(pairs),
    }


__all__ = [
    "AntiPatternGraph",
    "matched_anti_pattern_pairs",
    "anti_pattern_multiplier",
    "would_introduce_anti_pattern",
    "summarise_anti_patterns",
]
