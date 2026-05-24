"""T3.6 — Memetic Local Search.

A lightweight, pure-Python local refiner that takes the top-K individuals
after a generation and tries small perturbations on their *numeric*
genome parameters (stoploss, ROI rungs, condition thresholds, trailing
offsets) keeping any change that improves the fitness reported by a
caller-supplied evaluator.

The refiner uses **1+1 Evolution Strategy** with the 1/5 success rule
for step-size adaptation, rather than full Nelder-Mead:
  * No scipy dependency
  * No simplex bookkeeping (we'd need n+1 evaluations per step for an
    n-dim simplex, which is too expensive for a generation-end refiner)
  * Naturally bounded: clamps each numeric parameter to a sensible
    domain after perturbation
  * Anytime: caller can stop after any number of iterations

API
---
``LocalSearchRefiner(evaluate_fn, max_iters=12, step=0.15)``
    ``evaluate_fn(gene) -> float`` is the caller's plug-in.  It runs
    a backtest (or any proxy) on the candidate gene and returns a
    scalar fitness — larger is better.

``refiner.refine(gene) -> RefineResult``
    Mutates and returns a *copy* of the gene with possibly higher
    fitness, plus stats on how many evaluations were spent.

The module is **engine-agnostic**: it doesn't import the
FitnessEvaluator or backtester; tests use a synthetic quadratic
landscape.
"""
from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


# ── Parameter introspection ────────────────────────────────────────────


# Allowed numeric ranges for each parameter we will refine.  Values
# outside these get clamped after perturbation.
DEFAULT_BOUNDS: Dict[str, Tuple[float, float]] = {
    "stoploss": (-0.40, -0.01),
    "trailing_stop_positive": (0.0, 0.30),
    "trailing_stop_positive_offset": (0.0, 0.30),
    "roi_value": (0.0, 0.50),
    "cond_threshold": (-200.0, 200.0),
}


@dataclass
class _ParamRef:
    """A handle pointing at one numeric scalar inside a gene."""

    name: str
    getter: Callable[[Any], float]
    setter: Callable[[Any, float], None]
    lo: float
    hi: float

    def clamp(self, v: float) -> float:
        return max(self.lo, min(self.hi, v))


def _collect_numeric_params(gene) -> List[_ParamRef]:
    """Walk a StrategyGene and return setters/getters for each
    refinable numeric scalar.  Tolerant of legacy / fake genes used
    by unit tests — only walks attributes that actually exist."""
    refs: List[_ParamRef] = []

    if hasattr(gene, "stoploss"):
        lo, hi = DEFAULT_BOUNDS["stoploss"]

        def get_sl(g, _=None):
            return float(g.stoploss)

        def set_sl(g, v):
            g.stoploss = v

        refs.append(_ParamRef("stoploss", get_sl, set_sl, lo, hi))

    for attr in ("trailing_stop_positive", "trailing_stop_positive_offset"):
        if hasattr(gene, attr):
            lo, hi = DEFAULT_BOUNDS["trailing_stop_positive"]

            def make_get(a):
                return lambda g: float(getattr(g, a, 0.0) or 0.0)

            def make_set(a):
                return lambda g, v: setattr(g, a, v)

            refs.append(_ParamRef(attr, make_get(attr), make_set(attr), lo, hi))

    # ROI rungs
    roi = getattr(gene, "minimal_roi", None) or {}
    lo, hi = DEFAULT_BOUNDS["roi_value"]
    for key in list(roi.keys()):

        def make_get(k):
            return lambda g: float((getattr(g, "minimal_roi", {}) or {}).get(k, 0.0) or 0.0)

        def make_set(k):
            def _set(g, v):
                m = getattr(g, "minimal_roi", None)
                if m is None:
                    g.minimal_roi = {}
                    m = g.minimal_roi
                m[k] = v
            return _set

        refs.append(_ParamRef(f"roi[{key}]", make_get(key), make_set(key), lo, hi))

    # Condition thresholds
    for collection_name in ("entry_conditions", "exit_conditions"):
        conds = getattr(gene, collection_name, []) or []
        for i, _ in enumerate(conds):

            def make_get(cn, idx):
                return lambda g: float(getattr(getattr(g, cn)[idx], "threshold", 0.0) or 0.0)

            def make_set(cn, idx):
                def _set(g, v):
                    setattr(getattr(g, cn)[idx], "threshold", v)
                return _set

            lo, hi = DEFAULT_BOUNDS["cond_threshold"]
            refs.append(
                _ParamRef(
                    f"{collection_name}[{i}].threshold",
                    make_get(collection_name, i),
                    make_set(collection_name, i),
                    lo,
                    hi,
                )
            )

    return refs


# ── Refiner ────────────────────────────────────────────────────────────


@dataclass
class RefineResult:
    """Outcome of a single ``refine`` call."""

    refined_gene: Any
    initial_fitness: float
    final_fitness: float
    iterations: int
    evaluations: int
    accepted: int
    final_step: float
    improved: bool = field(init=False)

    def __post_init__(self):
        self.improved = self.final_fitness > self.initial_fitness


class LocalSearchRefiner:
    """1+1 ES local refiner with 1/5 success-rule step adaptation."""

    def __init__(
        self,
        evaluate_fn: Callable[[Any], float],
        max_iters: int = 12,
        initial_step: float = 0.15,
        step_min: float = 0.005,
        step_max: float = 0.5,
        rng: Optional[random.Random] = None,
    ):
        self.evaluate_fn = evaluate_fn
        self.max_iters = max(1, int(max_iters))
        self.initial_step = float(initial_step)
        self.step_min = float(step_min)
        self.step_max = float(step_max)
        self.rng = rng or random.Random()

    def _perturb(self, value: float, ref: _ParamRef, step: float) -> float:
        # Gaussian perturbation scaled by the parameter's range.
        rng_span = max(abs(ref.hi - ref.lo), 1e-6)
        noise = self.rng.gauss(0.0, step) * rng_span * 0.5
        return ref.clamp(value + noise)

    def refine(
        self,
        gene,
        initial_fitness: Optional[float] = None,
    ) -> RefineResult:
        refs = _collect_numeric_params(gene)
        if not refs:
            return RefineResult(
                refined_gene=gene,
                initial_fitness=initial_fitness or 0.0,
                final_fitness=initial_fitness or 0.0,
                iterations=0,
                evaluations=0,
                accepted=0,
                final_step=self.initial_step,
            )

        current = copy.deepcopy(gene)
        if initial_fitness is not None:
            current_fit = float(initial_fitness)
            start_fit = float(initial_fitness)
            evaluations = 0
        else:
            current_fit = float(self.evaluate_fn(current))
            start_fit = current_fit
            evaluations = 1

        step = self.initial_step
        accepted = 0
        iters = 0

        # 1/5 success rule tracking
        success_window = 5
        recent_successes = 0
        recent_attempts = 0

        for iters in range(1, self.max_iters + 1):
            candidate = copy.deepcopy(current)
            # Perturb each numeric scalar.
            for ref in refs:
                v = ref.getter(candidate)
                ref.setter(candidate, self._perturb(v, ref, step))

            # Maintain ROI monotonicity if a normalisation hook exists.
            normaliser = getattr(candidate, "_normalize_roi", None)
            if callable(normaliser):
                try:
                    normaliser()
                except Exception:
                    pass

            cand_fit = self.evaluate_fn(candidate)
            evaluations += 1
            recent_attempts += 1

            if cand_fit > current_fit:
                current = candidate
                current_fit = cand_fit
                accepted += 1
                recent_successes += 1

            # 1/5 rule: every window iterations, adapt step.
            if recent_attempts >= success_window:
                rate = recent_successes / recent_attempts
                if rate > 0.2:
                    step = min(self.step_max, step * 1.2)
                elif rate < 0.2:
                    step = max(self.step_min, step * 0.85)
                recent_successes = 0
                recent_attempts = 0

        return RefineResult(
            refined_gene=current,
            initial_fitness=start_fit,
            final_fitness=current_fit,
            iterations=iters,
            evaluations=evaluations,
            accepted=accepted,
            final_step=step,
        )


# ── Batch helper for generation-end use ───────────────────────────────


def memetic_refine_topk(
    individuals: List[Any],
    evaluate_fn: Callable[[Any], Tuple[float, Dict[str, Any]]],
    k: int = 5,
    max_iters: int = 12,
    initial_step: float = 0.15,
    rng: Optional[random.Random] = None,
) -> Dict[str, int]:
    """Refine the K individuals with highest fitness in-place.

    ``evaluate_fn`` is the FULL evaluator returning ``(fitness, metrics)``
    (i.e. the same shape the GA already uses).  The refiner wraps it
    to extract just the fitness scalar.

    The function only *upgrades* an individual when the refined gene
    has a strictly higher fitness — otherwise the individual is left
    untouched.  Returns a small stats dict.
    """
    if not individuals or k <= 0:
        return {"refined": 0, "improved": 0, "evaluations": 0}

    # Sort by current fitness desc; only refine those with a fitness.
    rated = [
        ind for ind in individuals
        if getattr(ind, "fitness", None) is not None
        and getattr(ind, "strategy_gene", None) is not None
    ]
    rated.sort(key=lambda i: i.fitness, reverse=True)
    targets = rated[: min(k, len(rated))]

    def scalar_eval(gene) -> float:
        fitness, _ = evaluate_fn(gene)
        return float(fitness) if not math.isnan(fitness) else -math.inf

    refiner = LocalSearchRefiner(
        scalar_eval,
        max_iters=max_iters,
        initial_step=initial_step,
        rng=rng,
    )

    refined = improved = evaluations = 0
    for ind in targets:
        res = refiner.refine(ind.strategy_gene, initial_fitness=ind.fitness)
        refined += 1
        evaluations += res.evaluations
        if res.improved:
            # Re-run the full evaluator on the refined gene to obtain
            # the full metrics dict (refiner only used the scalar).
            new_fit, new_metrics = evaluate_fn(res.refined_gene)
            if new_fit > ind.fitness:
                ind.strategy_gene = res.refined_gene
                if hasattr(ind, "set_fitness"):
                    ind.set_fitness(new_fit, new_metrics)
                else:
                    ind.fitness = new_fit
                    ind.metrics = new_metrics
                improved += 1
    return {"refined": refined, "improved": improved, "evaluations": evaluations}
