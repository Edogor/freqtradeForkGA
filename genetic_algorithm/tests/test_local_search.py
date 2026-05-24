"""T3.6 — Tests for memetic local search."""
from __future__ import annotations

import math
import random

import pytest

from genetic_algorithm.core.local_search import (
    LocalSearchRefiner,
    RefineResult,
    memetic_refine_topk,
    _collect_numeric_params,
)


# ── Fakes ─────────────────────────────────────────────────────────────


class _Cond:
    def __init__(self, threshold=0.0):
        self.indicator = "RSI"
        self.operator = "<"
        self.threshold = threshold


class _Gene:
    def __init__(
        self,
        stoploss=-0.10,
        trailing_stop_positive=0.0,
        trailing_stop_positive_offset=0.0,
        minimal_roi=None,
        entry_conditions=None,
        exit_conditions=None,
    ):
        self.stoploss = stoploss
        self.trailing_stop_positive = trailing_stop_positive
        self.trailing_stop_positive_offset = trailing_stop_positive_offset
        self.minimal_roi = dict(minimal_roi or {"0": 0.04, "30": 0.02, "60": 0.01})
        self.entry_conditions = entry_conditions or []
        self.exit_conditions = exit_conditions or []


class _Individual:
    def __init__(self, gene, fitness=None):
        self.strategy_gene = gene
        self.fitness = fitness
        self.metrics = {}

    def set_fitness(self, f, m):
        self.fitness = f
        self.metrics = m


# ── Param introspection ───────────────────────────────────────────────


class TestParamCollection:
    def test_collects_stoploss(self):
        refs = _collect_numeric_params(_Gene())
        names = [r.name for r in refs]
        assert "stoploss" in names
        assert "trailing_stop_positive" in names
        assert any(n.startswith("roi[") for n in names)

    def test_handles_empty_conditions(self):
        refs = _collect_numeric_params(_Gene(entry_conditions=[], exit_conditions=[]))
        assert all(not r.name.startswith("entry_conditions") for r in refs)

    def test_collects_condition_thresholds(self):
        g = _Gene(entry_conditions=[_Cond(30.0), _Cond(50.0)])
        refs = _collect_numeric_params(g)
        threshold_refs = [r for r in refs if "threshold" in r.name]
        assert len(threshold_refs) == 2

    def test_clamping(self):
        refs = _collect_numeric_params(_Gene())
        sl_ref = next(r for r in refs if r.name == "stoploss")
        assert sl_ref.clamp(-5.0) == sl_ref.lo
        assert sl_ref.clamp(5.0) == sl_ref.hi
        assert sl_ref.clamp(-0.10) == -0.10


# ── Convergence on a synthetic landscape ──────────────────────────────


class TestRefinerConvergence:
    def test_finds_better_stoploss_when_optimum_is_visible(self):
        # Synthetic: fitness peaks at stoploss = -0.08.
        target = -0.08

        def eval_fn(gene):
            # Pure function of stoploss for testability
            return -((gene.stoploss - target) ** 2) * 100.0

        g = _Gene(stoploss=-0.20)
        rng = random.Random(42)
        refiner = LocalSearchRefiner(eval_fn, max_iters=40, initial_step=0.20, rng=rng)
        result = refiner.refine(g)
        assert result.improved
        assert result.final_fitness > result.initial_fitness
        # Should move stoploss closer to target.
        assert abs(result.refined_gene.stoploss - target) < abs(g.stoploss - target)

    def test_no_numeric_params_returns_immediately(self):
        class Empty:
            pass

        called = []

        def eval_fn(gene):
            called.append(1)
            return 1.0

        refiner = LocalSearchRefiner(eval_fn, max_iters=5)
        result = refiner.refine(Empty(), initial_fitness=0.5)
        assert result.iterations == 0
        assert result.evaluations == 0
        assert called == []

    def test_never_returns_worse_fitness(self):
        # Random evaluator: refiner should keep current_fit monotone non-decreasing.
        rng = random.Random(0)

        def eval_fn(gene):
            return rng.random() - 0.5

        g = _Gene()
        refiner = LocalSearchRefiner(eval_fn, max_iters=20, rng=random.Random(7))
        # Set an initial_fitness so we have a starting baseline.
        result = refiner.refine(g, initial_fitness=-2.0)
        assert result.final_fitness >= result.initial_fitness

    def test_step_size_adapts(self):
        # If we never improve, step should shrink toward step_min.
        def eval_fn(gene):
            return -1.0  # always worse than initial

        refiner = LocalSearchRefiner(
            eval_fn, max_iters=30, initial_step=0.3, step_min=0.01
        )
        result = refiner.refine(_Gene(), initial_fitness=0.0)
        assert result.final_step <= refiner.initial_step

    def test_evaluations_counted(self):
        def eval_fn(gene):
            return 0.0

        refiner = LocalSearchRefiner(eval_fn, max_iters=7)
        result = refiner.refine(_Gene(), initial_fitness=0.0)
        # initial_fitness supplied → no startup eval; 7 iterations.
        assert result.evaluations == 7
        assert result.iterations == 7


# ── Bounds respected ──────────────────────────────────────────────────


class TestBounds:
    def test_perturbation_stays_within_bounds(self):
        def eval_fn(gene):
            # Reward extreme stoploss to push the perturbation hard.
            return -gene.stoploss

        g = _Gene(stoploss=-0.10)
        rng = random.Random(123)
        refiner = LocalSearchRefiner(eval_fn, max_iters=50, initial_step=0.5, rng=rng)
        result = refiner.refine(g)
        assert -0.40 <= result.refined_gene.stoploss <= -0.01


# ── memetic_refine_topk ────────────────────────────────────────────────


class TestMemeticRefineTopK:
    def test_empty_population_is_no_op(self):
        stats = memetic_refine_topk([], evaluate_fn=lambda g: (0.0, {}), k=3)
        assert stats == {"refined": 0, "improved": 0, "evaluations": 0}

    def test_only_top_k_refined(self):
        target = -0.08
        call_log = []

        def eval_fn(gene):
            call_log.append(gene)
            f = -((gene.stoploss - target) ** 2) * 100.0
            return f, {"profit": 1.0}

        # Make 5 individuals with descending fitness.
        inds = []
        for i, f in enumerate([0.9, 0.8, 0.7, 0.6, 0.5]):
            inds.append(_Individual(_Gene(stoploss=-0.20 - 0.01 * i), fitness=f))

        stats = memetic_refine_topk(
            inds,
            eval_fn,
            k=2,
            max_iters=15,
            initial_step=0.2,
            rng=random.Random(13),
        )
        assert stats["refined"] == 2
        assert stats["evaluations"] > 0
        # The 3rd-5th individuals must keep their original fitness.
        assert inds[2].fitness == 0.7
        assert inds[3].fitness == 0.6
        assert inds[4].fitness == 0.5

    def test_improvement_upgrades_individual(self):
        target = -0.08

        def eval_fn(gene):
            return -((gene.stoploss - target) ** 2) * 100.0, {"profit": 5.0}

        ind = _Individual(_Gene(stoploss=-0.25), fitness=-2.89)
        stats = memetic_refine_topk(
            [ind],
            eval_fn,
            k=1,
            max_iters=40,
            initial_step=0.25,
            rng=random.Random(99),
        )
        # Improved at least once
        assert stats["improved"] >= 1
        assert ind.fitness > -2.89
        # Metrics dict was refreshed
        assert ind.metrics.get("profit") == 5.0

    def test_unrated_individuals_skipped(self):
        def eval_fn(gene):
            return 1.0, {}

        # No fitness set
        unrated = _Individual(_Gene(), fitness=None)
        stats = memetic_refine_topk([unrated], eval_fn, k=3)
        assert stats["refined"] == 0
