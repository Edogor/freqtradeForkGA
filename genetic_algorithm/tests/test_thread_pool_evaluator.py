"""T3.9 tests for the crash-safe ThreadPoolEvaluator + SequentialEvaluator."""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest


# ── Fixtures ───────────────────────────────────────────────────────────


class FakeGene:
    def __init__(self, idx: int):
        self.idx = idx


class FakeIndividual:
    """Mirrors the bits of ``Individual`` used by the evaluators."""

    def __init__(self, idx: int):
        self.strategy_gene = FakeGene(idx)
        self.fitness = None
        self.metrics = None
        self.objectives = None

    def set_fitness(self, fitness, metrics):
        self.fitness = fitness
        self.metrics = metrics

    def set_objectives(self, objectives, metrics):
        self.objectives = objectives
        self.metrics = metrics


def _make_eval_side_effect(sleep=0.0, fail_indices=None, raise_indices=None):
    fail_indices = set(fail_indices or [])
    raise_indices = set(raise_indices or [])

    def _ev(gene):
        if sleep:
            time.sleep(sleep)
        if gene.idx in raise_indices:
            raise RuntimeError(f"boom-{gene.idx}")
        if gene.idx in fail_indices:
            return 0.0, {"trades": 0}
        return 1.0 + gene.idx * 0.1, {"profit": 5.0 + gene.idx, "trades": 10}

    return _ev


@pytest.fixture
def base_config():
    return {
        "parallel_evaluation": {
            "enabled": True,
            "backend": "thread",
            "num_workers": 2,
            "backtest_timeout": 5,
        },
    }


# ── SequentialEvaluator ────────────────────────────────────────────────


def _patched(config, side_effect):
    """Helper: build evaluator with FitnessEvaluator patched."""
    from genetic_algorithm.evaluation import thread_pool_evaluator as tpe

    fake = MagicMock()
    fake.evaluate = MagicMock(side_effect=side_effect)
    with patch.object(tpe, "select_evaluator_backend"):  # no-op, just import-side
        pass
    return fake


def _build(klass, config, side_effect):
    fake = MagicMock()
    fake.evaluate = MagicMock(side_effect=side_effect)
    with patch(
        "genetic_algorithm.evaluation.fitness.FitnessEvaluator",
        return_value=fake,
    ):
        return klass(config), fake


class TestSequentialEvaluator:
    def test_empty_batch(self, base_config):
        from genetic_algorithm.evaluation.thread_pool_evaluator import SequentialEvaluator

        ev, _ = _build(SequentialEvaluator, base_config, _make_eval_side_effect())
        result = ev.evaluate_batch([])
        assert result.successful == 0 and result.failed == 0

    def test_all_succeed(self, base_config):
        from genetic_algorithm.evaluation.thread_pool_evaluator import SequentialEvaluator

        ev, _ = _build(SequentialEvaluator, base_config, _make_eval_side_effect())
        inds = [FakeIndividual(i) for i in range(4)]
        result = ev.evaluate_batch(inds)
        assert result.successful == 4
        assert result.failed == 0
        for ind in inds:
            assert ind.fitness is not None and ind.fitness > 0

    def test_exception_in_one_does_not_break_others(self, base_config):
        from genetic_algorithm.evaluation.thread_pool_evaluator import SequentialEvaluator

        ev, _ = _build(
            SequentialEvaluator,
            base_config,
            _make_eval_side_effect(raise_indices={2}),
        )
        inds = [FakeIndividual(i) for i in range(4)]
        result = ev.evaluate_batch(inds)
        assert result.successful == 3
        assert result.failed == 1
        assert inds[2].fitness == 0.0
        assert "error" in inds[2].metrics

    def test_progress_callback_invoked(self, base_config):
        from genetic_algorithm.evaluation.thread_pool_evaluator import SequentialEvaluator

        ev, _ = _build(SequentialEvaluator, base_config, _make_eval_side_effect())
        inds = [FakeIndividual(i) for i in range(3)]
        calls = []
        ev.evaluate_batch(inds, progress_callback=lambda c, t: calls.append((c, t)))
        assert calls == [(1, 3), (2, 3), (3, 3)]

    def test_shutdown_is_idempotent(self, base_config):
        from genetic_algorithm.evaluation.thread_pool_evaluator import SequentialEvaluator

        ev, _ = _build(SequentialEvaluator, base_config, _make_eval_side_effect())
        ev.shutdown()
        ev.shutdown()  # second call must not raise


# ── ThreadPoolEvaluator ────────────────────────────────────────────────


class TestThreadPoolEvaluator:
    def test_empty_batch(self, base_config):
        from genetic_algorithm.evaluation.thread_pool_evaluator import ThreadPoolEvaluator

        ev, _ = _build(ThreadPoolEvaluator, base_config, _make_eval_side_effect())
        result = ev.evaluate_batch([])
        assert result.successful == 0 and result.failed == 0
        ev.shutdown()

    def test_all_succeed_and_fitness_set(self, base_config):
        from genetic_algorithm.evaluation.thread_pool_evaluator import ThreadPoolEvaluator

        ev, fake = _build(ThreadPoolEvaluator, base_config, _make_eval_side_effect())
        inds = [FakeIndividual(i) for i in range(6)]
        result = ev.evaluate_batch(inds)
        assert result.successful == 6
        assert result.failed == 0
        assert fake.evaluate.call_count == 6
        for ind in inds:
            assert ind.fitness is not None and ind.fitness > 0
        ev.shutdown()

    def test_exception_isolated(self, base_config):
        from genetic_algorithm.evaluation.thread_pool_evaluator import ThreadPoolEvaluator

        ev, _ = _build(
            ThreadPoolEvaluator,
            base_config,
            _make_eval_side_effect(raise_indices={1, 3}),
        )
        inds = [FakeIndividual(i) for i in range(5)]
        result = ev.evaluate_batch(inds)
        assert result.successful == 3
        assert result.failed == 2
        for i in (1, 3):
            assert inds[i].fitness == 0.0
            assert "error" in (inds[i].metrics or {})
        ev.shutdown()

    def test_timeout_marks_failed_without_crashing_others(self, base_config):
        from genetic_algorithm.evaluation.thread_pool_evaluator import ThreadPoolEvaluator

        cfg = dict(base_config)
        cfg["parallel_evaluation"] = {
            **base_config["parallel_evaluation"],
            "backtest_timeout": 0.2,
        }

        def side_effect(gene):
            if gene.idx == 1:
                time.sleep(2.0)
            return 1.0, {"profit": 1.0}

        ev, _ = _build(ThreadPoolEvaluator, cfg, side_effect)
        inds = [FakeIndividual(i) for i in range(3)]
        result = ev.evaluate_batch(inds)
        # 2 succeed, 1 times out
        assert result.successful == 2
        assert result.failed == 1
        assert inds[1].fitness == 0.0
        assert "timeout" in (inds[1].metrics or {}).get("error", "")
        ev.shutdown()

    def test_progress_callback_invoked(self, base_config):
        from genetic_algorithm.evaluation.thread_pool_evaluator import ThreadPoolEvaluator

        ev, _ = _build(ThreadPoolEvaluator, base_config, _make_eval_side_effect())
        inds = [FakeIndividual(i) for i in range(4)]
        calls = []
        ev.evaluate_batch(inds, progress_callback=lambda c, t: calls.append((c, t)))
        assert len(calls) == 4
        assert all(t == 4 for c, t in calls)
        assert {c for c, _ in calls} == {1, 2, 3, 4}
        ev.shutdown()

    def test_shutdown_idempotent(self, base_config):
        from genetic_algorithm.evaluation.thread_pool_evaluator import ThreadPoolEvaluator

        ev, _ = _build(ThreadPoolEvaluator, base_config, _make_eval_side_effect())
        ev.evaluate_batch([FakeIndividual(0)])
        ev.shutdown()
        ev.shutdown()  # no raise

    def test_pool_survives_across_batches(self, base_config):
        from genetic_algorithm.evaluation.thread_pool_evaluator import ThreadPoolEvaluator

        ev, _ = _build(
            ThreadPoolEvaluator,
            base_config,
            _make_eval_side_effect(raise_indices={0}),
        )
        # Batch 1: one crash
        r1 = ev.evaluate_batch([FakeIndividual(0), FakeIndividual(1)])
        assert r1.failed == 1 and r1.successful == 1
        # Batch 2: must still work
        r2 = ev.evaluate_batch([FakeIndividual(2), FakeIndividual(3)])
        assert r2.successful == 2 and r2.failed == 0
        ev.shutdown()

    def test_nsga2_objectives_set(self, base_config):
        from genetic_algorithm.evaluation.thread_pool_evaluator import ThreadPoolEvaluator

        cfg = dict(base_config)
        cfg["genetic_algorithm"] = {"mode": "nsga2"}
        cfg["nsga2"] = {
            "objectives": [{"name": "profit", "direction": "maximize"}],
            "min_trades": 0,
        }

        with patch(
            "genetic_algorithm.core.nsga2.extract_objectives_from_metrics",
            return_value=[7.0],
        ):
            ev, _ = _build(ThreadPoolEvaluator, cfg, _make_eval_side_effect())
            inds = [FakeIndividual(i) for i in range(2)]
            result = ev.evaluate_batch(inds)
            assert result.successful == 2
            for ind in inds:
                assert ind.objectives == [7.0]
        ev.shutdown()


# ── Backend factory ────────────────────────────────────────────────────


class TestBackendFactory:
    def test_disabled_returns_none(self):
        from genetic_algorithm.evaluation.thread_pool_evaluator import (
            select_evaluator_backend,
        )

        assert select_evaluator_backend({"parallel_evaluation": {"enabled": False}}) is None
        assert select_evaluator_backend({}) is None

    def test_thread_backend(self):
        from genetic_algorithm.evaluation.thread_pool_evaluator import (
            ThreadPoolEvaluator,
            select_evaluator_backend,
        )

        with patch("genetic_algorithm.evaluation.fitness.FitnessEvaluator", MagicMock()):
            ev = select_evaluator_backend(
                {"parallel_evaluation": {"enabled": True, "backend": "thread", "num_workers": 2}}
            )
        assert isinstance(ev, ThreadPoolEvaluator)
        ev.shutdown()

    def test_sequential_backend(self):
        from genetic_algorithm.evaluation.thread_pool_evaluator import (
            SequentialEvaluator,
            select_evaluator_backend,
        )

        with patch("genetic_algorithm.evaluation.fitness.FitnessEvaluator", MagicMock()):
            ev = select_evaluator_backend(
                {"parallel_evaluation": {"enabled": True, "backend": "sequential"}}
            )
        assert isinstance(ev, SequentialEvaluator)

    def test_process_backend_default(self):
        from genetic_algorithm.evaluation.parallel import ParallelEvaluator
        from genetic_algorithm.evaluation.thread_pool_evaluator import (
            select_evaluator_backend,
        )

        with patch.object(ParallelEvaluator, "__init__", return_value=None) as mock_init:
            ev = select_evaluator_backend(
                {"parallel_evaluation": {"enabled": True}}  # no backend → process default
            )
        assert isinstance(ev, ParallelEvaluator)
        mock_init.assert_called_once()
