"""Tests for RunEngine orchestrator — engine/runner.py.

These tests verify the orchestration logic (signals, timing, web events,
checkpoint scheduling, resource checks, teardown) WITHOUT exercising the
full GeneticAlgorithm.  We build a lightweight mock GA that exposes the
attributes RunEngine reads and the methods it calls.
"""

import signal
import threading
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from genetic_algorithm.engine.runner import GenerationResult, RunEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class _FakePopulation:
    individuals: list = field(default_factory=list)

    def get_best(self, n):
        return self.individuals[:n]

    def sort_by_fitness(self, reverse=False):
        self.individuals.sort(
            key=lambda x: getattr(x, "fitness", 0) or 0, reverse=reverse,
        )

    def __iter__(self):
        return iter(self.individuals)


@dataclass
class _FakeIndividual:
    id: str = "ind-1"
    fitness: float = 0.5
    strategy_gene: object = None
    objectives: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    def to_dict(self):
        return {"id": self.id, "fitness": self.fitness}


class _FakeTiming:
    history = []
    _phases = {}


class _FakeDiagnostics:
    timing = _FakeTiming()

    def start_run(self, config):
        pass

    def start_generation(self, gen):
        pass

    def end_generation(self, gen, stats, population, extras=None):
        pass

    def end_run(self, top_strategies=None):
        return None


class _FakeMonitor:
    def start(self, config):
        pass

    def on_generation_start(self, gen, total):
        pass

    def on_generation_end(self, gen, stats, timing, best, extras=None):
        pass

    def on_evolution_complete(self, summary):
        pass

    def on_phase_start(self, name):
        pass

    def on_phase_end(self, name, elapsed):
        pass

    def on_log(self, msg, level="info"):
        pass

    def on_new_best(self, ind):
        pass

    def on_convergence_warning(self, count, patience):
        pass

    def on_error(self, msg):
        pass


class _FakeTracker:
    def save_generation(self, gen, stats, pop):
        pass

    def save_final_results(self, best, stats):
        pass

    def record_new_best(self, gen, fitness, metrics):
        pass


def _build_ga(*, generations=3, mode="single", population=None, **overrides):
    """Build a minimal mock GA for RunEngine tests."""
    pop = population or _FakePopulation(
        individuals=[_FakeIndividual(id=f"ind-{i}") for i in range(5)]
    )

    ga = MagicMock()
    ga.logger = MagicMock()
    ga.generations = generations
    ga.mode = mode
    ga.population_size = len(pop.individuals)
    ga.max_runtime_minutes = overrides.get("max_runtime_minutes", None)
    ga.checkpoint_interval = overrides.get("checkpoint_interval", 0)
    ga.config = overrides.get("config", {})
    ga.current_generation = 0
    ga.best_individual = _FakeIndividual()
    ga.best_fitness_ever = 0.0
    ga.mutation_rate = 0.1
    ga.crossover_rate = 0.7
    ga.crossover_method = "uniform"
    ga.selection_method = "tournament"
    ga.elite_size = 1
    ga.no_improvement_count = 0
    ga.convergence_patience = 10
    ga._shutdown_requested = False
    ga._web_stop_event = None
    ga._web_pause_event = None
    ga._web_injection_queue = None
    ga._web_monitor = None
    ga._incremental = SimpleNamespace(enabled=False)
    ga._coevolution = SimpleNamespace(enabled=False)
    ga._lifecycle = SimpleNamespace(enabled=False)
    ga.visualizer = None
    ga.parallel_evaluator = None
    ga.trade_visualizer = None
    ga.trade_vis_mode = None
    ga.trade_vis_top_n = 3
    ga.llm_enabled = False
    ga.strategy_designer = SimpleNamespace(enabled=False)
    ga.pareto_front_size = 10
    ga.checkpoint_dir = __import__("pathlib").Path("/tmp/test_checkpoints")
    ga.diagnostics = _FakeDiagnostics()
    ga.monitor = _FakeMonitor()
    ga._tracker = _FakeTracker()
    ga.fitness_evaluator = SimpleNamespace()
    ga.hall_of_fame = MagicMock()
    ga.hall_of_fame.get_summary.return_value = {"size": 0}
    ga.feature_tracker = MagicMock()

    # initialize_population returns our fake pop
    ga.initialize_population.return_value = pop

    # process_generation returns a minimal result
    def _process_gen(population, gen, pareto_archive=None):
        return GenerationResult(
            population=population,
            stats={"best_fitness": 0.5, "avg_fitness": 0.3},
        )

    ga.process_generation.side_effect = _process_gen

    # advance_generation returns (pop, False) by default
    ga.advance_generation.return_value = (pop, False)

    # Apply overrides
    for k, v in overrides.items():
        setattr(ga, k, v)

    return ga


# =====================================================================
# GenerationResult dataclass
# =====================================================================


class TestGenerationResult:
    def test_defaults(self):
        r = GenerationResult(
            population=_FakePopulation(), stats={},
        )
        assert r.should_break is False
        assert r.break_reason == ""
        assert r.extras == {}

    def test_break(self):
        r = GenerationResult(
            population=_FakePopulation(),
            stats={},
            should_break=True,
            break_reason="holdout",
        )
        assert r.should_break
        assert r.break_reason == "holdout"


# =====================================================================
# RunEngine construction
# =====================================================================


class TestRunEngineInit:
    def test_stores_ga(self):
        ga = _build_ga()
        engine = RunEngine(ga)
        assert engine.ga is ga

    def test_uses_ga_logger(self):
        ga = _build_ga()
        engine = RunEngine(ga)
        assert engine.logger is ga.logger


# =====================================================================
# Setup
# =====================================================================


class TestSetup:
    def test_fresh_start(self):
        ga = _build_ga(generations=2)
        engine = RunEngine(ga)
        pop, start_gen, pareto_archive = engine._setup(resume_from=None)
        assert start_gen == 0
        assert pareto_archive is None
        ga.initialize_population.assert_called_once()

    def test_resume(self):
        pop = _FakePopulation(individuals=[_FakeIndividual()])
        ga = _build_ga(generations=5)
        ga.load_checkpoint.return_value = (pop, 3)
        engine = RunEngine(ga)
        result_pop, start_gen, _ = engine._setup(resume_from="/tmp/ckpt.json")
        assert start_gen == 3
        assert result_pop is pop
        ga.load_checkpoint.assert_called_once_with("/tmp/ckpt.json")

    def test_pareto_archive_created_for_nsga2(self):
        ga = _build_ga(mode="nsga2", config={"pareto_archive": {"enabled": True}})
        engine = RunEngine(ga)
        _, _, archive = engine._setup(None)
        assert archive is not None


# =====================================================================
# Time limit
# =====================================================================


class TestTimeLimit:
    def test_no_limit(self):
        ga = _build_ga(max_runtime_minutes=None)
        engine = RunEngine(ga)
        assert engine._check_time_limit(time.time(), 0, None) is False

    def test_within_limit(self):
        ga = _build_ga(max_runtime_minutes=60)
        engine = RunEngine(ga)
        assert engine._check_time_limit(time.time(), 0, None) is False

    def test_exceeded(self):
        ga = _build_ga(max_runtime_minutes=1)
        engine = RunEngine(ga)
        pop = _FakePopulation()
        start = time.time() - 120  # 2 minutes ago
        assert engine._check_time_limit(start, 5, pop) is True
        ga.save_checkpoint.assert_called_once()


# =====================================================================
# Web events
# =====================================================================


class TestWebEvents:
    def test_no_events(self):
        ga = _build_ga()
        engine = RunEngine(ga)
        assert engine._handle_web_events(0, _FakePopulation()) is False

    def test_stop_event(self):
        stop_event = threading.Event()
        stop_event.set()
        ga = _build_ga()
        ga._web_stop_event = stop_event
        engine = RunEngine(ga)
        assert engine._handle_web_events(0, _FakePopulation()) is True
        ga.save_checkpoint.assert_called_once()

    def test_injection_queue(self):
        ga = _build_ga()
        ga._web_injection_queue = [1, 2, 3]  # non-empty
        engine = RunEngine(ga)
        engine._handle_web_events(0, _FakePopulation())
        ga._drain_injection_queue.assert_called_once()


# =====================================================================
# Post-generation orchestration
# =====================================================================


class TestEndGeneration:
    def test_checkpoint_at_interval(self):
        ga = _build_ga(checkpoint_interval=2)
        engine = RunEngine(ga)
        pop = _FakePopulation()
        result = GenerationResult(population=pop, stats={})
        # gen 1 (0-indexed) → (1+1)%2 == 0 → should checkpoint
        engine._end_generation(1, result, pop)
        ga.save_checkpoint.assert_called_once_with(pop, 1)

    def test_no_checkpoint_off_interval(self):
        ga = _build_ga(checkpoint_interval=5)
        engine = RunEngine(ga)
        pop = _FakePopulation()
        result = GenerationResult(population=pop, stats={})
        engine._end_generation(2, result, pop)  # (2+1)%5 != 0
        ga.save_checkpoint.assert_not_called()


# =====================================================================
# Resource checks
# =====================================================================


class TestResourceChecks:
    def test_shutdown_requested(self):
        ga = _build_ga()
        ga._shutdown_requested = True
        engine = RunEngine(ga)
        assert engine._check_resources(5, _FakePopulation()) is True
        ga.save_checkpoint.assert_called_once()

    def test_no_shutdown(self):
        ga = _build_ga()
        engine = RunEngine(ga)
        assert engine._check_resources(0, _FakePopulation()) is False


# =====================================================================
# Full run loop
# =====================================================================


class TestRunLoop:
    def test_runs_all_generations(self):
        ga = _build_ga(generations=3)
        engine = RunEngine(ga)
        engine.run()
        assert ga.process_generation.call_count == 3
        assert ga.advance_generation.call_count == 3

    def test_process_generation_break_stops_loop(self):
        ga = _build_ga(generations=10)
        call_count = 0

        def _break_on_gen2(pop, gen, pareto=None):
            nonlocal call_count
            call_count += 1
            return GenerationResult(
                population=pop,
                stats={},
                should_break=(gen == 2),
                break_reason="holdout",
            )

        ga.process_generation.side_effect = _break_on_gen2
        engine = RunEngine(ga)
        engine.run()
        assert call_count == 3  # gen 0, 1, 2

    def test_advance_generation_break_stops_loop(self):
        pop = _FakePopulation(individuals=[_FakeIndividual()])
        ga = _build_ga(generations=10, population=pop)

        call_count = 0

        def _adv(population, gen, stats):
            nonlocal call_count
            call_count += 1
            return population, (gen == 1)

        ga.advance_generation.side_effect = _adv
        engine = RunEngine(ga)
        engine.run()
        assert call_count == 2  # gen 0, 1

    def test_returns_best_individuals(self):
        pop = _FakePopulation(
            individuals=[_FakeIndividual(id="best", fitness=1.0)]
        )
        ga = _build_ga(generations=1, population=pop)
        result = engine = RunEngine(ga)
        result = engine.run()
        assert isinstance(result, list)

    def test_signal_handlers_restored(self):
        ga = _build_ga(generations=1)
        original = signal.getsignal(signal.SIGINT)
        engine = RunEngine(ga)
        engine.run()
        assert signal.getsignal(signal.SIGINT) == original


# =====================================================================
# Teardown
# =====================================================================


class TestTeardown:
    def test_saves_final_checkpoint(self):
        ga = _build_ga(generations=1)
        engine = RunEngine(ga)
        pop = _FakePopulation(individuals=[_FakeIndividual()])
        engine._teardown(pop, None)
        # Should have called save_checkpoint for final
        ga.save_checkpoint.assert_called()

    def test_shuts_down_parallel_evaluator(self):
        eval_mock = MagicMock()
        ga = _build_ga(generations=1)
        ga.parallel_evaluator = eval_mock
        engine = RunEngine(ga)
        pop = _FakePopulation(individuals=[_FakeIndividual()])
        engine._teardown(pop, None)
        eval_mock.shutdown.assert_called_once()

    def test_closes_visualizer(self):
        viz_mock = MagicMock()
        ga = _build_ga(generations=1)
        ga.visualizer = viz_mock
        engine = RunEngine(ga)
        pop = _FakePopulation(individuals=[_FakeIndividual()])
        engine._teardown(pop, None)
        viz_mock.close.assert_called_once()

    def test_nsga2_mode_returns_list(self):
        ga = _build_ga(generations=1, mode="nsga2")
        ind = _FakeIndividual(objectives=[1.0, 0.5])
        pop = _FakePopulation(individuals=[ind])
        engine = RunEngine(ga)
        with patch(
            "genetic_algorithm.engine.runner.RunEngine._log_final_summary"
        ):
            result = engine._teardown(pop, None)
        assert isinstance(result, list)
