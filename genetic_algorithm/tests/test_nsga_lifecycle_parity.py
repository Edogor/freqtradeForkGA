"""End-to-end NSGA lifecycle and evaluator-parity regressions."""

from concurrent.futures import Future
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from genetic_algorithm.core.evolution import GeneticAlgorithm
from genetic_algorithm.core.individual import Individual
from genetic_algorithm.core.population import Population
from genetic_algorithm.core.strategy_gene import (
    ConditionGene,
    IndicatorGene,
    StrategyGene,
)
from genetic_algorithm.engine.generation import GenerationStep
from genetic_algorithm.evaluation import parallel
from genetic_algorithm.evaluation.parallel import ParallelEvaluator


def _gene(individual_id: int, period: int = 14, generation: int = 0) -> StrategyGene:
    return StrategyGene(
        generation=generation,
        individual_id=individual_id,
        indicators=[
            IndicatorGene(
                type="RSI",
                parameters={"period": period},
                instance_id="RSI_0",
            )
        ],
        entry_conditions=[
            ConditionGene(
                indicator="RSI_0",
                operator="cross_below",
                threshold=30,
                logic="AND",
            )
        ],
        exit_conditions=[
            ConditionGene(
                indicator="RSI_0",
                operator="cross_above",
                threshold=70,
                logic="AND",
            )
        ],
        timeframe="5m",
        stoploss=-0.1,
    )


def _evaluated_parent(individual_id: int, period: int) -> Individual:
    individual = Individual(strategy_gene=_gene(individual_id, period))
    metrics = {
        "profit": 10.0,
        "max_drawdown": 0.1,
        "sharpe_ratio": 1.0,
        "num_trades": 20,
    }
    individual.set_fitness(0.5, metrics)
    individual.set_objectives([0.1, -0.1], metrics)
    individual.rank = 1
    individual.crowding_distance = float("inf")
    return individual


def _generation_step() -> GenerationStep:
    config = {
        "genetic_algorithm": {"generations": 2},
        "indicators": {
            "min_entry_conditions": 1,
            "min_exit_conditions": 1,
        },
    }
    return GenerationStep(
        population_size=2,
        elite_size=0,
        mode="nsga2",
        config=config,
        logger=logging.getLogger("test.nsga.lifecycle"),
        strategy_generator=MagicMock(),
        crossover_rate=0.0,
        crossover_method="uniform",
        tournament_size=2,
        selection_method="nsga2",
        allow_self_crossover=True,
        adaptive_tournament=False,
        random_immigrants=0,
        diversity_threshold=0.0,
    )


def test_nsga_offspring_are_evaluated_before_mu_plus_lambda_selection():
    parents = Population(size=2, generation=0)
    parents.add_individual(_evaluated_parent(0, 10))
    parents.add_individual(_evaluated_parent(1, 20))

    step = _generation_step()
    events = []
    original_selection = step.select_nsga2_survivors

    def select_after_evaluation(parent_population, offspring_population):
        events.append("select")
        assert all(ind.evaluated for ind in offspring_population)
        assert all(ind.objectives is not None for ind in offspring_population)
        return original_selection(parent_population, offspring_population)

    step.select_nsga2_survivors = select_after_evaluation

    ga = object.__new__(GeneticAlgorithm)
    ga._generation_step = step
    ga._immigrant_provider = None
    ga._map_elites = None
    ga._aos = None
    ga.llm_enabled = False
    ga._external_immigrants = []
    ga.current_generation = 0
    ga.mutation_rate = 1.0
    ga.no_improvement_count = 0
    ga.best_fitness_ever = 0.5
    ga.generation_stats = []
    ga.mode = "nsga2"
    ga.logger = logging.getLogger("test.nsga.lifecycle.ga")

    def evaluate_offspring(population):
        events.append("evaluate")
        for child in population:
            metrics = {
                "profit": 100.0,
                "max_drawdown": 0.01,
                "sharpe_ratio": 3.0,
                "num_trades": 30,
            }
            child.set_fitness(1.0, metrics)
            child.set_objectives([1.0, -0.01], metrics)

    ga.evaluate_population = evaluate_offspring

    def force_genetic_change(individual, mutation_rate, config):
        child = Individual(strategy_gene=individual.strategy_gene.copy())
        child.strategy_gene.indicators[0].parameters["period"] += 100
        return child

    with patch(
        "genetic_algorithm.engine.generation.mutate",
        side_effect=force_genetic_change,
    ):
        survivors = GeneticAlgorithm.create_next_generation(ga, parents)

    assert events == ["evaluate", "select"]
    assert len(survivors.individuals) == 2
    assert all(
        ind.strategy_gene.indicators[0].parameters["period"] >= 110
        for ind in survivors
    )
    assert all(ind.strategy_gene.generation == 1 for ind in survivors)


def test_worker_receives_and_applies_nsga_min_trades(monkeypatch):
    class Evaluator:
        @staticmethod
        def evaluate(strategy_gene):
            return 0.9, {
                "profit": 100.0,
                "max_drawdown": 0.01,
                "sharpe_ratio": 3.0,
                "num_trades": 3,
            }

    monkeypatch.setattr(parallel, "_worker_evaluator", Evaluator())

    result = parallel._evaluate_strategy_in_worker(
        _gene(0).to_dict(),
        0,
        True,
        [
            {"name": "profit", "type": "maximize", "scale": 100.0},
            {"name": "max_drawdown", "type": "minimize", "scale": 1.0},
        ],
        nsga2_min_trades=5,
    )

    assert result["success"] is True
    assert result["objectives"] == [-1e6, -1.0]


def test_parallel_batch_forwards_nsga_min_trades_to_worker():
    submitted = []

    class Executor:
        def submit(self, fn, *args, **kwargs):
            submitted.append((fn, args, kwargs))
            future = Future()
            future.set_result(
                {
                    "index": args[1],
                    "fitness": 0.5,
                    "metrics": {"profit": 1.0, "num_trades": 5},
                    "objectives": [0.01],
                    "success": True,
                }
            )
            return future

    evaluator = object.__new__(ParallelEvaluator)
    evaluator.config = {"nsga2": {"min_trades": 17}}
    evaluator.nsga2_mode = True
    evaluator.objectives_config = [
        {"name": "profit", "type": "maximize", "scale": 100.0}
    ]
    evaluator._pool_generation_count = 1
    evaluator.backtest_timeout = 0
    evaluator.num_workers = 1
    evaluator._check_pool_health = MagicMock(return_value=True)
    evaluator._get_executor = MagicMock(return_value=Executor())

    individual = Individual(strategy_gene=_gene(0))
    result = evaluator.evaluate_batch([individual])

    assert result.successful == 1
    assert submitted[0][2]["nsga2_min_trades"] == 17
    assert individual.objectives == [0.01]


class _SynchronousExecutor:
    def submit(self, fn, *args, **kwargs):
        future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except Exception as exc:  # pragma: no cover - worker catches evaluator errors
            future.set_exception(exc)
        return future


def _parallel_evaluator(config):
    evaluator = object.__new__(ParallelEvaluator)
    evaluator.config = config
    evaluator.nsga2_mode = True
    evaluator.objectives_config = config["nsga2"]["objectives"]
    evaluator._pool_generation_count = 1
    evaluator.backtest_timeout = 0
    evaluator.num_workers = 1
    evaluator._check_pool_health = MagicMock(return_value=True)
    evaluator._get_executor = MagicMock(return_value=_SynchronousExecutor())
    return evaluator


def _sequential_ga(config, fitness_evaluator):
    ga = object.__new__(GeneticAlgorithm)
    ga.progress_enabled = False
    ga.monitor = MagicMock(active=False)
    ga.logger = logging.getLogger("test.nsga.sequential")
    ga.current_generation = 0
    ga.mode = "nsga2"
    ga.nsga2_config = config["nsga2"]
    ga.objectives_config = config["nsga2"]["objectives"]
    ga.fitness_evaluator = fitness_evaluator
    return ga


@pytest.mark.parametrize("num_trades", [3, 30])
def test_parallel_and_sequential_nsga_results_are_identical(
    monkeypatch, num_trades
):
    config = {
        "nsga2": {
            "min_trades": 5,
            "objectives": [
                {"name": "profit", "type": "maximize", "scale": 100.0},
                {
                    "name": "max_drawdown",
                    "type": "minimize",
                    "scale": 1.0,
                },
            ],
        }
    }

    class Evaluator:
        @staticmethod
        def evaluate(strategy_gene, strategy_name=None):
            return 0.75, {
                "profit": 25.0,
                "max_drawdown": 0.12,
                "sharpe_ratio": 1.5,
                "num_trades": num_trades,
            }

    evaluator = Evaluator()
    sequential = Individual(strategy_gene=_gene(0))
    parallel_individual = Individual(strategy_gene=_gene(0))

    _sequential_ga(config, evaluator)._evaluate_population_sequential([sequential])
    monkeypatch.setattr(parallel, "_worker_evaluator", evaluator)
    _parallel_evaluator(config).evaluate_batch([parallel_individual])

    assert parallel_individual.fitness == sequential.fitness
    assert parallel_individual.raw_fitness == sequential.raw_fitness
    assert parallel_individual.metrics == sequential.metrics
    assert parallel_individual.objectives == sequential.objectives
    assert parallel_individual.evaluated == sequential.evaluated is True


def test_parallel_and_sequential_failures_have_identical_semantics(monkeypatch):
    config = {
        "nsga2": {
            "min_trades": 5,
            "objectives": [
                {"name": "profit", "type": "maximize", "scale": 100.0}
            ],
        }
    }

    class Evaluator:
        @staticmethod
        def evaluate(strategy_gene, strategy_name=None):
            raise RuntimeError("same failure")

    evaluator = Evaluator()
    sequential = Individual(strategy_gene=_gene(0))
    parallel_individual = Individual(strategy_gene=_gene(0))

    _sequential_ga(config, evaluator)._evaluate_population_sequential([sequential])
    monkeypatch.setattr(parallel, "_worker_evaluator", evaluator)
    _parallel_evaluator(config).evaluate_batch([parallel_individual])

    assert parallel_individual.fitness == sequential.fitness == 0.0
    assert parallel_individual.raw_fitness == sequential.raw_fitness == 0.0
    assert parallel_individual.metrics == sequential.metrics
    assert parallel_individual.objectives is sequential.objectives is None
    assert parallel_individual.evaluated == sequential.evaluated is True


def test_parallel_worker_config_preserves_walk_forward_semantics():
    config = {
        "walk_forward": {
            "enabled": True,
            "aggregation": "weighted",
            "gap_penalty": {"enabled": False},
        }
    }

    worker_config = parallel._build_worker_config(config)

    assert worker_config == config
    assert worker_config is not config
    assert worker_config["walk_forward"] is not config["walk_forward"]
    assert worker_config["walk_forward"]["enabled"] is True


def test_parallel_walk_forward_is_not_recomputed_post_hoc():
    ga = object.__new__(GeneticAlgorithm)
    ga.config = {"walk_forward": {"enabled": True}}
    ga.parallel_enabled = True
    ga.parallel_evaluator = SimpleNamespace(walk_forward_in_workers=True)
    ga.fitness_evaluator = MagicMock()

    population = Population(size=0, generation=0)

    assert ga._post_hoc_walk_forward_validation(population) == 0
    ga.fitness_evaluator.evaluate.assert_not_called()


def test_parallel_walk_forward_without_parity_capability_fails_closed():
    ga = object.__new__(GeneticAlgorithm)
    ga.config = {"walk_forward": {"enabled": True}}
    ga.parallel_enabled = True
    ga.parallel_evaluator = SimpleNamespace(walk_forward_in_workers=False)

    with pytest.raises(RuntimeError, match="does not guarantee.*parity"):
        ga._post_hoc_walk_forward_validation(Population(size=0, generation=0))


def test_deferred_validation_metrics_refresh_nsga_objectives():
    ga = object.__new__(GeneticAlgorithm)
    ga.mode = "nsga2"
    ga.nsga2_config = {"min_trades": 5}
    ga.objectives_config = [
        {"name": "profit", "type": "maximize", "scale": 100.0},
        {"name": "max_drawdown", "type": "minimize", "scale": 1.0},
    ]
    ga.logger = logging.getLogger("test.nsga.refresh")

    individual = _evaluated_parent(0, 14)
    individual.metrics = {
        "profit": -20.0,
        "max_drawdown": 0.3,
        "num_trades": 10,
    }
    population = Population(size=1, generation=0)
    population.add_individual(individual)

    refreshed = ga._refresh_nsga2_objectives(
        population,
        reason="test validation",
    )

    assert refreshed == 1
    assert individual.objectives == pytest.approx([-0.2, -0.3])


def test_invalid_deferred_metrics_fail_closed_instead_of_using_stale_objectives():
    ga = object.__new__(GeneticAlgorithm)
    ga.mode = "nsga2"
    ga.nsga2_config = {"min_trades": 0}
    ga.objectives_config = [
        {"name": "profit", "type": "maximize", "scale": 100.0}
    ]
    ga.logger = logging.getLogger("test.nsga.refresh.invalid")

    individual = _evaluated_parent(0, 14)
    individual.metrics = {"num_trades": 10}
    population = Population(size=1, generation=0)
    population.add_individual(individual)

    with pytest.raises(RuntimeError, match="objective refresh failed"):
        ga._refresh_nsga2_objectives(population, reason="invalid validation")


def test_nsga_surrogate_combination_is_rejected(tmp_path):
    from genetic_algorithm.config.schema import resolve_config
    import yaml

    path = tmp_path / "nsga-surrogate.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "genetic_algorithm": {
                    "mode": "nsga2",
                    "fitness_sharing": False,
                },
                "surrogate": {"enabled": True},
            }
        ),
        encoding="utf-8",
    )

    assert any(
        "surrogate-only candidates have no measured objective vector" in error
        for error in resolve_config(path).errors
    )
