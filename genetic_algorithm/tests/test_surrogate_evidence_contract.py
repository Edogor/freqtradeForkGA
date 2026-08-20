"""Fitness provenance and final-boundary tests for surrogate-assisted search."""

import time
from unittest.mock import patch

import pytest

from genetic_algorithm.core.population import Population
from genetic_algorithm.core.strategy_gene import ConditionGene, IndicatorGene, StrategyGene
from genetic_algorithm.engine.hall_of_fame import HallOfFame, HallOfFameEntry
from genetic_algorithm.evaluation.surrogate import SurrogateModel
from genetic_algorithm.genome.individual import (
    FITNESS_EVIDENCE_BACKTEST_FAILED,
    FITNESS_EVIDENCE_BACKTEST_VALID,
    FITNESS_EVIDENCE_SURROGATE,
    Individual,
)
from genetic_algorithm.orchestration.candidate_export_v2 import (
    CandidateExportError,
    freeze_hof_entry_v2,
    freeze_individual_v2,
)
from genetic_algorithm.orchestration.evolution_worker_v2 import _usable_results


def _gene(generation: int = 0, individual_id: int = 1) -> StrategyGene:
    return StrategyGene(
        generation=generation,
        individual_id=individual_id,
        indicators=[
            IndicatorGene(
                type="RSI",
                parameters={"period": 14},
                instance_id="RSI_0",
            )
        ],
        entry_conditions=[
            ConditionGene(
                indicator="RSI_0",
                operator="<",
                threshold=30.0,
            )
        ],
        exit_conditions=[
            ConditionGene(
                indicator="RSI_0",
                operator=">",
                threshold=70.0,
            )
        ],
        stoploss=-0.1,
        timeframe="1h",
        minimal_roi={"0": 0.04, "30": 0.02},
        max_open_trades=2,
    )


def _config() -> dict:
    return {
        "backtesting": {
            "pairs": ["BTC/USDT"],
            "timeframe": "1h",
            "timerange": "20230101-20230401",
        },
        "indicators": {
            "available": ["RSI"],
            "min_entry_conditions": 1,
            "min_exit_conditions": 1,
        },
        "strategy_constraints": {"timeframes": ["1h"]},
        "short_selling": {"enabled": False},
        "multi_timeframe": {"enabled": False},
        "regime_aware": {"enabled": False},
    }


def _measured(fitness: float = 0.5, individual_id: int = 1) -> Individual:
    individual = Individual(strategy_gene=_gene(individual_id=individual_id))
    individual.set_fitness(
        fitness,
        {
            "profit": fitness * 10,
            "num_trades": 20,
            "max_drawdown": 0.1,
            "sharpe_ratio": 1.0,
        },
    )
    return individual


def _surrogate(fitness: float = 0.9, individual_id: int = 2) -> Individual:
    individual = Individual(strategy_gene=_gene(individual_id=individual_id))
    individual.set_surrogate_fitness(
        fitness,
        predicted_fitness=fitness / 0.85,
        surrogate_mutated=False,
        validation_r2=0.8,
    )
    return individual


def test_fitness_evidence_roundtrip_and_real_backtest_replaces_estimate():
    individual = _surrogate()

    restored = Individual.from_dict(individual.to_dict())

    assert restored.fitness_evidence == FITNESS_EVIDENCE_SURROGATE
    assert restored.has_measured_fitness is False

    restored.set_fitness(0.4, {"profit": 2.0, "num_trades": 12})

    assert restored.fitness_evidence == FITNESS_EVIDENCE_BACKTEST_VALID
    assert restored.has_measured_fitness is True
    assert "surrogate_fitness" not in restored.metrics


def test_failed_backtest_is_not_measured_finalist():
    individual = Individual(strategy_gene=_gene())
    individual.set_fitness(0.0, {"error": "timeout", "num_trades": 0})

    assert individual.fitness_evidence == FITNESS_EVIDENCE_BACKTEST_FAILED
    assert individual.has_measured_fitness is False


def test_population_and_v2_worker_shortlist_ignore_higher_surrogate_score():
    measured = _measured(0.4)
    estimated = _surrogate(0.99)
    population = Population(size=2)
    population.individuals = [estimated, measured]

    assert population.get_best(1) == [estimated]
    assert population.get_best_measured(1) == [measured]
    assert _usable_results([estimated, measured]) == [measured]


def test_hall_of_fame_archives_only_measured_candidates(tmp_path):
    hall = HallOfFame(directory=str(tmp_path), max_size=10, min_fitness=0.0)
    population = Population(size=2)
    population.individuals = [_surrogate(0.99), _measured(0.4)]

    added = hall.update(population, generation=3)

    assert added == 1
    assert len(hall.entries) == 1
    assert hall.entries[0].fitness == pytest.approx(0.4)
    assert hall.entries[0].fitness_evidence == FITNESS_EVIDENCE_BACKTEST_VALID
    assert HallOfFame(directory=str(tmp_path)).entries[0].has_measured_fitness


def test_hall_of_fame_ignores_persisted_surrogate_entry(tmp_path):
    hall = HallOfFame(directory=str(tmp_path))
    estimated = HallOfFameEntry(
        strategy_gene_dict=_gene().to_dict(),
        fitness=0.9,
        metrics={"surrogate_evaluated": True, "surrogate_fitness": 1.0},
        generation_found=2,
        run_timestamp=time.time(),
    )
    hall.entries = [estimated]
    hall._save()

    restored = HallOfFame(directory=str(tmp_path))

    assert restored.entries == []


def test_candidate_export_rejects_surrogate_individual_and_hof_entry():
    estimated = _surrogate()
    estimated_entry = HallOfFameEntry(
        strategy_gene_dict=estimated.strategy_gene.to_dict(),
        fitness=float(estimated.fitness),
        metrics=estimated.metrics,
        generation_found=1,
        run_timestamp=time.time(),
        fitness_evidence=FITNESS_EVIDENCE_SURROGATE,
    )

    with pytest.raises(CandidateExportError, match="real-backtest evidence"):
        freeze_individual_v2(estimated, _config())
    with pytest.raises(CandidateExportError, match="real-backtest evidence"):
        freeze_hof_entry_v2(estimated_entry, _config())


def test_surrogate_ready_requires_generation_oos_quality_gate():
    surrogate = SurrogateModel(
        {
            "surrogate": {
                "enabled": True,
                "min_validation_r2": 0.3,
                "min_validation_samples": 10,
            }
        }
    )
    surrogate._is_trained = True
    surrogate._model = object()
    surrogate._last_validation_samples = 10

    surrogate._last_validation_r2 = 0.29
    assert surrogate.ready is False

    surrogate._last_validation_r2 = 0.3
    assert surrogate.ready is True


def test_training_uses_newest_generation_as_disjoint_validation_group():
    surrogate = SurrogateModel(
        {
            "surrogate": {
                "enabled": True,
                "min_training_samples": 20,
                "validation_fraction": 0.2,
                "min_validation_samples": 10,
                "min_validation_r2": 0.3,
                "retrain_interval": 1,
            }
        }
    )
    surrogate._training_X = [
        [float(index % 5)] * 65 for index in range(30)
    ]
    surrogate._training_y = [float(index % 7) / 7 for index in range(30)]
    surrogate._training_generations = [0] * 20 + [1] * 10
    surrogate._training_fingerprints = {str(index) for index in range(30)}

    with patch("sklearn.metrics.r2_score", return_value=0.8):
        assert surrogate._train() is True

    assert surrogate._last_validation_generations == [1]
    assert surrogate._last_validation_samples == 10
    assert surrogate.ready is True


def test_training_refuses_random_split_when_only_one_generation_exists():
    surrogate = SurrogateModel(
        {
            "surrogate": {
                "enabled": True,
                "min_training_samples": 20,
                "min_validation_samples": 5,
            }
        }
    )
    surrogate._training_X = [[float(index)] * 65 for index in range(20)]
    surrogate._training_y = [float(index) for index in range(20)]
    surrogate._training_generations = [0] * 20

    assert surrogate._train() is False
    assert surrogate.ready is False


def test_training_target_uses_measured_raw_fitness_not_shared_fitness():
    surrogate = SurrogateModel({"surrogate": {"enabled": True}})
    individual = _measured(0.4)
    individual.set_shared_fitness(0.05)
    population = Population(size=1)
    population.individuals = [individual]

    assert surrogate.add_training_data(population) == 1
    assert surrogate._training_y == pytest.approx([0.4])


def test_checkpoint_restore_cannot_reuse_unserialized_model_or_quality_gate():
    surrogate = SurrogateModel(
        {
            "surrogate": {
                "enabled": True,
                "min_validation_r2": 0.3,
                "min_validation_samples": 10,
            }
        }
    )
    surrogate._is_trained = True
    surrogate._model = object()
    surrogate._last_validation_r2 = 0.8
    surrogate._last_validation_samples = 20
    state = surrogate.to_dict()

    restored = SurrogateModel(surrogate._ga_config)
    restored.load_from_dict(state)

    assert restored._is_trained is False
    assert restored._model is None
    assert restored.ready is False


def test_hall_of_fame_rejects_unknown_evidence_state():
    with pytest.raises(ValueError, match="Unknown hall-of-fame fitness evidence"):
        HallOfFameEntry(
            strategy_gene_dict=_gene().to_dict(),
            fitness=0.5,
            metrics={},
            generation_found=1,
            run_timestamp=time.time(),
            fitness_evidence="MAYBE_REAL",
        )
