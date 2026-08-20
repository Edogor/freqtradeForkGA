"""Regression tests for surrogate filtering at the genome/fitness boundary."""

from unittest.mock import MagicMock, patch

import pytest

from genetic_algorithm.evaluation.surrogate import SurrogateModel


def test_mutated_skipped_candidate_is_rescored_with_nested_ga_mutation_rate():
    surrogate = SurrogateModel(
        {
            "genetic_algorithm": {"mutation_rate": 0.77},
            "surrogate": {
                "enabled": True,
                "filter_percentile": 50,
                "mutate_skipped": True,
            },
        }
    )
    surrogate._is_trained = True
    surrogate._model = object()
    surrogate._last_validation_r2 = 0.8
    surrogate._last_validation_samples = 20

    high_gene = object()
    low_gene = object()
    mutated_gene = object()
    high = MagicMock(strategy_gene=high_gene, metrics={})
    low = MagicMock(strategy_gene=low_gene, metrics={}, evaluated=False)
    mutated = MagicMock(strategy_gene=mutated_gene, metrics={}, evaluated=False)
    low.adopt_unevaluated_genome.side_effect = lambda source: setattr(
        low, "strategy_gene", source.strategy_gene
    )

    def _record_surrogate(fitness, **metadata):
        low.fitness = fitness
        low.raw_fitness = fitness
        low.evaluated = True
        low.metrics.update(metadata)

    low.set_surrogate_fitness.side_effect = _record_surrogate

    predictions = {
        high_gene: 0.9,
        low_gene: 0.1,
        mutated_gene: 0.4,
    }
    surrogate.predict = MagicMock(side_effect=lambda gene: predictions[gene])

    with patch(
        "genetic_algorithm.core.mutation.mutate",
        return_value=mutated,
    ) as mutate_mock:
        to_backtest, to_skip = surrogate.filter_candidates([high, low])

    assert to_backtest == [high]
    assert to_skip == [low]
    mutate_mock.assert_called_once_with(low, 0.77, surrogate._ga_config)
    assert surrogate.predict.call_count == 3
    assert low.strategy_gene is mutated_gene
    assert low.fitness == pytest.approx(0.4 * 0.85)
    assert low.metrics["predicted_fitness"] == pytest.approx(0.4)
    assert low.metrics["surrogate_mutated"] is True


def test_failed_post_mutation_prediction_fails_open_to_backtest():
    surrogate = SurrogateModel(
        {
            "genetic_algorithm": {"mutation_rate": 0.2},
            "surrogate": {
                "enabled": True,
                "filter_percentile": 50,
                "mutate_skipped": True,
            },
        }
    )
    surrogate._is_trained = True
    surrogate._model = object()
    surrogate._last_validation_r2 = 0.8
    surrogate._last_validation_samples = 20

    high_gene = object()
    low_gene = object()
    mutated_gene = object()
    high = MagicMock(strategy_gene=high_gene, metrics={})
    low = MagicMock(strategy_gene=low_gene, metrics={}, evaluated=False)
    mutated = MagicMock(strategy_gene=mutated_gene, metrics={}, evaluated=False)
    low.adopt_unevaluated_genome.side_effect = lambda source: setattr(
        low, "strategy_gene", source.strategy_gene
    )
    predictions = {high_gene: 0.9, low_gene: 0.1, mutated_gene: None}
    surrogate.predict = MagicMock(side_effect=lambda gene: predictions[gene])

    with patch("genetic_algorithm.core.mutation.mutate", return_value=mutated):
        to_backtest, to_skip = surrogate.filter_candidates([high, low])

    assert to_backtest == [high, low]
    assert to_skip == []
    assert not low.evaluated
