"""Fail-closed island result extraction and coordinator tests."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from genetic_algorithm.core.individual import Individual
from genetic_algorithm.core.strategy_gene import (
    ConditionGene,
    IndicatorGene,
    StrategyGene,
)
from genetic_algorithm.engine.island_results import (
    IslandResultContractError,
    extract_island_finalists,
)
from genetic_algorithm.engine.islands import IslandCoordinator
from genetic_algorithm.evaluation.panel_contract import (
    PANEL_ROLE_COMMON_REPLAY,
    build_evaluation_panel,
)


def _gene(individual_id: int, period: int) -> StrategyGene:
    return StrategyGene(
        generation=0,
        individual_id=individual_id,
        indicators=[
            IndicatorGene(
                type="RSI",
                parameters={"period": period},
                instance_id="RSI_0",
            )
        ],
        entry_conditions=[ConditionGene(indicator="RSI_0", operator="<", threshold=30.0)],
        exit_conditions=[ConditionGene(indicator="RSI_0", operator=">", threshold=70.0)],
        timeframe="1h",
        stoploss=-0.1,
    )


def _panel_id(*, fee: float = 0.001) -> str:
    return build_evaluation_panel(
        {
            "backtesting": {
                "pairs": ["BTC/USDT", "ETH/USDT"],
                "timerange": "20230101-20240101",
                "timeframe": "1h",
                "fee": fee,
            },
            "fitness_weights": {"profit": 1.0},
        },
        role=PANEL_ROLE_COMMON_REPLAY,
    ).panel_id


def _finalist(
    individual_id: int,
    *,
    period: int,
    fitness: float,
    panel_id: str,
    profit: float | None = None,
    num_trades: int | None = 20,
) -> Individual:
    candidate = Individual(strategy_gene=_gene(individual_id, period))
    metrics = {
        "profit": fitness * 10 if profit is None else profit,
        "num_trades": num_trades,
        "common_replay": True,
        "fitness_panel_id": panel_id,
        "fitness_panel_role": PANEL_ROLE_COMMON_REPLAY,
    }
    candidate.set_fitness(fitness, metrics)
    return candidate


def _valid_results():
    panel_id = _panel_id()
    first = _finalist(1, period=11, fitness=0.8, panel_id=panel_id)
    second = _finalist(2, period=22, fitness=0.4, panel_id=panel_id)
    third = _finalist(3, period=33, fitness=0.2, panel_id=panel_id)
    return (
        {
            "alpha": [first, second],
            "beta": [third],
            "__global__": [first, second, third],
        },
        (first, second, third),
    )


def test_extract_flattens_every_island_and_preserves_replay_rank():
    results, candidates = _valid_results()

    batch = extract_island_finalists(
        results,
        expected_islands=["alpha", "beta"],
    )

    assert batch.island_names == ("alpha", "beta")
    assert batch.flattened_candidates == candidates
    assert batch.finalists == candidates
    assert batch.candidates_for("beta") == (candidates[2],)
    assert batch.top(2) == candidates[:2]
    assert batch.common_panel_id == candidates[0].fitness_panel_id


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda results, candidates: results.pop("beta"),
            "key set differs",
        ),
        (
            lambda results, candidates: results.update({"unexpected": []}),
            "key set differs",
        ),
        (
            lambda results, candidates: results.update({"alpha": [object()]}),
            "non-Individual",
        ),
        (
            lambda results, candidates: results.update({"__global__": []}),
            "no common-replay finalist",
        ),
    ],
)
def test_extract_rejects_incomplete_or_untyped_result_matrix(mutate, message):
    results, candidates = _valid_results()
    mutate(results, candidates)

    with pytest.raises(IslandResultContractError, match=message):
        extract_island_finalists(results, expected_islands=["alpha", "beta"])


def test_extract_rejects_local_search_score_disguised_as_global():
    results, candidates = _valid_results()
    candidates[0].metrics["common_replay"] = False

    with pytest.raises(IslandResultContractError, match="COMMON_REPLAY"):
        extract_island_finalists(results, expected_islands=["alpha", "beta"])


def test_extract_rejects_candidate_not_present_in_any_island():
    results, candidates = _valid_results()
    outsider = _finalist(
        9,
        period=99,
        fitness=0.9,
        panel_id=str(candidates[0].fitness_panel_id),
    )
    results["__global__"] = [outsider, *results["__global__"]]

    with pytest.raises(IslandResultContractError, match="absent from all island"):
        extract_island_finalists(results, expected_islands=["alpha", "beta"])


def test_extract_rejects_duplicate_phenotype_and_unsorted_rank():
    results, candidates = _valid_results()
    duplicate = _finalist(
        8,
        period=11,
        fitness=0.7,
        panel_id=str(candidates[0].fitness_panel_id),
    )
    results["beta"].append(duplicate)
    results["__global__"] = [candidates[0], duplicate, candidates[1]]
    with pytest.raises(IslandResultContractError, match="duplicate"):
        extract_island_finalists(results, expected_islands=["alpha", "beta"])

    results, candidates = _valid_results()
    results["__global__"] = [candidates[1], candidates[0], candidates[2]]
    with pytest.raises(IslandResultContractError, match="not sorted"):
        extract_island_finalists(results, expected_islands=["alpha", "beta"])


def test_extract_rejects_mixed_panels_and_missing_metrics():
    results, candidates = _valid_results()
    candidates[1].fitness_panel_id = _panel_id(fee=0.002)
    candidates[1].metrics["fitness_panel_id"] = candidates[1].fitness_panel_id
    with pytest.raises(IslandResultContractError, match="one exact replay panel"):
        extract_island_finalists(results, expected_islands=["alpha", "beta"])

    results, candidates = _valid_results()
    candidates[0].metrics["profit"] = None
    with pytest.raises(IslandResultContractError, match="numeric metric 'profit'"):
        extract_island_finalists(results, expected_islands=["alpha", "beta"])

    results, candidates = _valid_results()
    candidates[0].metrics["num_trades"] = -1
    with pytest.raises(IslandResultContractError, match="invalid num_trades"):
        extract_island_finalists(results, expected_islands=["alpha", "beta"])


def test_coordinator_returns_validated_batch_not_backend_dict():
    results, candidates = _valid_results()
    backend = MagicMock()
    backend.island_configs = [
        SimpleNamespace(name="alpha"),
        SimpleNamespace(name="beta"),
    ]
    backend.evolve.return_value = results

    batch = IslandCoordinator(backend).evolve()

    assert batch.finalists == candidates
    backend.evolve.assert_called_once_with()


def test_coordinator_rejects_legacy_list_result():
    backend = MagicMock()
    backend.island_configs = [SimpleNamespace(name="alpha")]
    backend.evolve.return_value = ["legacy-result"]

    with pytest.raises(IslandResultContractError, match="must be a mapping"):
        IslandCoordinator(backend).evolve()
