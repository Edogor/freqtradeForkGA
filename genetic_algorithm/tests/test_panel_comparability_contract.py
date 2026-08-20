"""Regression contract for panel-bound fitness and cross-island ranking."""

from datetime import datetime, timedelta
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from genetic_algorithm.core.generic_island_model import GenericIslandModelEvolution
from genetic_algorithm.core.evolution import GeneticAlgorithm
from genetic_algorithm.core.island_model import IslandConfig, IslandModelEvolution
from genetic_algorithm.core.population import Population
from genetic_algorithm.core.strategy_gene import ConditionGene, IndicatorGene, StrategyGene
from genetic_algorithm.engine.hall_of_fame import HallOfFame
from genetic_algorithm.evaluation.panel_contract import (
    PANEL_ROLE_COMMON_REPLAY,
    PANEL_ROLE_OPTIMIZATION,
    build_evaluation_panel,
    replay_on_common_panel,
)
from genetic_algorithm.genome.individual import Individual
from genetic_algorithm.utils.regime_detector import RegimeSegment, RegimeType


def _config(*, pairs=None, fee=0.001, seed=42, profit_weight=0.5):
    return {
        "genetic_algorithm": {
            "random_seed": seed,
            "population_size": 10,
            "generations": 5,
        },
        "backtesting": {
            "pairs": pairs or ["BTC/USDT"],
            "timerange": "20230101-20240101",
            "timeframe": "1h",
            "fee": fee,
        },
        "strategy_constraints": {"timeframes": ["1h"]},
        "fitness_weights": {"profit": profit_weight, "drawdown": 1 - profit_weight},
        "fitness_penalties": {},
        "walk_forward": {"enabled": False},
        "pair_validation": {"enabled": False},
        "regime_aware": {"enabled": False},
        "monte_carlo": {"enabled": False},
        "evaluation_v2": {"enabled": False},
    }


def _gene(individual_id: int, period: int = 14) -> StrategyGene:
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
        entry_conditions=[
            ConditionGene(indicator="RSI_0", operator="<", threshold=30.0)
        ],
        exit_conditions=[
            ConditionGene(indicator="RSI_0", operator=">", threshold=70.0)
        ],
        timeframe="1h",
        stoploss=-0.1,
    )


def _measured(
    fitness: float,
    individual_id: int,
    panel_id: str,
    *,
    period: int | None = None,
) -> Individual:
    individual = Individual(
        strategy_gene=_gene(individual_id, period or 10 + individual_id)
    )
    individual.set_fitness(
        fitness,
        {"profit": fitness * 10, "num_trades": 20, "max_drawdown": 0.1},
    )
    individual.assign_fitness_panel(panel_id, PANEL_ROLE_OPTIMIZATION)
    return individual


def _panel(
    config=None,
    *,
    role=PANEL_ROLE_OPTIMIZATION,
    data_manifest_hash=None,
):
    return build_evaluation_panel(
        config or _config(),
        role=role,
        data_manifest_hash=data_manifest_hash,
    )


def test_panel_identity_ignores_search_budget_but_changes_evaluation_semantics():
    baseline = _panel(_config(seed=1))
    other_seed = _panel(_config(seed=999))
    other_seed_config = _config(seed=999)
    other_seed_config["genetic_algorithm"]["population_size"] = 200
    other_budget = _panel(other_seed_config)

    assert baseline.panel_id == other_seed.panel_id == other_budget.panel_id
    assert baseline.panel_id != _panel(_config(pairs=["ETH/USDT"])).panel_id
    assert baseline.panel_id != _panel(_config(fee=0.002)).panel_id
    assert baseline.panel_id != _panel(_config(profit_weight=0.7)).panel_id
    assert baseline.panel_id != _panel(
        _config(), role=PANEL_ROLE_COMMON_REPLAY
    ).panel_id


def test_only_explicit_content_manifest_allows_cross_run_panel_archive(tmp_path):
    unverified = _panel()
    first = HallOfFame(
        directory=str(tmp_path),
        required_panel_id=unverified.panel_id,
        required_panel_role=unverified.role,
    )
    second = HallOfFame(
        directory=str(tmp_path),
        required_panel_id=unverified.panel_id,
        required_panel_role=unverified.role,
    )
    assert unverified.data_identity_verified is False
    assert first.filepath != second.filepath

    verified = _panel(_config(), data_manifest_hash="a" * 64)
    verified_first = HallOfFame(
        directory=str(tmp_path),
        required_panel_id=verified.panel_id,
        required_panel_role=verified.role,
        panel_data_identity_verified=True,
    )
    verified_second = HallOfFame(
        directory=str(tmp_path),
        required_panel_id=verified.panel_id,
        required_panel_role=verified.role,
        panel_data_identity_verified=True,
    )
    assert verified.data_identity_verified is True
    assert verified_first.filepath == verified_second.filepath


def test_regime_segment_set_is_part_of_panel_identity():
    segment_a = RegimeSegment(
        segment_id="a",
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 2, 1),
        regime=RegimeType.BULLISH,
        confidence=0.8,
    )
    segment_b = RegimeSegment(
        segment_id="b",
        start_date=datetime(2023, 2, 1),
        end_date=datetime(2023, 3, 1),
        regime=RegimeType.BEARISH,
        confidence=0.8,
    )
    evaluator = SimpleNamespace(base_evaluator=object(), _optimization_segments=[])

    first = build_evaluation_panel(
        _config(), evaluator=evaluator, segments=[segment_a]
    )
    second = build_evaluation_panel(
        _config(), evaluator=evaluator, segments=[segment_b]
    )

    assert first.panel_id != second.panel_id


def test_individual_panel_evidence_survives_roundtrip_and_surrogate_clears_it():
    panel = _panel()
    individual = _measured(0.4, 1, panel.panel_id)

    restored = Individual.from_dict(individual.to_dict())

    assert restored.has_comparable_fitness(panel.panel_id)
    assert restored.fitness_panel_role == PANEL_ROLE_OPTIMIZATION

    restored.set_surrogate_fitness(
        0.3,
        predicted_fitness=0.35,
        surrogate_mutated=False,
        validation_r2=0.7,
    )
    assert restored.fitness_panel_id is None
    assert not restored.has_comparable_fitness(panel.panel_id)


def test_ga_boundary_stamps_success_and_failure_but_not_surrogate_estimate():
    panel = _panel()
    ga = object.__new__(GeneticAlgorithm)
    ga.evaluation_panel = panel
    ga._evaluation_island = "island_a"
    successful = Individual(strategy_gene=_gene(1, 11))
    successful.set_fitness(0.4, {"profit": 2.0, "num_trades": 20})
    failed = Individual(strategy_gene=_gene(2, 22))
    failed.set_fitness(0.0, {"error": "timeout", "num_trades": 0})
    estimated = Individual(strategy_gene=_gene(3, 33))
    estimated.set_surrogate_fitness(
        0.5,
        predicted_fitness=0.6,
        surrogate_mutated=False,
        validation_r2=0.8,
    )

    assert ga._stamp_evaluation_panel([successful, failed, estimated]) == 2
    assert successful.has_comparable_fitness(panel.panel_id)
    assert failed.fitness_panel_id == panel.panel_id
    assert failed.has_comparable_fitness(panel.panel_id) is False
    assert estimated.fitness_panel_id is None
    assert successful.metrics["evaluation_island"] == "island_a"


def test_hall_of_fame_accepts_only_its_bound_panel_and_reloads_fail_closed(tmp_path):
    panel_a = _panel(_config(pairs=["BTC/USDT"]))
    panel_b = _panel(_config(pairs=["ETH/USDT"]))
    wrong = _measured(0.99, 1, panel_b.panel_id)
    right = _measured(0.4, 2, panel_a.panel_id)
    population = Population(size=2)
    population.individuals = [wrong, right]

    hall = HallOfFame(
        directory=str(tmp_path),
        required_panel_id=panel_a.panel_id,
        required_panel_role=PANEL_ROLE_OPTIMIZATION,
    )
    assert hall.update(population, generation=1) == 1
    assert hall.entries[0].fitness == pytest.approx(0.4)
    assert hall.entries[0].fitness_panel_id == panel_a.panel_id

    restored_wrong_panel = HallOfFame(
        directory=str(tmp_path),
        required_panel_id=panel_b.panel_id,
        required_panel_role=PANEL_ROLE_OPTIMIZATION,
    )
    assert restored_wrong_panel.entries == []


def test_common_replay_reranks_local_scores_and_excludes_failed_replay():
    source_a = _panel(_config(pairs=["BTC/USDT"]))
    source_b = _panel(_config(pairs=["ETH/USDT"]))
    common = _panel(_config(pairs=["BTC/USDT", "ETH/USDT"]), role=PANEL_ROLE_COMMON_REPLAY)
    locally_high = _measured(0.95, 1, source_a.panel_id, period=11)
    locally_low = _measured(0.10, 2, source_b.panel_id, period=22)
    failed = _measured(0.80, 3, source_b.panel_id, period=33)
    evaluator = MagicMock()
    evaluator.evaluate.side_effect = [
        (0.2, {"profit": 1.0, "num_trades": 20}),
        (0.8, {"profit": 5.0, "num_trades": 30}),
        (0.0, {"error": "backtest failed", "num_trades": 0}),
    ]

    ranked = replay_on_common_panel(
        [locally_high, locally_low, failed],
        evaluator=evaluator,
        panel=common,
        logger=logging.getLogger("test.common-replay"),
    )

    assert ranked == [locally_low, locally_high]
    assert all(item.has_comparable_fitness(common.panel_id) for item in ranked)
    assert locally_low.metrics["source_fitness"] == pytest.approx(0.10)
    assert locally_low.metrics["source_panel_id"] == source_b.panel_id
    assert failed.has_comparable_fitness(common.panel_id) is False


def test_common_replay_deduplicates_same_phenotype_before_backtest():
    source = _panel()
    common = _panel(role=PANEL_ROLE_COMMON_REPLAY)
    first = _measured(0.4, 1, source.panel_id, period=14)
    duplicate = _measured(0.9, 2, source.panel_id, period=14)
    evaluator = MagicMock()
    evaluator.evaluate.return_value = (
        0.5,
        {"profit": 2.0, "num_trades": 20},
    )

    ranked = replay_on_common_panel(
        [first, duplicate],
        evaluator=evaluator,
        panel=common,
        logger=logging.getLogger("test.common-replay"),
    )

    assert ranked == [first]
    evaluator.evaluate.assert_called_once()


def test_common_replay_uses_parallel_batch_and_preserves_stable_ranking_context():
    source_a = _panel(_config(pairs=["BTC/USDT"]))
    source_b = _panel(_config(pairs=["ETH/USDT"]))
    common = _panel(
        _config(pairs=["BTC/USDT", "ETH/USDT"]),
        role=PANEL_ROLE_COMMON_REPLAY,
    )
    locally_high = _measured(0.95, 1, source_a.panel_id, period=11)
    locally_low = _measured(0.10, 2, source_b.panel_id, period=22)
    failed = _measured(0.80, 3, source_b.panel_id, period=33)
    evaluator = MagicMock()
    parallel = MagicMock()

    def evaluate_batch(candidates):
        candidates[0].set_fitness(0.2, {"profit": 1.0, "num_trades": 20})
        candidates[1].set_fitness(0.8, {"profit": 5.0, "num_trades": 30})
        candidates[2].set_fitness(
            0.0,
            {"error": "parallel backtest failed", "num_trades": 0},
        )

    parallel.evaluate_batch.side_effect = evaluate_batch

    ranked = replay_on_common_panel(
        [locally_high, locally_low, failed],
        evaluator=evaluator,
        panel=common,
        logger=logging.getLogger("test.parallel-common-replay"),
        parallel_evaluator=parallel,
    )

    assert ranked == [locally_low, locally_high]
    parallel.evaluate_batch.assert_called_once_with(
        [locally_high, locally_low, failed]
    )
    evaluator.evaluate.assert_not_called()
    assert locally_high.metrics["source_fitness"] == pytest.approx(0.95)
    assert locally_low.metrics["source_panel_id"] == source_b.panel_id
    assert failed.has_comparable_fitness(common.panel_id) is False


def test_cross_panel_merge_selection_is_round_robin_not_raw_score():
    model = object.__new__(GenericIslandModelEvolution)
    model._gene_hash = lambda individual: str(
        individual.strategy_gene.indicators[0].parameters["period"]
    )
    panel_a = _panel(_config(pairs=["BTC/USDT"]))
    panel_b = _panel(_config(pairs=["ETH/USDT"]))
    a1 = _measured(0.99, 1, panel_a.panel_id, period=11)
    a2 = _measured(0.98, 2, panel_a.panel_id, period=12)
    b1 = _measured(0.10, 3, panel_b.panel_id, period=21)
    b2 = _measured(0.09, 4, panel_b.panel_id, period=22)

    selected, comparable, _ = model._select_cross_panel_pool(
        [[a1, a2], [b1, b2]], 2
    )

    assert comparable is False
    assert selected == [a1, b1]

    b1.assign_fitness_panel(panel_a.panel_id, PANEL_ROLE_OPTIMIZATION)
    b2.assign_fitness_panel(panel_a.panel_id, PANEL_ROLE_OPTIMIZATION)
    selected, comparable, _ = model._select_cross_panel_pool(
        [[a1, a2], [b1, b2]], 2
    )
    assert comparable is True
    assert selected == [a1, a2]


def test_distinct_island_panels_disable_misconfigured_shared_evaluator():
    model = object.__new__(GenericIslandModelEvolution)
    panel_a = _panel(_config(pairs=["BTC/USDT"]))
    panel_b = _panel(_config(pairs=["ETH/USDT"]))
    model.islands = {
        "a": SimpleNamespace(evaluation_panel=panel_a),
        "b": SimpleNamespace(evaluation_panel=panel_b),
    }
    shared_evaluator = MagicMock()
    model._shared_parallel_evaluator = shared_evaluator
    model.logger = logging.getLogger("test.shared-panel")

    assert model._enforce_search_panel_contract() is False
    shared_evaluator.shutdown.assert_called_once()
    assert model._shared_parallel_evaluator is None


def test_regime_common_panel_uses_union_and_disables_specialist_bonus():
    model = object.__new__(IslandModelEvolution)
    model.config = _config()
    model.config["island_model"] = {"enabled": True}
    model.config["generic_island_model"] = {"enabled": False}
    model.config["regime_aware"] = {
        "enabled": True,
        "regime_specialization": {"enabled": True, "specialist_boost": 2.0},
    }
    start = datetime(2023, 1, 1)
    bullish = RegimeSegment(
        segment_id="bull",
        start_date=start,
        end_date=start + timedelta(days=30),
        regime=RegimeType.BULLISH,
        confidence=0.8,
    )
    bearish = RegimeSegment(
        segment_id="bear",
        start_date=start + timedelta(days=30),
        end_date=start + timedelta(days=60),
        regime=RegimeType.BEARISH,
        confidence=0.7,
    )
    model.island_configs = [
        IslandConfig("bull", "bullish", segments=[bullish]),
        IslandConfig("bear", "bearish", segments=[bearish]),
        IslandConfig("all", "balanced", segments=[bullish, bearish]),
    ]
    fake_evaluator = SimpleNamespace(
        base_evaluator=object(),
        _optimization_segments=[bullish, bearish],
    )

    with patch(
        "genetic_algorithm.core.island_model.RegimeAwareEvaluator",
        return_value=fake_evaluator,
    ) as evaluator_class:
        _, panel = model._build_common_replay_evaluator()

    call_config = evaluator_class.call_args.args[0]
    call_segments = evaluator_class.call_args.kwargs["segments"]["optimization"]
    assert call_config["regime_aware"]["regime_specialization"]["enabled"] is False
    assert [segment.segment_id for segment in call_segments] == ["bear", "bull"]
    assert panel.role == PANEL_ROLE_COMMON_REPLAY
