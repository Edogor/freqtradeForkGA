"""Contract regressions for GA operators that produce StrategyGene objects."""

from __future__ import annotations

import random

import pytest

from genetic_algorithm.core.crossover import _deduplicate_conditions, crossover
from genetic_algorithm.core.mutation import (
    clamp_condition_thresholds,
    mutate_condition_reassign,
    mutate_parameters,
    mutate_timeframes,
)
from genetic_algorithm.genome.gene import ConditionGene, IndicatorGene, StrategyGene
from genetic_algorithm.genome.individual import Individual
from genetic_algorithm.strategies.operator_registry import (
    is_valid_operator,
    resolve_indicator_type,
)


def _duplicate_fixed_column_parent(offset: int) -> Individual:
    """Create a repair-input parent that exposes dedup, trim and bad operators."""
    gene = StrategyGene(
        generation=0,
        individual_id=offset,
        indicators=[
            IndicatorGene(
                type="MACD",
                parameters={"fast_period": 8 + offset, "slow_period": 26},
                weight=1.0,
                instance_id="MACD_0",
            ),
            IndicatorGene(
                type="MACD",
                parameters={"fast_period": 12 + offset, "slow_period": 30},
                weight=0.9,
                instance_id="MACD_1",
            ),
            IndicatorGene(
                type="RSI",
                parameters={"period": 14 + offset},
                weight=0.8,
                instance_id="RSI_0",
            ),
            IndicatorGene(
                type="EMA",
                parameters={"period": 40 + offset},
                weight=0.1,
                instance_id="EMA_0",
            ),
        ],
        entry_conditions=[
            ConditionGene("MACD_0", ">", 0),
            ConditionGene("MACD_1", "<", 0),
            ConditionGene("EMA_0", "cross_above", 0),
        ],
        exit_conditions=[ConditionGene("RSI_0", ">", 70)],
        short_entry_conditions=[
            ConditionGene("MACD_1", "<", 0),
            ConditionGene("RSI_0", "not_an_operator", 70),
        ],
        short_exit_conditions=[
            ConditionGene("EMA_0", "cross_below", 0),
            ConditionGene("RSI_0", "not_an_operator", 30),
        ],
    )
    # Deliberately not a persistence-valid strategy-gene-v2: crossover must
    # repair the two synthetic invalid operators before its output can cross
    # the exact semantic boundary.
    assert StrategyGene.from_dict(gene.to_dict()).to_dict() == gene.to_dict()
    return Individual(strategy_gene=gene)


_CROSSOVER_CONFIG = {
    "indicators": {
        "min_entry_conditions": 2,
        "min_exit_conditions": 1,
        "max_per_strategy": 2,
    }
}


def _assert_exact_gene(gene: StrategyGene) -> None:
    indicator_ids = {indicator.instance_id for indicator in gene.indicators}
    conditions = (
        gene.entry_conditions
        + gene.exit_conditions
        + gene.short_entry_conditions
        + gene.short_exit_conditions
    )

    assert gene.get_missing_indicators() == []
    assert all(condition.indicator in indicator_ids for condition in conditions)
    assert all(
        is_valid_operator(resolve_indicator_type(condition.indicator), condition.operator)
        for condition in conditions
    )
    payload = gene.to_dict()
    assert StrategyGene.from_dict_exact(payload).to_dict() == payload


@pytest.mark.parametrize("method", ["single_point", "uniform", "component"])
def test_crossover_dedup_and_trim_outputs_are_exact(method: str) -> None:
    """MACD_1/EMA_0 must not survive after their indicators are removed."""
    for seed in range(25):
        random.seed(seed)
        children = crossover(
            _duplicate_fixed_column_parent(0),
            _duplicate_fixed_column_parent(1),
            generation=1,
            ind_id=100,
            method=method,
            config=_CROSSOVER_CONFIG,
        )

        for child in children:
            gene = child.strategy_gene
            assert len(gene.indicators) <= 2
            assert len(gene.entry_conditions) >= 2
            assert len(gene.exit_conditions) >= 1
            _assert_exact_gene(gene)


def test_prune_orphaned_conditions_can_reach_zero() -> None:
    gene = StrategyGene(
        generation=0,
        individual_id=0,
        indicators=[IndicatorGene("RSI", {"period": 14}, instance_id="RSI_0")],
        entry_conditions=[ConditionGene("MISSING_ENTRY_0", "<", 30)],
        exit_conditions=[ConditionGene("MISSING_EXIT_0", ">", 70)],
        short_entry_conditions=[ConditionGene("MISSING_SHORT_ENTRY_0", ">", 70)],
        short_exit_conditions=[ConditionGene("MISSING_SHORT_EXIT_0", "<", 30)],
    )

    assert gene.prune_orphaned_conditions() == 4
    assert gene.entry_conditions == []
    assert gene.exit_conditions == []
    assert gene.short_entry_conditions == []
    assert gene.short_exit_conditions == []


def test_condition_dedup_preserves_operator_parameters() -> None:
    conditions = [
        ConditionGene("RSI_0", "between", 20, threshold_upper=30),
        ConditionGene("RSI_0", "between", 20, threshold_upper=40),
        ConditionGene("RSI_0", "increasing", 0, lookback=3),
        ConditionGene("RSI_0", "increasing", 0, lookback=8),
    ]

    assert _deduplicate_conditions(conditions) == conditions


def test_between_normalization_always_produces_a_strict_interval() -> None:
    bounded = ConditionGene("RSI_0", "between", 100.0, threshold_upper=100.0)
    unbounded = ConditionGene("EMA_0", "between", 0.0, threshold_upper=0.0)

    clamp_condition_thresholds([bounded, unbounded])

    assert 0.0 <= bounded.threshold < bounded.threshold_upper <= 100.0
    assert unbounded.threshold < unbounded.threshold_upper


def _condition_reassign_parent() -> Individual:
    return Individual(
        strategy_gene=StrategyGene(
            generation=0,
            individual_id=0,
            indicators=[
                IndicatorGene("RSI", {"period": 14}, instance_id="RSI_0"),
                IndicatorGene("EMA", {"period": 20}, instance_id="EMA_0"),
                IndicatorGene(
                    "MACD",
                    {"fast_period": 12, "slow_period": 26, "signal_period": 9},
                    instance_id="MACD_0",
                ),
            ],
            entry_conditions=[
                ConditionGene("RSI_0", "<", 30),
                ConditionGene("EMA_0", "cross_above", 0),
            ],
            exit_conditions=[ConditionGene("RSI_0", ">", 70)],
        )
    )


def test_condition_reassign_deduplicates_before_returning() -> None:
    # Seed 2 reassigns RSI_0 onto the already identical EMA_0 condition.
    random.seed(2)
    gene = mutate_condition_reassign(
        _condition_reassign_parent(),
        mutation_rate=1.0,
        config={"indicators": {}},
    ).strategy_gene

    assert len(gene.entry_conditions) == 1
    _assert_exact_gene(gene)

    # Exercise the remaining branch choices as a cheap producer contract test.
    for seed in range(50):
        random.seed(seed)
        gene = mutate_condition_reassign(
            _condition_reassign_parent(),
            mutation_rate=1.0,
            config={"indicators": {}},
        ).strategy_gene
        _assert_exact_gene(gene)


def test_parameter_mutation_deduplicates_converged_short_thresholds(
    monkeypatch,
) -> None:
    gene = StrategyGene(
        generation=0,
        individual_id=0,
        indicators=[IndicatorGene("ADX", {"period": 14}, instance_id="ADX_0")],
        entry_conditions=[ConditionGene("ADX_0", ">", 20)],
        short_entry_conditions=[
            ConditionGene("ADX_0", ">", 22),
            ConditionGene("ADX_0", ">", 32),
        ],
    )

    monkeypatch.setattr(random, "random", lambda: 0.0)

    def _fixed_randint(lower: int, upper: int) -> int:
        if (lower, upper) == (20, 35):
            return 25
        return (lower + upper) // 2

    monkeypatch.setattr(random, "randint", _fixed_randint)
    result = mutate_parameters(
        Individual(strategy_gene=gene),
        mutation_rate=1.0,
        config={
            "indicators": {"ADX": {}},
            "strategy_constraints": {},
        },
    ).strategy_gene

    assert len(result.short_entry_conditions) == 1
    assert result.short_entry_conditions[0].threshold == 25
    _assert_exact_gene(result)


def test_remove_timeframe_prunes_short_condition_references(monkeypatch) -> None:
    gene = StrategyGene(
        generation=0,
        individual_id=0,
        indicators=[
            IndicatorGene("RSI", {"period": 14}, instance_id="RSI_0"),
            IndicatorGene(
                "EMA",
                {"period": 50},
                instance_id="EMA_1h_0",
                timeframe="1h",
            ),
        ],
        entry_conditions=[ConditionGene("RSI_0", "<", 30)],
        exit_conditions=[ConditionGene("RSI_0", ">", 70)],
        short_entry_conditions=[ConditionGene("EMA_1h_0", "cross_below", 0)],
        short_exit_conditions=[ConditionGene("EMA_1h_0", "cross_above", 0)],
        informative_timeframes=["1h"],
        can_short=True,
    )

    original_choice = random.choice

    def _choose_remove(sequence):
        if "remove_timeframe" in sequence:
            return "remove_timeframe"
        return original_choice(sequence)

    monkeypatch.setattr(random, "choice", _choose_remove)
    result = mutate_timeframes(
        Individual(strategy_gene=gene),
        mutation_rate=1.0,
        config={
            "multi_timeframe": {
                "enabled": True,
                "available": ["1h", "4h"],
                "max_timeframes": 1,
            },
            "indicators": {"available": ["RSI", "EMA"]},
        },
    ).strategy_gene

    assert result.informative_timeframes == []
    assert result.short_entry_conditions == []
    assert result.short_exit_conditions == []
    _assert_exact_gene(result)
