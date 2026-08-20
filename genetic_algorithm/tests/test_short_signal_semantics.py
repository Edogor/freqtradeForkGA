"""Side-aware signal generation and mutation regressions."""

import random

import pytest

from genetic_algorithm.core.strategy_gene import ConditionGene, IndicatorGene
from genetic_algorithm.engine.operators.mutation import (
    _create_random_condition,
    _mutate_condition_threshold,
)
from genetic_algorithm.genome.codegen import StrategyGenerator


def _generator() -> StrategyGenerator:
    return StrategyGenerator(
        {
            "indicators": {
                "available": ["RSI"],
                "min_per_strategy": 1,
                "max_per_strategy": 1,
                "min_entry_conditions": 1,
                "max_entry_conditions": 1,
                "min_exit_conditions": 1,
                "max_exit_conditions": 1,
                "RSI": {
                    "period": [14, 14],
                    "buy_threshold": [30, 30],
                    "sell_threshold": [70, 70],
                },
            },
            "strategy_constraints": {
                "timeframes": ["5m"],
                "stoploss_range": [-0.1, -0.1],
                "roi_range": [0.01, 0.02],
                "max_open_trades_range": [1, 1],
            },
            "short_selling": {
                "enabled": True,
                "probability": 1.0,
                "independent_conditions": True,
            },
        }
    )


@pytest.mark.parametrize(
    ("indicator_type", "bullish_operator", "bearish_operator"),
    [
        ("RSI", "cross_below", "cross_above"),
        ("MACD", "cross_above", "cross_below"),
        ("BBANDS", "cross_below", "cross_above"),
        ("EMA", "cross_above", "cross_below"),
        ("CMF", ">", "<"),
        ("CDL_ENGULFING", ">", "<"),
    ],
)
def test_generator_mirrors_direction_between_long_and_short(
    indicator_type, bullish_operator, bearish_operator
):
    generator = _generator()
    indicator = IndicatorGene(type=indicator_type, parameters={})

    long_entry = generator._generate_condition_for_indicator(
        indicator, is_entry=True, side="long"
    )
    long_exit = generator._generate_condition_for_indicator(
        indicator, is_entry=False, side="long"
    )
    short_entry = generator._generate_condition_for_indicator(
        indicator, is_entry=True, side="short"
    )
    short_exit = generator._generate_condition_for_indicator(
        indicator, is_entry=False, side="short"
    )

    assert long_entry.operator == short_exit.operator == bullish_operator
    assert long_exit.operator == short_entry.operator == bearish_operator


def test_generator_keeps_activity_filters_entry_exit_based():
    generator = _generator()
    atr = IndicatorGene(type="ATR", parameters={})

    assert generator._generate_condition_for_indicator(
        atr, True, "long"
    ).operator == ">"
    assert generator._generate_condition_for_indicator(
        atr, True, "short"
    ).operator == ">"
    assert generator._generate_condition_for_indicator(
        atr, False, "long"
    ).operator == "<"
    assert generator._generate_condition_for_indicator(
        atr, False, "short"
    ).operator == "<"


def test_one_sided_candles_are_available_only_for_matching_direction():
    generator = _generator()
    bullish = IndicatorGene(type="CDL_HAMMER", parameters={})
    bearish = IndicatorGene(type="CDL_EVENINGSTAR", parameters={})

    assert generator._generate_condition_for_indicator(bullish, True, "long")
    assert generator._generate_condition_for_indicator(bullish, False, "short")
    assert generator._generate_condition_for_indicator(bullish, True, "short") is None
    assert generator._generate_condition_for_indicator(bullish, False, "long") is None

    assert generator._generate_condition_for_indicator(bearish, True, "short")
    assert generator._generate_condition_for_indicator(bearish, False, "long")
    assert generator._generate_condition_for_indicator(bearish, True, "long") is None
    assert generator._generate_condition_for_indicator(bearish, False, "short") is None


def test_independent_short_strategy_uses_bearish_entry_and_bullish_exit():
    random.seed(7)

    strategy = _generator().generate_random_strategy(0, 0)

    assert strategy.can_short is True
    assert strategy.short_entry_conditions
    assert strategy.short_exit_conditions
    assert strategy.short_entry_conditions[0].operator == "cross_above"
    assert strategy.short_entry_conditions[0].threshold == 70
    assert strategy.short_exit_conditions[0].operator == "cross_below"
    assert strategy.short_exit_conditions[0].threshold == 30


def test_short_threshold_mutation_uses_opposite_direction_ranges():
    short_entry = ConditionGene(
        indicator="RSI", operator="cross_above", threshold=70
    )
    short_exit = ConditionGene(
        indicator="RSI", operator="cross_below", threshold=30
    )

    _mutate_condition_threshold(
        short_entry, {}, True, 0, [], side="short"
    )
    _mutate_condition_threshold(
        short_exit, {}, False, 0, [], side="short"
    )

    assert 60 <= short_entry.threshold <= 80
    assert 20 <= short_exit.threshold <= 40


def test_condition_factory_is_side_aware_for_short_rsi():
    entry = _create_random_condition("RSI", True, {}, side="short")
    exit_ = _create_random_condition("RSI", False, {}, side="short")

    assert entry.operator == "cross_above"
    assert 60 <= entry.threshold <= 80
    assert exit_.operator == "cross_below"
    assert 20 <= exit_.threshold <= 40
