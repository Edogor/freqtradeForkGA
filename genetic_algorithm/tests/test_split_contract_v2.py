"""Tests for explicit spatial/temporal/final evaluation split semantics."""

from __future__ import annotations

from datetime import date

import pytest

from genetic_algorithm.orchestration.promotion_policy_v2 import (
    ScenarioRequirementV2,
    ShadowGatePolicyV2,
)
from genetic_algorithm.orchestration.split_contract_v2 import (
    SplitContractError,
    build_evaluation_split_plan,
)


def _config(*, embargo_days: int = 0) -> dict:
    return {
        "backtesting": {
            "exchange": "binance",
            "pairs": ["BTC/USDT", "ETH/USDT"],
            # Freqtrade's end is exclusive: evolution covers all of 2024.
            "timerange": "20240101-20250101",
        },
        "split_v2": {"min_embargo_days": embargo_days},
    }


def _scenario(
    scenario_id: str,
    pair: str,
    role: str,
    start: date,
    end: date,
    *,
    timeframe: str = "1h",
) -> ScenarioRequirementV2:
    return ScenarioRequirementV2(
        scenario_id=scenario_id,
        pair=pair,
        timeframe=timeframe,
        role=role,
        period_start=start,
        period_end=end,
        cost_multiplier=1.0,
    )


def _policy(*scenarios: ScenarioRequirementV2) -> ShadowGatePolicyV2:
    return ShadowGatePolicyV2(
        policy_version="split-contract-test",
        required_scenarios=list(scenarios),
    )


def test_valid_plan_separates_pair_temporal_and_final_axes():
    policy = _policy(
        _scenario(
            "pair-sol",
            "SOL/USDT",
            "PAIR_VALIDATION",
            date(2024, 4, 1),
            date(2024, 6, 30),
        ),
        _scenario(
            "temporal-btc",
            "BTC/USDT",
            "TEMPORAL_VALIDATION",
            date(2025, 1, 1),
            date(2025, 3, 31),
        ),
        _scenario(
            "final-btc",
            "BTC/USDT",
            "FINAL_TEST",
            date(2025, 4, 1),
            date(2025, 6, 30),
        ),
        _scenario(
            "final-sol",
            "SOL/USDT",
            "FINAL_TEST",
            date(2025, 1, 1),
            date(2025, 3, 31),
        ),
    )

    plan = build_evaluation_split_plan(_config(), policy)

    assert plan.evolution_period_start == date(2024, 1, 1)
    assert plan.evolution_period_end_exclusive == date(2025, 1, 1)
    assert [item.scenario_id for item in plan.scenarios] == [
        "final-btc",
        "final-sol",
        "pair-sol",
        "temporal-btc",
    ]
    assert len(plan.split_hash) == 64


def test_ambiguous_inner_validation_is_rejected():
    policy = _policy(
        _scenario(
            "legacy-inner",
            "BTC/USDT",
            "INNER_VALIDATION",
            date(2025, 1, 1),
            date(2025, 3, 31),
        ),
        _scenario(
            "final",
            "ETH/USDT",
            "FINAL_TEST",
            date(2025, 1, 1),
            date(2025, 3, 31),
        ),
    )

    with pytest.raises(SplitContractError, match="ambiguous INNER_VALIDATION"):
        build_evaluation_split_plan(_config(), policy)


def test_pair_validation_must_use_unseen_pair():
    policy = _policy(
        _scenario(
            "bad-pair",
            "BTC/USDT",
            "PAIR_VALIDATION",
            date(2024, 4, 1),
            date(2024, 6, 30),
        ),
        _scenario(
            "final",
            "ETH/USDT",
            "FINAL_TEST",
            date(2025, 1, 1),
            date(2025, 3, 31),
        ),
    )

    with pytest.raises(SplitContractError, match="reuses an evolution pair"):
        build_evaluation_split_plan(_config(), policy)


def test_enabled_pair_split_excludes_validation_pairs_from_evolution():
    config = _config()
    config["pair_validation"] = {
        "enabled": True,
        "training_pairs": ["BTC/USDT"],
        "validation_pairs": ["ETH/USDT"],
    }
    policy = _policy(
        _scenario(
            "pair-eth",
            "ETH/USDT",
            "PAIR_VALIDATION",
            date(2024, 4, 1),
            date(2024, 6, 30),
        ),
        _scenario(
            "final-sol",
            "SOL/USDT",
            "FINAL_TEST",
            date(2025, 1, 1),
            date(2025, 3, 31),
        ),
    )

    plan = build_evaluation_split_plan(config, policy)

    assert plan.evolution_pairs == ["BTC/USDT"]
    assert next(item for item in plan.scenarios if item.scenario_id == "pair-eth").pair == (
        "ETH/USDT"
    )


def test_enabled_pair_split_must_partition_every_backtesting_pair():
    config = _config()
    config["pair_validation"] = {
        "enabled": True,
        "training_pairs": ["BTC/USDT"],
        "validation_pairs": ["SOL/USDT"],
    }

    with pytest.raises(
        SplitContractError,
        match="train/validation union must equal backtesting.pairs",
    ):
        build_evaluation_split_plan(
            config,
            _policy(
                _scenario(
                    "pair-sol",
                    "SOL/USDT",
                    "PAIR_VALIDATION",
                    date(2024, 4, 1),
                    date(2024, 6, 30),
                ),
                _scenario(
                    "final-xrp",
                    "XRP/USDT",
                    "FINAL_TEST",
                    date(2025, 1, 1),
                    date(2025, 3, 31),
                ),
            ),
        )


def test_temporal_validation_must_use_seen_pair_after_evolution():
    unseen_policy = _policy(
        _scenario(
            "bad-temporal",
            "SOL/USDT",
            "TEMPORAL_VALIDATION",
            date(2025, 1, 1),
            date(2025, 3, 31),
        ),
        _scenario(
            "final",
            "BTC/USDT",
            "FINAL_TEST",
            date(2025, 4, 1),
            date(2025, 6, 30),
        ),
    )
    with pytest.raises(SplitContractError, match="must use an evolution pair"):
        build_evaluation_split_plan(_config(), unseen_policy)

    overlapping_policy = _policy(
        _scenario(
            "bad-temporal",
            "BTC/USDT",
            "TEMPORAL_VALIDATION",
            date(2024, 10, 1),
            date(2025, 2, 1),
        ),
        _scenario(
            "final",
            "ETH/USDT",
            "FINAL_TEST",
            date(2025, 1, 1),
            date(2025, 3, 31),
        ),
    )
    with pytest.raises(SplitContractError, match="overlaps/precedes evolution"):
        build_evaluation_split_plan(_config(), overlapping_policy)


def test_final_cannot_precede_or_overlap_same_pair_evidence_even_on_other_timeframe():
    policy = _policy(
        _scenario(
            "final-resampled",
            "BTC/USDT",
            "FINAL_TEST",
            date(2024, 6, 1),
            date(2024, 9, 1),
            timeframe="4h",
        )
    )

    with pytest.raises(SplitContractError, match="not strictly after prior"):
        build_evaluation_split_plan(_config(), policy)


def test_final_on_completely_unseen_pair_may_be_spatial_holdout_in_same_period():
    policy = _policy(
        _scenario(
            "final-xrp",
            "XRP/USDT",
            "FINAL_TEST",
            date(2024, 6, 1),
            date(2024, 9, 1),
        )
    )

    plan = build_evaluation_split_plan(_config(), policy)

    assert plan.scenarios[0].role.value == "FINAL_TEST"


def test_embargo_applies_between_same_pair_stages():
    policy = _policy(
        _scenario(
            "temporal-btc",
            "BTC/USDT",
            "TEMPORAL_VALIDATION",
            date(2025, 1, 5),
            date(2025, 3, 31),
        ),
        _scenario(
            "final-btc",
            "BTC/USDT",
            "FINAL_TEST",
            date(2025, 4, 5),
            date(2025, 6, 30),
        ),
    )

    with pytest.raises(SplitContractError, match="evolution or violates embargo"):
        build_evaluation_split_plan(_config(embargo_days=5), policy)
