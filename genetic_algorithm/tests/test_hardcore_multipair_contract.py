"""Fail-closed contract tests for the hardcore six-pair search lanes."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pandas as pd
import pytest

from genetic_algorithm.config.schema import (
    load_config,
    validate_config,
    validate_resolved_config_v2_or_raise,
)
from genetic_algorithm.core.generic_island_model import (
    GenericIslandConfig,
    GenericIslandModelEvolution,
)
from genetic_algorithm.orchestration.evolution_worker_v2 import (
    _evolution_worker_kind,
    _validate_evolution_contract,
)
from genetic_algorithm.orchestration.promotion_policy_v2 import (
    shadow_gate_policy_from_config,
)
from genetic_algorithm.orchestration.runner_v2 import _validate_supported_config
from genetic_algorithm.orchestration.split_contract_v2 import (
    build_evaluation_split_plan,
)
from genetic_algorithm.evaluation.direct_backtester import DirectBacktester
from genetic_algorithm.genome.codegen import StrategyGenerator
from genetic_algorithm.core.strategy_gene import (
    ConditionGene,
    IndicatorGene,
    StrategyGene,
)


PRESET_DIR = "genetic_algorithm/config/presets"
PRODUCTION_PRESETS = (
    f"{PRESET_DIR}/hardcore_multipair_15m_v1.yaml",
    f"{PRESET_DIR}/hardcore_multipair_1h_v1.yaml",
)
CANARY_PRESETS = (
    f"{PRESET_DIR}/hardcore_multipair_canary_15m_v1.yaml",
    f"{PRESET_DIR}/hardcore_multipair_canary_1h_v1.yaml",
)
EXPECTED_PAIRS = {
    "BTC/USDT",
    "SOL/USDT",
    "XRP/USDT",
    "BNB/USDT",
    "ETH/USDT",
    "PEPE/USDT",
}
EXPECTED_TIMERANGES = {
    "15m": "1683590400-1774568700",
    "1h": "1683590400-1774566000",
}


def _contract_errors(config: dict) -> list[str]:
    errors, _warnings = validate_config(config)
    return errors


@pytest.mark.parametrize("preset_path", PRODUCTION_PRESETS)
def test_production_preset_is_exact_twelve_island_six_pair_contract(preset_path):
    config = load_config(preset_path)

    validate_resolved_config_v2_or_raise(config)
    islands = config["generic_island_model"]["islands"]

    assert config["safety_profile"]["name"] == "hardcore_multipair_v1"
    assert config["safety_profile"]["canary"] is False
    assert config["generic_island_model"]["num_islands"] == 12
    assert len(islands) == 12
    assert set(config["backtesting"]["pairs"]) == EXPECTED_PAIRS
    assert config["backtesting"]["timerange"] == EXPECTED_TIMERANGES[
        config["backtesting"]["timeframe"]
    ]
    assert all(set(island["pairs"]) == EXPECTED_PAIRS for island in islands)
    assert config["pair_validation"]["validate_top_n_only"] == 0
    assert config["genetic_algorithm"]["seed_known_archetypes"] is False
    assert config["raw_multipair_score"]["policy_version"] == "raw-multipair-score-v2"
    assert config["strategy_constraints"]["canonicalize_executable_genome"] is True


def test_hardcore_profile_rejects_built_in_archetype_seeding():
    config = load_config(PRODUCTION_PRESETS[0])
    config["genetic_algorithm"]["seed_known_archetypes"] = True

    errors = _contract_errors(config)

    assert any(
        "requires fresh islands without built-in archetype seeds" in error
        for error in errors
    ), errors


@pytest.mark.parametrize(
    ("mutate", "expected_error"),
    [
        (
            lambda config: config["generic_island_model"]["islands"][0].__setitem__(
                "walk_forward_enabled", True
            ),
            "generic_island_model.islands[0].walk_forward_enabled",
        ),
        (
            lambda config: config["generic_island_model"]["walk_forward"].__setitem__(
                "enabled", True
            ),
            "generic_island_model.walk_forward",
        ),
    ],
)
def test_hardcore_profile_recursively_rejects_walk_forward(mutate, expected_error):
    config = load_config(PRODUCTION_PRESETS[0])
    mutate(config)

    errors = _contract_errors(config)

    assert any(expected_error in error for error in errors), errors


def test_hardcore_profile_requires_raw_multipair_score():
    config = load_config(PRODUCTION_PRESETS[0])
    config["raw_multipair_score"]["enabled"] = False

    errors = _contract_errors(config)

    assert any(
        "requires raw_multipair_score.enabled: true" in error for error in errors
    )


def test_hardcore_profile_requires_executable_genome_canonicalization():
    config = load_config(PRODUCTION_PRESETS[0])
    config["strategy_constraints"]["canonicalize_executable_genome"] = False

    errors = _contract_errors(config)

    assert any("requires executable-genome canonicalization" in error for error in errors)


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        (
            lambda config: config["backtesting"].__setitem__(
                "timerange", "20230509-20260326"
            ),
            "exact inclusive 2023-05-09 through 2026-03-26",
        ),
        (
            lambda config: config["promotion_v2"]["required_scenarios"][0].__setitem__(
                "period_end", "2026-03-25"
            ),
            "differs from the fixed raw panel",
        ),
        (
            lambda config: config["generic_island_model"]["islands"][0].__setitem__(
                "qualification_v3", {"enabled": True}
            ),
            "recursively forbids enabled feature",
        ),
        (
            lambda config: config["nsga2"].__setitem__("enabled", True),
            "forbids nsga2.enabled: true",
        ),
    ],
)
def test_hardcore_profile_rejects_panel_drift_and_nested_robustness(
    mutation, expected_error
):
    config = load_config(PRODUCTION_PRESETS[0])
    mutation(config)

    errors = _contract_errors(config)

    assert any(expected_error in error for error in errors), errors


@pytest.mark.parametrize("break_island_contract", ["declared-count", "explicit-list"])
def test_hardcore_profile_rejects_wrong_island_count(break_island_contract):
    config = load_config(PRODUCTION_PRESETS[0])
    if break_island_contract == "declared-count":
        config["generic_island_model"]["num_islands"] = 11
    else:
        config["generic_island_model"]["islands"].pop()

    errors = _contract_errors(config)

    assert any(
        "requires generic_island_model.num_islands: 12" in error
        or "requires exactly twelve explicit islands" in error
        for error in errors
    ), errors


@pytest.mark.parametrize("preset_path", PRODUCTION_PRESETS)
def test_runner_and_worker_accept_hardcore_profile_without_market_data(preset_path):
    config = copy.deepcopy(load_config(preset_path))
    policy = shadow_gate_policy_from_config(config)
    split_plan = build_evaluation_split_plan(config, policy)
    timeframe = config["backtesting"]["timeframe"]
    manifest = SimpleNamespace(
        seeds=[config["genetic_algorithm"]["random_seed"]],
        split_manifest_hash=split_plan.split_hash,
    )
    data_manifest = SimpleNamespace(
        files=[
            SimpleNamespace(pair=pair, timeframe=timeframe)
            for pair in config["backtesting"]["pairs"]
        ]
    )

    _validate_supported_config(config)
    assert _evolution_worker_kind(config) == "GENERIC_ISLAND_EVOLUTION"
    _validate_evolution_contract(manifest, config, policy, data_manifest)


@pytest.mark.parametrize("preset_path", (*PRODUCTION_PRESETS, *CANARY_PRESETS))
def test_every_materialized_hardcore_subisland_passes_child_safety_contract(
    preset_path,
):
    config = load_config(preset_path)
    coordinator = object.__new__(GenericIslandModelEvolution)
    coordinator.config = config
    coordinator.generations = config["generic_island_model"]["generations"]

    for raw in config["generic_island_model"]["islands"]:
        island = GenericIslandConfig(
            name=raw["name"],
            population_size=raw["population_size"],
            generations=raw["generations"],
            seed=raw["seed"],
            indicator_pool=raw["indicator_pool"],
            pairs=raw["pairs"],
            walk_forward_enabled=raw["walk_forward_enabled"],
        )
        child = coordinator._build_island_config(island)

        validate_resolved_config_v2_or_raise(child)
        assert child["safety_profile"]["name"] == "hardcore_multipair_child_v1"
        assert child["generic_island_model"]["enabled"] is False
        assert child["parallel_evaluation"]["enabled"] is False
        assert child["raw_multipair_score"]["enabled"] is True
        assert set(child["backtesting"]["pairs"]) == EXPECTED_PAIRS


def test_shared_cache_slices_max_warmup_to_exact_strategy_warmup():
    from freqtrade.configuration import TimeRange

    dates = pd.date_range(
        "2023-05-05 18:00:00+00:00",
        "2023-05-10 00:00:00+00:00",
        freq="1h",
    )
    frame = pd.DataFrame({"date": dates, "close": range(len(dates))})
    timerange = TimeRange.parse_timerange("1683590400-1774566000")

    sliced = DirectBacktester._slice_cached_startup_data(
        {"PEPE/USDT": frame},
        timerange,
        required_startup=75,
    )

    assert sliced is not None
    pepe = sliced["PEPE/USDT"]
    assert (pepe["date"] < pd.Timestamp("2023-05-09T00:00:00Z")).sum() == 75
    assert pepe.iloc[75]["date"] == pd.Timestamp("2023-05-09T00:00:00Z")


def test_hardcore_codegen_cap_matches_maximum_allowed_indicator_lookback():
    config = load_config(PRODUCTION_PRESETS[1])
    generator = StrategyGenerator(config)
    gene = StrategyGene(
        generation=0,
        individual_id=0,
        indicators=[
            IndicatorGene(
                type="MACD",
                parameters={
                    "fast_period": 21,
                    "slow_period": 50,
                    "signal_period": 14,
                },
            )
        ],
        entry_conditions=[
            ConditionGene(indicator="MACD", operator=">", threshold=0.0)
        ],
        exit_conditions=[
            ConditionGene(indicator="MACD", operator="<", threshold=0.0)
        ],
        timeframe="1h",
    )

    assert generator._compute_startup_candle_count(gene) == 75

    gene.indicators[0].parameters["slow_period"] = 60
    with pytest.raises(ValueError, match="above configured cap 75"):
        generator._compute_startup_candle_count(gene)
