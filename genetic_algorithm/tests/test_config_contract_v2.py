"""Versioned, fail-closed configuration contract regressions."""

from __future__ import annotations

import copy

import pytest
import yaml

from genetic_algorithm import cli
from genetic_algorithm.config import schema as config_schema
from genetic_algorithm.config.schema import (
    DEFAULTS,
    load_config,
    resolve_config,
    resolve_preset,
    validate_resolved_config_v2_or_raise,
)
from genetic_algorithm.engine.nsga2 import extract_objectives_from_metrics
from genetic_algorithm.core.evolution import GeneticAlgorithm
from genetic_algorithm.run_ga import load_and_update_config


def _write(tmp_path, payload, name: str = "config.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("override", "expected_path"),
    [
        ({"fitnes_weights": {}}, "fitnes_weights"),
        ({"backtesting": {"slipage_pct": 0.001}}, "backtesting.slipage_pct"),
        ({"pair_validation": {"enabld": False}}, "pair_validation.enabld"),
        ({"advanced": {"llm": {"enabld": False}}}, "advanced.llm.enabld"),
    ],
)
def test_schema_v2_rejects_unknown_paths_at_every_depth(tmp_path, override, expected_path):
    path = _write(tmp_path, {"preset": "safe_v2", **override})

    resolution = resolve_config(path)

    assert any(expected_path in error for error in resolution.errors)
    with pytest.raises(ValueError, match=expected_path.replace("[", r"\[")):
        load_config(path)


def test_unknown_paths_are_sorted_and_reported_together(tmp_path):
    path = _write(
        tmp_path,
        {"preset": "safe_v2", "z_unknown": 1, "a_unknown": 2},
    )

    errors = resolve_config(path).errors

    unknown_error = next(error for error in errors if error.startswith("unknown config paths"))
    assert unknown_error.endswith("a_unknown, z_unknown")


def test_unknown_key_inside_list_item_is_rejected(tmp_path):
    path = _write(
        tmp_path,
        {
            "preset": "safe_v2",
            "nsga2": {
                "objectives": [
                    {"name": "profit", "type": "maximize", "scale": 100.0, "typo": True}
                ]
            },
        },
    )

    assert any(
        "nsga2.objectives[0].typo" in error for error in resolve_config(path).errors
    )

    scenario_path = _write(
        tmp_path,
        {
            "preset": "safe_v2",
            "promotion_v2": {
                "required_scenarios": [
                    {
                        "scenario_id": "btc-temporal",
                        "pair": "BTC/USDT",
                        "timeframe": "1h",
                        "role": "TEMPORAL_VALIDATION",
                        "period_start": "2025-01-01",
                        "period_end": "2025-02-01",
                        "cost_multiplier": 1.0,
                        "typo": True,
                    }
                ]
            },
        },
        name="scenario.yaml",
    )
    assert any(
        "promotion_v2.required_scenarios[0].typo" in error
        for error in resolve_config(scenario_path).errors
    )


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        ({"backtesting": {"fee": {"value": 0.001}}}, "backtesting.fee must be float"),
        ({"genetic_algorithm": {"population_size": True}}, "population_size must be int"),
        ({"evaluation_v2": {"enabled": "false"}}, "evaluation_v2.enabled must be bool"),
        ({"parallel_evaluation": {"num_workers": "4"}}, "num_workers must be int or null"),
    ],
)
def test_schema_v2_uses_strict_types(tmp_path, override, expected):
    path = _write(tmp_path, {"preset": "safe_v2", **override})

    assert any(expected in error for error in resolve_config(path).errors)


def test_programmatic_overrides_use_the_same_unknown_key_contract(tmp_path):
    path = _write(tmp_path, {"preset": "safe_v2"})

    resolution = resolve_config(path, overrides={"backtesting": {"slipage_pct": 0.1}})

    assert any("backtesting.slipage_pct" in error for error in resolution.errors)


def test_schema_v1_keeps_legacy_unknown_keys_but_emits_migration_warning(tmp_path):
    path = _write(tmp_path, {"legacy_extension": {"enabled": True}})

    resolution = resolve_config(path)

    assert resolution.errors == ()
    assert any("schema v1 accepts undeclared" in warning for warning in resolution.warnings)
    assert load_config(path)["legacy_extension"]["enabled"] is True


def test_v2_artifact_boundary_requires_version_and_fully_resolved_defaults():
    config = load_config("genetic_algorithm/config/presets/safe_v2.yaml")
    validate_resolved_config_v2_or_raise(config)
    assert config["evaluation_v2"]["expectancy_cluster_days"] == 1

    legacy = copy.deepcopy(config)
    legacy["config_schema_version"] = 1
    with pytest.raises(ValueError, match="config_schema_version: 2"):
        validate_resolved_config_v2_or_raise(legacy)

    incomplete = copy.deepcopy(config)
    del incomplete["parallel_evaluation"]["backtest_timeout"]
    with pytest.raises(ValueError, match=r"parallel_evaluation\.backtest_timeout"):
        validate_resolved_config_v2_or_raise(incomplete)


def test_automation_island_v2_preset_is_a_valid_shadow_search_contract():
    config = load_config(
        "genetic_algorithm/config/presets/automation_island_v2.yaml"
    )

    validate_resolved_config_v2_or_raise(config)
    assert config["safety_profile"] == {
        "name": "automation_island_v2",
        "enforce": True,
        "shadow_mode": True,
        "automation_eligible": False,
    }
    assert config["generic_island_model"]["enabled"] is True
    assert config["pair_validation"]["validate_top_n_only"] == 0
    assert config["fitness_penalties"]["target_trades_per_pair"] == 0
    assert config["fitness_penalties"]["target_trades_per_active_month"] == 5.0
    assert config["promotion_v2"]["policy_version"] == "automation-island-v2.1-shadow"
    assert config["fitness_weights"]["profit"] == 0.34
    assert config["fitness_weights"]["profit_factor"] == 0.20
    assert config["fitness_weights"]["trade_frequency"] == 0.06
    assert config["fitness_bounds"]["profit_factor_break_even_normalization"] is True
    assert config["fitness_penalties"]["pair_trade_coverage_exponent"] == 0.5
    assert config["fitness_weights"]["drawdown_duration"] == 0.11
    assert sum(config["fitness_weights"].values()) == pytest.approx(1.0)
    assert config["fitness_bounds"]["profit_min"] == -5.0
    assert config["fitness_bounds"]["profit_max"] == 20.0
    assert config["fitness_bounds"]["drawdown_normalization_target"] == 0.25
    assert config["pair_validation"]["weight_train"] == 0.5
    assert config["pair_validation"]["weight_val"] == 0.5
    assert config["pair_validation"]["worst_split_weight"] == 0.25
    assert config["strategy_constraints"]["stoploss_range"] == [-0.12, -0.03]
    assert config["strategy_constraints"]["max_open_trades_range"] == [3, 3]
    assert config["output"]["top_n"] == 10


def test_new_search_alignment_fields_remain_optional_for_frozen_v2_configs():
    config = load_config(
        "genetic_algorithm/config/presets/automation_island_v2.yaml"
    )
    del config["genetic_algorithm"]["search_seed_salt"]
    del config["fitness_weights"]["drawdown_duration"]
    del config["fitness_weights"]["consecutive_losses"]
    del config["fitness_penalties"]["target_trades_per_active_month"]

    validate_resolved_config_v2_or_raise(config)


@pytest.mark.parametrize("invalid_value", [0, -1, False, float("nan")])
def test_active_month_trade_target_requires_a_positive_finite_number(
    tmp_path,
    invalid_value,
):
    path = _write(
        tmp_path,
        {
            "preset": "automation_island_v2",
            "fitness_penalties": {
                "target_trades_per_active_month": invalid_value,
            },
        },
    )

    assert any(
        "fitness_penalties.target_trades_per_active_month must be a finite "
        "number > 0" in error
        for error in resolve_config(path).errors
    )


@pytest.mark.parametrize(
    ("override", "expected_error"),
    [
        (
            {"safety_profile": {"automation_eligible": True}},
            "forbids strategy automation_eligible",
        ),
        (
            {"safety_profile": {"shadow_mode": False}},
            "requires shadow_mode: true",
        ),
        (
            {"generic_island_model": {"enabled": False}},
            "requires generic_island_model.enabled",
        ),
        (
            {"pair_validation": {"validate_top_n_only": 1}},
            "requires pair_validation.validate_top_n_only: 0",
        ),
        (
            {"walk_forward": {"enabled": True}},
            "forbids walk_forward.enabled: true",
        ),
        (
            {"sis": {"enabled": True}},
            "forbids sis.enabled: true",
        ),
        (
            {"advanced": {"llm": {"enabled": True}}},
            "forbids advanced.llm.enabled: true",
        ),
        (
            {"surrogate": {"enabled": True}},
            "forbids surrogate.enabled: true",
        ),
        (
            {"genetic_algorithm": {"max_runtime_minutes": None}},
            "requires genetic_algorithm.max_runtime_minutes",
        ),
        (
            {"genetic_algorithm": {"max_runtime_minutes": 0}},
            "max_runtime_minutes must be a positive integer",
        ),
    ],
)
def test_automation_island_v2_rejects_unsafe_or_deferred_features(
    tmp_path,
    override,
    expected_error,
):
    path = _write(tmp_path, {"preset": "automation_island_v2", **override})

    assert any(expected_error in error for error in resolve_config(path).errors)


def test_automation_island_child_v2_rejects_recursive_island_engine(tmp_path):
    config = load_config(
        "genetic_algorithm/config/presets/automation_island_v2.yaml"
    )
    config["safety_profile"]["name"] = "automation_island_child_v2"
    config["generic_island_model"]["enabled"] = False
    path = _write(tmp_path, config)

    assert resolve_config(path).errors == ()

    config["generic_island_model"]["enabled"] = True
    path = _write(tmp_path, config, name="recursive-child.yaml")
    assert any(
        "forbids recursive generic_island_model.enabled" in error
        for error in resolve_config(path).errors
    )


@pytest.mark.parametrize("invalid_value", [0, -1, False, 1.5])
def test_expectancy_cluster_days_requires_a_positive_integer(
    tmp_path,
    invalid_value,
):
    path = _write(
        tmp_path,
        {
            "preset": "safe_v2",
            "evaluation_v2": {"expectancy_cluster_days": invalid_value},
        },
    )

    assert any(
        "evaluation_v2.expectancy_cluster_days must be a positive integer" in error
        for error in resolve_config(path).errors
    )


def test_deprecated_key_has_actionable_migration_message(tmp_path):
    path = _write(
        tmp_path,
        {"preset": "safe_v2", "monte_carlo": {"enabled": False, "iterations": 50}},
    )

    errors = resolve_config(path).errors

    assert any(
        "monte_carlo.iterations; use monte_carlo.num_permutations" in error
        for error in errors
    )


def test_legacy_surrogate_model_path_has_actionable_migration_message(tmp_path):
    path = _write(tmp_path, {"surrogate_model": {"enabled": True}})

    resolution = resolve_config(path)

    assert any(
        "deprecated config path surrogate_model; use surrogate" in warning
        for warning in resolution.warnings
    )


def test_multi_objective_mode_points_to_supported_nsga2_contract(tmp_path):
    path = _write(
        tmp_path,
        {"genetic_algorithm": {"mode": "multi_objective"}},
    )

    assert any(
        "use genetic_algorithm.mode: nsga2 and configure nsga2.objectives" in error
        for error in resolve_config(path).errors
    )


def test_nsga2_rejects_historically_ignored_objective_keys(tmp_path):
    path = _write(
        tmp_path,
        {
            "genetic_algorithm": {"mode": "nsga2"},
            "nsga2": {
                "objectives": [
                    {
                        "name": "profit",
                        "direction": "maximize",
                        "weight": 1.0,
                    }
                ]
            },
        },
    )

    assert any(
        "uses ignored direction/weight keys; use type and scale" in error
        for error in resolve_config(path).errors
    )


@pytest.mark.parametrize(
    "path",
    [
        "genetic_algorithm/config/benchmark/run6_nsga2_multiobjective.yaml",
        "genetic_algorithm/config/benchmark_v2/run6_nsga2.yaml",
    ],
)
def test_nsga2_benchmark_objectives_use_runtime_keys(path):
    resolution = resolve_config(path)

    assert resolution.errors == ()
    for objective in resolution.config["nsga2"]["objectives"]:
        assert "type" in objective
        assert "scale" in objective
        assert "direction" not in objective
        assert "weight" not in objective


def test_nsga2_trade_frequency_objective_uses_num_trades_metric():
    objectives = extract_objectives_from_metrics(
        {"num_trades": 40},
        [
            {
                "name": "trade_frequency",
                "type": "goldilocks",
                "target": 50,
                "tolerance": 20,
            }
        ],
    )

    assert objectives == pytest.approx([0.5])


def test_v2_contract_declares_active_short_surrogate_and_nsga_fields(tmp_path):
    path = _write(
        tmp_path,
        {
            "preset": "safe_v2",
            "short_selling": {
                "enabled": False,
                "probability": 0.4,
                "independent_conditions": True,
            },
            "surrogate": {
                "enabled": False,
                "retrain_interval": 2,
                "validation_fraction": 0.25,
                "min_validation_r2": 0.35,
                "min_validation_samples": 12,
                "model": "gradient_boosting",
                "mutate_skipped": False,
                "adaptive_filter": {
                    "enabled": False,
                    "min_percentile": 30,
                    "r2_threshold": 0.4,
                },
            },
            "nsga2": {
                "min_trades": 20,
                "crowding_distance_percentile": 0.2,
            },
        },
    )

    assert not any("unknown config paths" in error for error in resolve_config(path).errors)


def test_v2_contract_validates_independent_pair_search_controls(tmp_path):
    path = _write(
        tmp_path,
        {
            "config_schema_version": 2,
            "pair_validation": {
                "enabled": True,
                "evaluation_mode": "portfolio_magic",
                "worst_pair_weight": 1.1,
                "min_profitable_pair_ratio": -0.1,
                "profitable_pair_penalty_floor": 0.1,
                "max_pair_loss_pct": -5.0,
                "worst_pair_loss_penalty_floor": 0.1,
            },
        },
    )

    errors = resolve_config(path).errors
    assert any("evaluation_mode" in error for error in errors)
    assert any("worst_pair_weight" in error for error in errors)
    assert any("min_profitable_pair_ratio" in error for error in errors)
    assert any("max_pair_loss_pct" in error for error in errors)


@pytest.mark.parametrize(
    ("override", "expected_error"),
    [
        ({"min_training_samples": 19}, "min_training_samples"),
        ({"retrain_interval": 0}, "retrain_interval"),
        ({"min_validation_samples": 4}, "min_validation_samples"),
        ({"filter_percentile": 0}, "filter_percentile"),
        ({"validation_fraction": 0.09}, "validation_fraction"),
        ({"min_validation_r2": 1.01}, "min_validation_r2"),
        ({"model": "linear_regression"}, "surrogate.model"),
    ],
)
def test_v2_contract_rejects_invalid_surrogate_quality_gate_values(
    tmp_path, override, expected_error
):
    path = _write(
        tmp_path,
        {
            "config_schema_version": 2,
            "surrogate": {"enabled": False, **override},
        },
    )

    assert any(expected_error in error for error in resolve_config(path).errors)


def test_v2_contract_declares_regime_coverage_settings(tmp_path):
    path = _write(
        tmp_path,
        {
            "config_schema_version": 2,
            "regime_aware": {
                "enabled": False,
                "aggregation": "cvar",
                "cvar_alpha": 0.25,
                "confidence_weighting": True,
                "min_segment_trades": 5,
                "regime_weights": {
                    "bullish": 1.0,
                    "bearish": 1.2,
                    "sideways": 1.0,
                    "volatile": 1.1,
                    "uncertain": 0.8,
                },
                "regime_specialization": {
                    "enabled": False,
                    "specialist_boost": 1.5,
                    "diversity_weight": 0.05,
                    "initial_regime_ratio": 0.3,
                },
            },
        },
    )

    resolution = resolve_config(path)

    assert resolution.errors == ()


@pytest.mark.parametrize(
    ("regime_override", "expected_error"),
    [
        ({"aggregation": "median"}, "regime_aware.aggregation"),
        ({"min_segment_trades": -1}, "min_segment_trades"),
        ({"cvar_alpha": 0.0}, "cvar_alpha"),
        ({"regime_weights": {"bullish": 0.0}}, "regime_weights.bullish"),
        (
            {"regime_specialization": {"specialist_boost": 0.0}},
            "specialist_boost",
        ),
    ],
)
def test_v2_contract_rejects_unsafe_regime_aggregation_values(
    tmp_path, regime_override, expected_error
):
    path = _write(
        tmp_path,
        {
            "config_schema_version": 2,
            "regime_aware": regime_override,
        },
    )

    assert any(expected_error in error for error in resolve_config(path).errors)


def test_active_defaults_match_runtime_key_names():
    assert DEFAULTS["monte_carlo"]["num_permutations"] == 100
    assert "iterations" not in DEFAULTS["monte_carlo"]
    assert DEFAULTS["surrogate"]["min_training_samples"] == 50
    assert DEFAULTS["surrogate"]["filter_percentile"] == 60
    assert DEFAULTS["generic_island_model"]["migration"]["count"] == 2
    assert DEFAULTS["regime_aware"]["method"] == "adx_di_hysteresis"


def test_cli_runner_and_standard_engine_share_the_fatal_contract(tmp_path, capsys):
    path = _write(tmp_path, {"preset": "safe_v2", "backtesting": {"slipage_pct": 0.1}})

    with pytest.raises(ValueError) as direct:
        load_config(path)
    with pytest.raises(ValueError) as engine:
        GeneticAlgorithm._load_config(object(), str(path))
    with pytest.raises(ValueError) as legacy_runner:
        load_and_update_config(path)

    assert str(engine.value) == str(direct.value) == str(legacy_runner.value)
    assert cli.main(["config", "validate", str(path)]) == 1
    assert "backtesting.slipage_pct" in capsys.readouterr().out


def test_non_mapping_yaml_root_is_rejected(tmp_path):
    path = _write(tmp_path, ["not", "a", "mapping"])

    with pytest.raises(ValueError, match="root must be a YAML mapping"):
        load_config(path)


def test_preset_resolution_is_detached_and_idempotent():
    raw = {"preset": "safe_v2", "backtesting": {"pairs": ["BTC/USDT"]}}
    before = copy.deepcopy(raw)

    first = resolve_preset(raw)
    second = resolve_preset(raw)

    assert raw == before
    assert first == second


def test_nested_preset_resolution_and_cycle_rejection(tmp_path, monkeypatch):
    monkeypatch.setattr(config_schema, "_PRESETS_DIR", tmp_path)
    _write(
        tmp_path,
        {"backtesting": {"pairs": ["BTC/USDT"], "fee": 0.002}},
        name="base.yaml",
    )
    _write(
        tmp_path,
        {"preset": "base", "backtesting": {"fee": 0.001}},
        name="child.yaml",
    )

    resolved = resolve_preset(
        {"preset": "child", "backtesting": {"pairs": ["ETH/USDT"]}}
    )

    assert resolved["backtesting"] == {
        "pairs": ["ETH/USDT"],
        "fee": 0.001,
    }

    _write(tmp_path, {"preset": "cycle_b"}, name="cycle_a.yaml")
    _write(tmp_path, {"preset": "cycle_a"}, name="cycle_b.yaml")
    with pytest.raises(
        ValueError,
        match="Recursive preset inheritance: cycle_a -> cycle_b -> cycle_a",
    ):
        resolve_preset({"preset": "cycle_a"})


def test_preset_name_cannot_escape_preset_directory(tmp_path):
    path = _write(tmp_path, {"preset": "../../outside"})

    with pytest.raises(ValueError, match="Invalid preset name"):
        load_config(path)
