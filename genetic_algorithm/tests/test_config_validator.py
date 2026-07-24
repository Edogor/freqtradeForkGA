"""Mechanical GA config invariants and anti-heuristic regressions."""

from __future__ import annotations

import copy
import re
import runpy
from pathlib import Path

import pytest
import yaml

from genetic_algorithm.config.invariants import (
    CONFIG_INVARIANT_POLICY_VERSION,
    derive_island_population_slots,
    validate_runtime_invariants,
)
from genetic_algorithm.config.schema import (
    DEFAULTS,
    deep_merge,
    resolve_config_data,
    resolve_preset,
)
from genetic_algorithm.utils.config_validator import (
    preflight_check,
    validate_and_log,
    validate_ga_config,
)


def _base_config() -> dict:
    config = copy.deepcopy(DEFAULTS)
    config["backtesting"]["timerange"] = "20230101-20260101"
    return config


def _messages(config: dict) -> tuple[list[str], list[str]]:
    return validate_ga_config(config)


def _has(messages: list[str], text: str) -> bool:
    return any(text in message for message in messages)


def test_policy_is_explicitly_versioned():
    assert CONFIG_INVARIANT_POLICY_VERSION == "ga-config-invariants-v1"


@pytest.mark.parametrize(
    ("population", "expected"),
    [(2, (1, 1)), (3, (2, 1)), (10, (2, 2)), (100, (10, 10))],
)
def test_derived_island_slots_are_always_feasible(population, expected):
    elite, immigrants = derive_island_population_slots(population)

    assert (elite, immigrants) == expected
    assert 0 <= elite < population
    assert 0 <= immigrants <= population - elite


def test_default_runtime_config_satisfies_mechanical_contract():
    errors, _ = _messages(_base_config())

    assert errors == []
    assert validate_runtime_invariants(_base_config()) == _messages(_base_config())


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("population_size", 1, "population_size"),
        ("population_size", 10_001, "population_size"),
        ("generations", 0, "generations"),
        ("generations", 100_001, "generations"),
        ("elite_size", -1, "elite_size"),
        ("tournament_size", 0, "tournament_size"),
        ("random_immigrants", -1, "random_immigrants"),
        ("convergence_patience", 0, "convergence_patience"),
    ],
)
def test_integer_runtime_bounds_are_mechanical(field, value, expected):
    config = _base_config()
    config["genetic_algorithm"][field] = value

    errors, _ = _messages(config)

    assert _has(errors, expected)


def test_population_slot_relationships_are_enforced():
    config = _base_config()
    config["genetic_algorithm"].update(
        {
            "population_size": 10,
            "elite_size": 10,
            "tournament_size": 11,
            "random_immigrants": 11,
        }
    )

    errors, _ = _messages(config)

    assert _has(errors, "elite_size must be smaller")
    assert _has(errors, "tournament_size must not exceed")
    assert _has(errors, "random_immigrants exceeds")


def test_immigrants_must_fit_non_elite_slots():
    config = _base_config()
    config["genetic_algorithm"].update(
        {"population_size": 10, "elite_size": 4, "random_immigrants": 7}
    )

    errors, _ = _messages(config)

    assert _has(errors, "non-elite population slots")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("mutation_rate", -0.01),
        ("mutation_rate", 1.01),
        ("max_mutation_rate", -0.01),
        ("max_mutation_rate", 1.01),
        ("crossover_rate", -0.01),
        ("crossover_rate", 1.01),
        ("sharing_radius", -0.01),
        ("diversity_threshold", 1.01),
        ("behavioral_distance_weight", 1.01),
        ("mutation_cooldown_factor", 1.01),
    ],
)
def test_probability_like_values_are_bounded(field, value):
    config = _base_config()
    config["genetic_algorithm"][field] = value

    errors, _ = _messages(config)

    assert _has(errors, field)


def test_base_mutation_cannot_exceed_adaptive_ceiling():
    config = _base_config()
    config["genetic_algorithm"].update(
        {"mutation_rate": 0.4, "max_mutation_rate": 0.3}
    )

    errors, _ = _messages(config)

    assert _has(errors, "mutation_rate must not exceed")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("mode", "multi_objective"),
        ("selection_method", "best_only"),
        ("crossover_method", "invented"),
    ],
)
def test_runtime_dispatch_values_must_exist(field, value):
    config = _base_config()
    config["genetic_algorithm"][field] = value

    errors, _ = _messages(config)

    assert _has(errors, field)


def test_tournament_size_one_is_valid_but_explicitly_random():
    config = _base_config()
    config["genetic_algorithm"]["tournament_size"] = 1

    errors, warnings = _messages(config)

    assert errors == []
    assert _has(warnings, "uniform random")


def test_nsga2_reports_ignored_legacy_selection_without_false_failure():
    config = _base_config()
    config["genetic_algorithm"].update(
        {"mode": "nsga2", "selection_method": "tournament"}
    )

    errors, warnings = _messages(config)

    assert errors == []
    assert _has(warnings, "engine forces nsga2 selection")


@pytest.mark.parametrize(
    ("pairs", "expected"),
    [
        ([], "non-empty list"),
        (["BTC/USDT", "BTC/USDT"], "duplicate pairs"),
        (["BTC/USDT", ""], "non-empty strings"),
    ],
)
def test_pair_list_has_only_structural_constraints(pairs, expected):
    config = _base_config()
    config["backtesting"]["pairs"] = pairs

    errors, _ = _messages(config)

    assert _has(errors, expected)


def test_many_pairs_and_large_population_are_not_declared_unsafe():
    config = _base_config()
    config["genetic_algorithm"].update(
        {
            "population_size": 100,
            "elite_size": 10,
            "tournament_size": 8,
            "random_immigrants": 15,
        }
    )
    config["backtesting"]["pairs"] = [
        "BTC/USDT",
        "ETH/USDT",
        "BNB/USDT",
        "SOL/USDT",
        "XRP/USDT",
        "ADA/USDT",
    ]

    errors, warnings = _messages(config)

    assert errors == []
    assert not _has(warnings, "safe ceiling")
    assert not _has(warnings, "ANTI-PATTERN")
    assert not _has(warnings, "generaliz")


@pytest.mark.parametrize(
    "timerange",
    [
        "not-a-date",
        "20240230-20240301",
        "20250101-20240101",
        "20250101-20250101",
    ],
)
def test_timerange_must_be_parseable_and_forward(timerange):
    config = _base_config()
    config["backtesting"]["timerange"] = timerange

    errors, _ = _messages(config)

    assert _has(errors, "timerange")


def test_empty_timerange_is_valid_with_factual_warning():
    config = _base_config()
    config["backtesting"]["timerange"] = ""

    errors, warnings = _messages(config)

    assert errors == []
    assert _has(warnings, "all available data")


def test_fitness_weights_are_normalized_not_forced_to_sum_to_one():
    config = _base_config()
    config["fitness_weights"] = {"profit": 3.0, "drawdown": 2.0}

    errors, warnings = _messages(config)

    assert errors == []
    assert not _has(warnings, "sum")


@pytest.mark.parametrize(
    "weights",
    [
        {"profit": -0.1},
        {"profit": float("nan")},
        {"profit": 0.0, "drawdown": 0.0},
        {"not_a_metric": 1.0},
    ],
)
def test_fitness_weights_must_be_finite_nonnegative_and_effective(weights):
    config = _base_config()
    config["fitness_weights"] = weights

    errors, _ = _messages(config)

    assert errors


def test_pair_validation_requires_disjoint_nonempty_sets():
    config = _base_config()
    config["pair_validation"] = {
        "enabled": True,
        "training_pairs": ["BTC/USDT"],
        "validation_pairs": ["BTC/USDT"],
        "weight_train": 0.6,
        "weight_val": 0.4,
    }

    errors, _ = _messages(config)

    assert _has(errors, "must be disjoint")


def test_pair_validation_accepts_any_positive_weight_scale():
    config = _base_config()
    config["pair_validation"] = {
        "enabled": True,
        "training_pairs": ["BTC/USDT"],
        "validation_pairs": ["ETH/USDT"],
        "weight_train": 3.0,
        "weight_val": 1.0,
    }

    errors, _ = _messages(config)

    assert errors == []


@pytest.mark.parametrize("workers", [0, -1, True])
def test_explicit_parallel_worker_count_must_be_positive_integer(workers):
    config = _base_config()
    config["parallel_evaluation"]["num_workers"] = workers

    errors, _ = _messages(config)

    assert _has(errors, "num_workers")


def test_auto_parallel_worker_count_remains_valid():
    config = _base_config()
    config["parallel_evaluation"]["num_workers"] = None

    errors, _ = _messages(config)

    assert errors == []


@pytest.mark.parametrize("population", [2, 6, 8, 60, 100])
def test_island_population_has_no_empirical_magic_number(population):
    config = _base_config()
    config["generic_island_model"].update(
        {"enabled": True, "num_islands": 4, "population_per_island": population}
    )
    config["generic_island_model"]["migration"]["count"] = min(2, population)

    errors, warnings = _messages(config)

    assert errors == []
    assert not _has(warnings, "safe limit")
    assert not _has(warnings, "overfit")


def test_generic_island_migration_must_fit_source_population():
    config = _base_config()
    config["generic_island_model"].update(
        {"enabled": True, "num_islands": 4, "population_per_island": 3}
    )
    config["generic_island_model"]["migration"]["count"] = 4

    errors, _ = _messages(config)

    assert _has(errors, "migration.count must not exceed")


def test_classic_island_rejects_walk_forward_it_would_silently_disable():
    config = _base_config()
    config["island_model"]["enabled"] = True
    config["walk_forward"]["enabled"] = True

    errors, _ = _messages(config)

    assert _has(errors, "classic island engine disables walk-forward")


def test_generic_island_can_deliberately_run_walk_forward():
    config = _base_config()
    config["generic_island_model"]["enabled"] = True
    config["walk_forward"]["enabled"] = True

    errors, _ = _messages(config)

    assert errors == []


def test_canonical_resolver_does_not_emit_retired_tuning_claims():
    raw = {
        "genetic_algorithm": {
            "population_size": 100,
            "elite_size": 10,
            "tournament_size": 8,
            "random_immigrants": 10,
        },
        "backtesting": {
            "pairs": [
                "BTC/USDT",
                "ETH/USDT",
                "BNB/USDT",
                "SOL/USDT",
                "XRP/USDT",
            ],
            "timerange": "20230101-20260101",
        },
    }

    resolution = resolve_config_data(raw)

    assert resolution.errors == ()
    joined = "\n".join(resolution.warnings)
    assert "ANTI-PATTERN" not in joined
    assert "safe ceiling" not in joined
    assert "generaliz" not in joined


def test_editor_hook_delegates_to_canonical_resolver():
    hook_path = (
        Path(__file__).resolve().parents[2]
        / ".github"
        / "hooks"
        / "scripts"
        / "validate_ga_config.py"
    )
    hook_validate = runpy.run_path(str(hook_path))["validate_ga_config"]
    config = _base_config()
    config["genetic_algorithm"].update(
        {
            "population_size": 100,
            "elite_size": 10,
            "tournament_size": 8,
            "random_immigrants": 10,
        }
    )
    config["backtesting"]["pairs"] = [
        "BTC/USDT",
        "ETH/USDT",
        "BNB/USDT",
        "SOL/USDT",
        "XRP/USDT",
    ]

    assert hook_validate(config) == []

    config["genetic_algorithm"]["population_size"] = 1
    assert _has(hook_validate(config), "population_size")


def test_repo_config_inventory_has_only_known_historical_design_errors():
    config_root = Path(__file__).resolve().parents[1] / "config"
    expected_invalid = {
        "done/06_E24_e7_pop8_gen15.yaml",
        "exploration/wave1/E3_island_wf_stress.yaml",
        "exploration/wave2/E7_island_wf_optimized.yaml",
        "exploration/wave3/E10_island_wf_highmut.yaml",
        "exploration/wave3/E11_island_wf_scaled.yaml",
        "exploration/wave3/E13_island_wf_longwindow.yaml",
        "exploration/wave3/E9_island_wf_llm.yaml",
        "exploration/wave4/E14_island_wf_strict_mc.yaml",
        "exploration/wave4/E16_island_wf_elite_mc.yaml",
        "exploration/wave4/E18_island_wf_component_cx.yaml",
    }
    observed_invalid: set[str] = set()
    for path in sorted(config_root.rglob("*.yaml")):
        raw = yaml.safe_load(path.read_text()) or {}
        resolved = deep_merge(DEFAULTS, resolve_preset(raw))
        errors, warnings = validate_runtime_invariants(resolved)
        if errors:
            assert errors == [
                "island_model and walk_forward cannot both be enabled because "
                "the classic island engine disables walk-forward"
            ]
            observed_invalid.add(path.relative_to(config_root).as_posix())
        joined = "\n".join(warnings)
        assert "ANTI-PATTERN" not in joined
        assert "safe ceiling" not in joined
        assert "Sacred limit" not in joined

    assert observed_invalid == expected_invalid


def test_active_validator_instructions_contain_no_retired_magic_limits():
    root = Path(__file__).resolve().parents[2]
    paths = [
        root / "genetic_algorithm" / "config" / "invariants.py",
        root / "genetic_algorithm" / "utils" / "config_validator.py",
        root / ".github" / "hooks" / "scripts" / "validate_ga_config.py",
        root / ".github" / "instructions" / "ga-config.instructions.md",
        root / ".github" / "prompts" / "generate-ga-config.prompt.md",
        root / ".github" / "prompts" / "plan-productionGaConfig.prompt.md",
        root / ".github" / "prompts" / "plan-next-wave.prompt.md",
        root / ".github" / "skills" / "launch-wave" / "SKILL.md",
    ]
    retired = re.compile(
        r"safe ceiling|Sacred limit|population_size\s*[<>]=?\s*15|"
        r"population_per_island\s*[<>]=?\s*(?:6|60)|"
        r"tournament_size.*(?:3-6|3–6)",
        re.IGNORECASE,
    )

    for path in paths:
        assert not retired.search(path.read_text()), path


def test_preflight_adds_only_contextual_data_availability(tmp_path):
    config = _base_config()

    errors, warnings = preflight_check(config, data_root=tmp_path / "missing")

    assert errors == []
    assert _has(warnings, "data directory")


def test_validate_and_log_uses_canonical_schema():
    assert validate_and_log(_base_config()) is True
    assert validate_and_log({}) is False
