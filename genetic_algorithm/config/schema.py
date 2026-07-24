"""
Config schema — sensible defaults, validation, and preset merging.

Design:
  - Legacy runtime defaults live in ``DEFAULTS``.
  - V2-only operational keys are part of the validation shape without being
    injected into legacy configs when an empty value could change behaviour.
  - Users only need to specify the non-default runtime values they need.
  - Presets (quick_test, standard, island, production, nsga2) are thin
    YAML files that override a few defaults.
  - ``load_config(path)`` loads YAML → deep-merges onto DEFAULTS → validates.

Usage::

    from genetic_algorithm.config.schema import load_config
    config = load_config("genetic_algorithm/config/presets/island.yaml")
"""

from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

from genetic_algorithm.config.invariants import validate_runtime_invariants


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults — every supported key with its default value
# ---------------------------------------------------------------------------

DEFAULTS: Dict[str, Any] = {
    "config_schema_version": 1,
    # === Backtesting ===
    "backtesting": {
        "pairs": ["BTC/USDT"],
        "timerange": "",
        "stake_amount": 0.15,
        "max_open_trades": 3,
        "fee": 0.001,
        "slippage_pct": 0.0005,
        "exchange": "binance",
        "auto_download_data": True,
        "timeout": 120,
        "enable_cache": True,
        "fee_noise_std": 0.0,
    },
    # === Strict V2 evaluation (shadow-only until promotion gates exist) ===
    "evaluation_v2": {
        "enabled": False,
        "mark_to_market": False,
        "annual_risk_free_rate": 0.0,
        "bootstrap_samples": 1000,
        "bootstrap_block_days": 10,
        "trade_bootstrap_block": 5,
        "expectancy_cluster_days": 1,
        "bootstrap_confidence": 0.95,
    },
    # === Shadow promotion gates (requires an explicit, predeclared panel) ===
    "promotion_v2": {
        "enabled": False,
        "policy_version": "promotion-v2.0-shadow",
        "required_scenarios": [],
        "min_expectancy_lcb": 0.0,
        "min_median_annual_return_lcb": 0.0,
        "min_worst_scenario_return": -0.05,
        "max_drawdown_ucb": 0.25,
        "max_daily_es5_ucb": 0.05,
        "min_effective_sample_size": 30.0,
        "min_active_months": 6,
        "min_trades_per_active_month": 5.0,
        "min_profitable_scenario_ratio": 0.6,
        "max_consecutive_losses": 10,
        "max_drawdown_duration_days": 90.0,
        "require_final_test": True,
    },
    # === Genetic Algorithm ===
    "genetic_algorithm": {
        "random_seed": None,
        "max_runtime_minutes": None,
        "population_size": 30,
        "generations": 12,
        "mutation_rate": 0.20,
        "max_mutation_rate": 0.35,
        "crossover_rate": 0.75,
        "crossover_method": "uniform",
        "elite_size": 4,
        "tournament_size": 3,
        "selection_method": "tournament",
        "convergence_patience": 8,
        "adaptive_mutation": True,
        "max_adaptation_factor": 2.5,
        "adaptation_step": 0.15,
        "mutation_cooldown_factor": 0.5,
        "adaptive_tournament": False,
        "behavioral_distance_weight": 0.0,
        "fitness_sharing": True,
        "sharing_radius": 0.25,
        "diversity_threshold": 0.15,
        "allow_self_crossover": False,
        "random_immigrants": 5,
        "mode": "single_objective",
    },
    # === Fitness weights (must sum to ~1.0) ===
    "fitness_weights": {
        "profit": 0.28,
        "sharpe_ratio": 0.18,
        "sortino_ratio": 0.13,
        "profit_factor": 0.08,
        "drawdown": 0.15,
        "win_rate": 0.05,
        "trade_frequency": 0.05,
        "monthly_stability": 0.04,
        "cross_pair": 0.04,
    },
    # === Fitness penalties ===
    "fitness_penalties": {
        "min_trades": 12,
        "max_drawdown": 0.25,
        "min_win_rate": 0.35,
        "complexity_weight": 0.015,
        "unused_indicator_weight": 0.02,
        "pair_loss_threshold": -8.0,
        "target_trades_per_pair": 0,
        "pair_trade_penalty_floor": 0.01,
    },
    # === Indicators ===
    "indicators": {
        "available": [
            "RSI",
            "MACD",
            "BBANDS",
            "EMA",
            "SMA",
            "ADX",
            "SUPERTREND",
            "PSAR",
            "CMF",
            "CDL_ENGULFING",
            "CDL_HAMMER",
            "CDL_DOJI",
        ],
        "min_per_strategy": 2,
        "max_per_strategy": 5,
        "min_entry_conditions": 2,
        "max_entry_conditions": 4,
        "min_exit_conditions": 1,
        "max_exit_conditions": 3,
    },
    # === Strategy constraints ===
    "strategy_constraints": {
        "timeframes": ["5m", "15m", "1h"],
        "stoploss_range": [-0.20, -0.05],
        "roi_range": [0.01, 0.10],
        "max_open_trades_range": [1, 5],
        "min_indicators": 2,
        "max_indicators": 8,
        "min_entry_conditions": 1,
        "max_entry_conditions": 3,
        "min_exit_conditions": 1,
        "max_exit_conditions": 3,
        "allow_indicators": None,  # None = all allowed
    },
    # === Parallel evaluation ===
    "parallel_evaluation": {
        "enabled": True,
        "num_workers": None,  # None = auto-detect
        "backtest_timeout": 120,
    },
    # === Walk-forward validation ===
    "walk_forward": {
        "enabled": False,
        "train_days": 120,
        "validation_days": 30,
        "step_days": 30,
        "mode": "rolling",
        "aggregation": "mean",
        "embargo_days": 5,
        "gap_penalty": {"enabled": True, "threshold": 0.1, "max_penalty": 0.5},
        "min_train_trades": 10,
        "max_windows": 10,
    },
    # === Holdout validation ===
    "holdout_validation": {
        "enabled": True,
        "holdout_pct": 0.15,
    },
    # === Holdout monitoring (during evolution) ===
    "holdout_monitoring": {
        "enabled": True,
        "interval": 2,
        "top_n": 5,
        "early_stop": True,
        "early_stop_threshold": 0.60,
        "early_stop_checks": 3,
        "trend_early_stop": True,
        "trend_checks": 3,
        "fitness_penalty": True,
        "penalty_factor": 0.8,
    },
    # === Monte Carlo ===
    "monte_carlo": {
        "enabled": False,
        "num_permutations": 100,
    },
    # === Deflated Sharpe ===
    "deflated_sharpe": {
        "enabled": True,
    },
    # === Surrogate model ===
    "surrogate": {
        "enabled": False,
        "min_training_samples": 50,
        "filter_percentile": 60,
    },
    # === Regime detection ===
    "regime_aware": {
        "enabled": False,
        "method": "adx_di_hysteresis",
    },
    # === Island model (generic) ===
    "generic_island_model": {
        "enabled": False,
        "num_islands": 15,
        "population_per_island": 10,
        "generations": None,  # None = inherit from genetic_algorithm.generations
        "migration": {
            "topology": "ring",
            "interval": 3,
            "count": 2,
        },
    },
    # === Island model (regime-locked) ===
    "island_model": {
        "enabled": False,
    },
    # === NSGA-II ===
    "nsga2": {
        "objectives": [
            {"name": "profit", "type": "maximize", "scale": 100.0},
            {"name": "max_drawdown", "type": "minimize", "scale": 1.0},
        ],
        "pareto_front_size": 20,
    },
    # === SIS (Strategy Intelligence System) ===
    "sis": {
        "enabled": False,
    },
    # === LLM ===
    "llm": {
        "enabled": False,
    },
    # === Terminal monitor ===
    "terminal_monitor": {
        "enabled": True,
        "default_mode": "simple",
    },
    # === Web dashboard ===
    "web_dashboard": {
        "enabled": False,
        "host": "0.0.0.0",
        "port": 8000,
    },
    # === Output / storage ===
    "output": {
        "dir": "genetic_algorithm/output",
        "top_n": 5,
        "stats_file": "evolution_stats.json",
    },
    "storage": {
        "min_disk_gb": 5.0,
    },
}


# Supported additions that are intentionally not injected as defaults.  Empty
# operational paths can change legacy ``dict.get(..., default)`` behaviour, so
# they belong to the validation shape without being merged into every config.
_V2_SHAPE_EXTRAS: Dict[str, Any] = {
    "safety_profile": {
        "name": "safe_v2",
        "enforce": True,
        "shadow_mode": True,
        "automation_eligible": False,
    },
    "backtesting": {
        "timeframe": "1h",
        "dataformat_ohlcv": "feather",
        "dynamic_slippage": False,
        "spread_pct": 0.0,
        "funding_rate": 0.0,
    },
    "output": {"plots_dir": ""},
    "pair_validation": {
        "enabled": False,
        "training_pairs": ["BTC/USDT"],
        "validation_pairs": ["ETH/USDT"],
        "weight_train": 0.6,
        "weight_val": 0.4,
        "min_val_fitness": 0.0,
        "validate_top_n_only": 0,
    },
    "generic_island_model": {
        "parallel_islands": False,
        "specialization": {
            "rotate_seeds": True,
            "indicator_pools": True,
            "indicator_overlap": 0.5,
            "pair_rotation": False,
            "pair_subset_size": 3,
        },
        "migration": {
            "merge_rounds": False,
            "merge_interval": 5,
            "tournament_size": 3,
        },
        "walk_forward": {"enabled": False},
        "external_migration": {"enabled": False, "directory": ""},
        "islands": [
            {
                "name": "island-1",
                "population_size": 10,
                "generations": 10,
                "seed": 42,
                "indicator_pool": ["RSI"],
                "pairs": ["BTC/USDT"],
                "walk_forward_enabled": False,
            }
        ],
    },
    "cpcv": {"enabled": False},
    "advanced": {"llm": {"enabled": False}},
    "short_selling": {
        "enabled": False,
        "probability": 0.5,
        "independent_conditions": False,
    },
    "surrogate": {
        "retrain_interval": 3,
        "validation_fraction": 0.2,
        "model": "random_forest",
        "mutate_skipped": True,
        "min_validation_r2": 0.3,
        "min_validation_samples": 10,
        "adaptive_filter": {
            "enabled": True,
            "min_percentile": 35,
            "r2_threshold": 0.3,
        },
    },
    "regime_aware": {
        "aggregation": "harmonic_mean",
        "cvar_alpha": 0.2,
        "confidence_weighting": True,
        "min_segment_trades": 0,
        "regime_weights": {
            "bullish": 1.0,
            "bearish": 1.0,
            "sideways": 1.0,
            "volatile": 1.0,
            "uncertain": 1.0,
        },
        "regime_specialization": {
            "enabled": False,
            "specialist_boost": 1.5,
            "diversity_weight": 0.05,
            "initial_regime_ratio": 0.3,
        },
    },
    "nsga2": {
        "objectives": [
            {
                "name": "profit",
                "type": "maximize",
                "scale": 100.0,
                "target": 50.0,
                "tolerance": 25.0,
            }
        ],
        "min_trades": 0,
        "crowding_distance_percentile": 0.1,
    },
    "multi_timeframe": {
        "enabled": False,
        "available": ["1h"],
        "max_timeframes": 2,
        "higher_timeframe_preference": ["EMA"],
    },
    "split_v2": {"min_embargo_days": 0},
    "promotion_v2": {
        "required_scenarios": [
            {
                "scenario_id": "scenario",
                "pair": "BTC/USDT",
                "timeframe": "1h",
                "role": "TEMPORAL_VALIDATION",
                "period_start": "2025-01-01",
                "period_end": "2025-02-01",
                "cost_multiplier": 1.0,
            }
        ],
    },
    "storage": {
        "checkpoint_dir": "",
        "checkpoint_interval": 5,
        "runs_dir": "",
        "cache_dir": "",
        "generated_strategy_dir": "",
        "backtest_export_dir": "",
        "backtest_user_data_dir": "",
        "max_cache_disk_mb": 0,
        "min_disk_gb_runtime": 0.0,
    },
    "hall_of_fame": {
        "directory": "",
        "max_size": 50,
        "min_fitness": 0.0,
        "inject_count": 0,
    },
    "logging": {
        "level": "INFO",
        "file": "",
        "console": True,
        "format": "",
    },
    "warm_start": {
        "enabled": False,
        "source_type": "population",
        "top_n": 15,
        "source_experiment": "",
        "source_checkpoint": "",
        "source_checkpoint_sha256": "0" * 64,
        "source_checkpoint_migration_source": "",
        "source_checkpoint_migration_source_sha256": "0" * 64,
        "source_hof": "",
        "source_hof_sha256": "0" * 64,
        "source_hof_migration_source": "",
        "source_hof_migration_source_sha256": "0" * 64,
        "genome_schema_version": "strategy-gene-v2",
    },
    "checkpoint_provenance": {
        "schema_version": "3.0",
        "engine_kind": "STANDARD",
        "config_hash": "0" * 64,
        "code_manifest_hash": "0" * 64,
        "data_manifest_hash": "0" * 64,
        "genome_schema_version": "strategy-gene-v2",
        "island_names": [""],
    },
    "experiment_name": "",
}

_DEPRECATED_CONFIG_PATHS: Dict[Tuple[str, ...], str] = {
    ("monte_carlo", "iterations"): "monte_carlo.num_permutations",
    ("monte_carlo", "n_permutations"): "monte_carlo.num_permutations",
    ("surrogate", "min_samples"): "surrogate.min_training_samples",
    ("surrogate_model",): "surrogate",
    ("generic_island_model", "migration", "rate"): "generic_island_model.migration.count",
    ("regime_aware", "detection_method"): "regime_aware.method",
    ("genetic_algorithm", "seed"): "genetic_algorithm.random_seed",
    ("strategy_constraints", "min_trades"): "fitness_penalties.min_trades",
    ("strategy_constraints", "max_drawdown"): "fitness_penalties.max_drawdown",
    ("strategy_constraints", "min_win_rate"): "fitness_penalties.min_win_rate",
}

_OPTIONAL_CONFIG_TYPES: Dict[str, type] = {
    "genetic_algorithm.random_seed": int,
    "genetic_algorithm.max_runtime_minutes": int,
    "parallel_evaluation.num_workers": int,
    "generic_island_model.generations": int,
    "strategy_constraints.allow_indicators": list,
}


_SAFE_V2_DISABLED_FEATURES: Tuple[Tuple[str, ...], ...] = (
    ("walk_forward",),
    ("holdout_validation",),
    ("holdout_monitoring",),
    ("monte_carlo",),
    ("deflated_sharpe",),
    ("surrogate",),
    ("regime_aware",),
    ("generic_island_model",),
    ("island_model",),
    ("pair_validation",),
    ("cpcv",),
    ("sis",),
    ("llm",),
    ("advanced", "llm"),
    ("short_selling",),
    ("multi_timeframe",),
)

_AUTOMATION_ISLAND_V2_DISABLED_FEATURES: Tuple[Tuple[str, ...], ...] = (
    ("walk_forward",),
    ("holdout_validation",),
    ("holdout_monitoring",),
    ("monte_carlo",),
    ("deflated_sharpe",),
    ("surrogate",),
    ("regime_aware",),
    ("island_model",),
    ("cpcv",),
    ("sis",),
    ("llm",),
    ("advanced", "llm"),
    ("short_selling",),
    ("multi_timeframe",),
)


def _nested_config(config: Dict[str, Any], path: Sequence[str]) -> Dict[str, Any]:
    """Return a nested config mapping, or an empty mapping when absent."""
    value: Any = config
    for key in path:
        if not isinstance(value, dict):
            return {}
        value = value.get(key, {})
    return value if isinstance(value, dict) else {}


# ---------------------------------------------------------------------------
# Deep merge
# ---------------------------------------------------------------------------


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge *override* into a copy of *base*.

    - Dict values are merged recursively.
    - Non-dict values in *override* replace those in *base*.
    - Keys in *override* that do not exist in *base* are kept (pass-through).
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def config_contract_shape_v2() -> Dict[str, Any]:
    """Return the explicit, version-2 persisted-config shape.

    The returned mapping is a detached value so callers cannot mutate the
    process-wide contract accidentally.
    """
    return deep_merge(DEFAULTS, _V2_SHAPE_EXTRAS)


def _join_path(prefix: str, key: object) -> str:
    return f"{prefix}.{key}" if prefix else str(key)


def _unknown_paths(value: Any, shape: Any, prefix: str = "") -> List[str]:
    if isinstance(value, dict):
        if not isinstance(shape, dict):
            return []
        unknown: List[str] = []
        for key in sorted(value, key=str):
            path = _join_path(prefix, key)
            if key not in shape:
                unknown.append(path)
            else:
                unknown.extend(_unknown_paths(value[key], shape[key], path))
        return unknown
    if isinstance(value, list) and isinstance(shape, list) and shape:
        unknown = []
        for index, item in enumerate(value):
            unknown.extend(_unknown_paths(item, shape[0], f"{prefix}[{index}]"))
        return unknown
    return []


def unknown_config_paths(config: Dict[str, Any]) -> List[str]:
    """Return sorted paths not declared by the V2 config contract."""
    return sorted(set(_unknown_paths(config, config_contract_shape_v2())))


def _matches_template_type(value: Any, template: Any) -> bool:
    if template is None:
        return True
    if isinstance(template, bool):
        return isinstance(value, bool)
    if isinstance(template, int):
        return isinstance(value, int) and not isinstance(value, bool)
    if isinstance(template, float):
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return isinstance(value, type(template))


def _type_errors(value: Any, shape: Any, prefix: str = "") -> List[str]:
    optional_type = _OPTIONAL_CONFIG_TYPES.get(prefix)
    if optional_type is not None:
        if value is None:
            return []
        valid = isinstance(value, optional_type)
        if optional_type is int:
            valid = valid and not isinstance(value, bool)
        if not valid:
            return [
                f"{prefix} must be {optional_type.__name__} or null, got {type(value).__name__}"
            ]

    if not _matches_template_type(value, shape):
        expected = type(shape).__name__ if shape is not None else "any"
        return [f"{prefix or '<root>'} must be {expected}, got {type(value).__name__}"]

    errors: List[str] = []
    if isinstance(value, dict) and isinstance(shape, dict):
        for key in sorted(set(value).intersection(shape), key=str):
            errors.extend(_type_errors(value[key], shape[key], _join_path(prefix, key)))
    elif isinstance(value, list) and isinstance(shape, list) and shape:
        for index, item in enumerate(value):
            errors.extend(_type_errors(item, shape[0], f"{prefix}[{index}]"))
    return errors


def config_type_errors(config: Dict[str, Any]) -> List[str]:
    """Return strict container/scalar type errors for a resolved config."""
    return sorted(set(_type_errors(config, config_contract_shape_v2())))


# ---------------------------------------------------------------------------
# Preset resolution
# ---------------------------------------------------------------------------

_PRESETS_DIR = Path(__file__).parent / "presets"


def resolve_preset(config: Dict[str, Any]) -> Dict[str, Any]:
    """If *config* contains a ``preset`` key, load that preset YAML and
    merge *config* on top of it (config overrides preset overrides defaults).
    """
    if not isinstance(config, dict):
        raise ValueError("Config root must be a YAML mapping")
    config = copy.deepcopy(config)
    preset_name = config.pop("preset", None)
    if not preset_name:
        return config
    if (
        not isinstance(preset_name, str)
        or Path(preset_name).name != preset_name
        or preset_name in {".", ".."}
    ):
        raise ValueError(f"Invalid preset name: {preset_name!r}")

    preset_file = _PRESETS_DIR / f"{preset_name}.yaml"
    if not preset_file.exists():
        available = [p.stem for p in _PRESETS_DIR.glob("*.yaml")]
        raise FileNotFoundError(
            f"Preset '{preset_name}' not found at {preset_file}. Available: {available}"
        )

    with open(preset_file) as fh:
        preset_data = yaml.safe_load(fh) or {}
    if not isinstance(preset_data, dict):
        raise ValueError(f"Preset '{preset_name}' root must be a YAML mapping")

    # preset overrides defaults, then user config overrides preset
    merged = deep_merge(preset_data, config)
    return merged


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_config(config: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """Validate a loaded config dict.

    Returns (errors, warnings).  Errors are fatal; warnings are advisory.
    """
    errors: List[str] = []
    warnings: List[str] = []

    schema_version = config.get("config_schema_version", 1)
    if type(schema_version) is not int or schema_version not in {1, 2}:
        errors.append("config_schema_version must be integer 1 or 2")

    unknown = unknown_config_paths(config)
    type_errors = config_type_errors(config)
    errors.extend(type_errors)
    if schema_version == 2:
        if unknown:
            errors.append("unknown config paths: " + ", ".join(unknown))
    elif unknown:
        preview = unknown[:20]
        suffix = f" (+{len(unknown) - len(preview)} more)" if len(unknown) > len(preview) else ""
        warnings.append(
            "schema v1 accepts undeclared config paths for compatibility: "
            + ", ".join(preview)
            + suffix
        )

    for path, replacement in _DEPRECATED_CONFIG_PATHS.items():
        value: Any = config
        present = True
        for key in path:
            if not isinstance(value, dict) or key not in value:
                present = False
                break
            value = value[key]
        if present:
            message = f"deprecated config path {'.'.join(path)}; use {replacement}"
            if schema_version == 2:
                errors.append(message)
            else:
                warnings.append(message)

    ga = _nested_config(config, ("genetic_algorithm",))
    bt = _nested_config(config, ("backtesting",))
    max_runtime_minutes = ga.get("max_runtime_minutes")
    if (
        max_runtime_minutes is not None
        and (
            not isinstance(max_runtime_minutes, int)
            or isinstance(max_runtime_minutes, bool)
            or max_runtime_minutes < 1
        )
    ):
        errors.append(
            "genetic_algorithm.max_runtime_minutes must be a positive integer or null"
        )

    # --- NSGA-II contract ---
    mode = ga.get("mode")
    if mode == "multi_objective":
        errors.append(
            "genetic_algorithm.mode: multi_objective is not supported; use "
            "genetic_algorithm.mode: nsga2 and configure nsga2.objectives"
        )

    if mode == "nsga2":
        objective_cfg = _nested_config(config, ("nsga2",)).get("objectives")
        supported_objectives = {
            "profit",
            "max_drawdown",
            "sharpe_ratio",
            "sortino_ratio",
            "profit_factor",
            "win_rate",
            "num_trades",
            # Compatibility aliases resolved in engine.nsga2.
            "trade_frequency",
            "trade_count",
        }
        if not isinstance(objective_cfg, list) or not objective_cfg:
            errors.append("nsga2.objectives must be a non-empty list in nsga2 mode")
        else:
            seen_objectives = set()
            for index, objective in enumerate(objective_cfg):
                path = f"nsga2.objectives[{index}]"
                if not isinstance(objective, dict):
                    errors.append(f"{path} must be a mapping")
                    continue
                if "direction" in objective or "weight" in objective:
                    errors.append(f"{path} uses ignored direction/weight keys; use type and scale")
                name = objective.get("name")
                if name not in supported_objectives:
                    errors.append(f"{path}.name must be a supported backtest metric, got {name!r}")
                elif name in seen_objectives:
                    errors.append(f"{path}.name duplicates objective {name!r}")
                else:
                    seen_objectives.add(name)

                objective_type = objective.get("type", "maximize")
                if objective_type not in {"maximize", "minimize", "goldilocks"}:
                    errors.append(f"{path}.type must be maximize, minimize, or goldilocks")
                scale = objective.get("scale", 1.0)
                if (
                    not isinstance(scale, (int, float))
                    or isinstance(scale, bool)
                    or not math.isfinite(scale)
                    or scale <= 0
                ):
                    errors.append(f"{path}.scale must be a finite number > 0")
                if objective_type == "goldilocks":
                    tolerance = objective.get("tolerance", 25)
                    if (
                        not isinstance(tolerance, (int, float))
                        or isinstance(tolerance, bool)
                        or not math.isfinite(tolerance)
                        or tolerance <= 0
                    ):
                        errors.append(
                            f"{path}.tolerance must be a finite number > 0 for goldilocks"
                        )

        if _nested_config(config, ("surrogate",)).get("enabled", False):
            errors.append(
                "NSGA-II is incompatible with surrogate.enabled: true because "
                "surrogate-only candidates have no measured objective vector"
            )

    surrogate = _nested_config(config, ("surrogate",))
    surrogate_integer_bounds = {
        "min_training_samples": (20, None),
        "retrain_interval": (1, None),
        "min_validation_samples": (5, None),
    }
    for key, (minimum, maximum) in surrogate_integer_bounds.items():
        value = surrogate.get(key)
        if value is None:
            continue
        if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
            errors.append(f"surrogate.{key} must be an integer >= {minimum}")
    filter_percentile = surrogate.get("filter_percentile", 60)
    if (
        not isinstance(filter_percentile, (int, float))
        or isinstance(filter_percentile, bool)
        or not math.isfinite(filter_percentile)
        or not 1 <= filter_percentile <= 100
    ):
        errors.append("surrogate.filter_percentile must be between 1 and 100")
    validation_fraction = surrogate.get("validation_fraction", 0.2)
    if (
        not isinstance(validation_fraction, (int, float))
        or isinstance(validation_fraction, bool)
        or not math.isfinite(validation_fraction)
        or not 0.1 <= validation_fraction <= 0.5
    ):
        errors.append("surrogate.validation_fraction must be between 0.1 and 0.5")
    min_validation_r2 = surrogate.get("min_validation_r2", 0.3)
    if (
        not isinstance(min_validation_r2, (int, float))
        or isinstance(min_validation_r2, bool)
        or not math.isfinite(min_validation_r2)
        or not 0.0 <= min_validation_r2 <= 1.0
    ):
        errors.append("surrogate.min_validation_r2 must be between 0.0 and 1.0")
    model_type = surrogate.get("model", "random_forest")
    if model_type not in {"random_forest", "gradient_boosting"}:
        errors.append("surrogate.model must be random_forest or gradient_boosting")

    short_selling = _nested_config(config, ("short_selling",))
    if "probability" in short_selling:
        short_probability = short_selling["probability"]
        if (
            not isinstance(short_probability, (int, float))
            or isinstance(short_probability, bool)
            or not 0.0 <= short_probability <= 1.0
        ):
            errors.append("short_selling.probability must be between 0.0 and 1.0")

    # --- Regime aggregation contract ---
    regime_aware = _nested_config(config, ("regime_aware",))
    aggregation = regime_aware.get("aggregation", "harmonic_mean")
    if aggregation not in {"mean", "min", "harmonic_mean", "cvar"}:
        errors.append("regime_aware.aggregation must be mean, min, harmonic_mean, or cvar")
    min_segment_trades = regime_aware.get("min_segment_trades", 0)
    if type(min_segment_trades) is not int or min_segment_trades < 0:
        errors.append("regime_aware.min_segment_trades must be a non-negative integer")
    cvar_alpha = regime_aware.get("cvar_alpha", 0.2)
    if (
        not isinstance(cvar_alpha, (int, float))
        or isinstance(cvar_alpha, bool)
        or not math.isfinite(cvar_alpha)
        or not 0.0 < cvar_alpha <= 1.0
    ):
        errors.append("regime_aware.cvar_alpha must be a finite number in (0, 1]")
    regime_weights = regime_aware.get("regime_weights", {})
    if not isinstance(regime_weights, dict):
        errors.append("regime_aware.regime_weights must be a mapping")
    else:
        for regime_name, weight in sorted(regime_weights.items()):
            if (
                not isinstance(weight, (int, float))
                or isinstance(weight, bool)
                or not math.isfinite(weight)
                or weight <= 0.0
            ):
                errors.append(
                    f"regime_aware.regime_weights.{regime_name} must be a finite number > 0"
                )

    specialization = _nested_config(config, ("regime_aware", "regime_specialization"))
    specialist_boost = specialization.get("specialist_boost", 1.5)
    if (
        not isinstance(specialist_boost, (int, float))
        or isinstance(specialist_boost, bool)
        or not math.isfinite(specialist_boost)
        or specialist_boost <= 0.0
    ):
        errors.append(
            "regime_aware.regime_specialization.specialist_boost must be a finite number > 0"
        )
    diversity_weight = specialization.get("diversity_weight", 0.05)
    if (
        not isinstance(diversity_weight, (int, float))
        or isinstance(diversity_weight, bool)
        or not math.isfinite(diversity_weight)
        or diversity_weight < 0.0
    ):
        errors.append(
            "regime_aware.regime_specialization.diversity_weight must be a finite number >= 0"
        )

    # --- Legacy warm-start is explicit and content-addressed ---
    warm_start = _nested_config(config, ("warm_start",))
    if warm_start.get("enabled", False):
        source_type = warm_start.get("source_type")
        if source_type not in {"population", "hof", "both"}:
            errors.append("warm_start.source_type must be population, hof, or both")
        top_n = warm_start.get("top_n")
        if type(top_n) is not int or top_n < 1:
            errors.append("warm_start.top_n must be a positive integer")
        if warm_start.get("genome_schema_version") != "strategy-gene-v2":
            errors.append("warm_start.genome_schema_version must be strategy-gene-v2")

        declarations = []
        if source_type in {"population", "both"}:
            declarations.append(("source_checkpoint", "source_checkpoint_sha256"))
        if source_type in {"hof", "both"}:
            declarations.append(("source_hof", "source_hof_sha256"))
        for path_key, hash_key in declarations:
            path_value = warm_start.get(path_key)
            digest = warm_start.get(hash_key)
            if not isinstance(path_value, str) or not path_value.strip():
                errors.append(f"warm_start.{path_key} must be an explicit path")
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                errors.append(f"warm_start.{hash_key} must be a lowercase SHA-256 digest")

            migration_path_key = f"{path_key}_migration_source"
            migration_hash_key = f"{path_key}_migration_source_sha256"
            migration_path = warm_start.get(migration_path_key, "")
            migration_digest = warm_start.get(migration_hash_key, "0" * 64)
            migration_declared = bool(migration_path) or migration_digest != "0" * 64
            if migration_declared:
                if not isinstance(migration_path, str) or not migration_path.strip():
                    errors.append(f"warm_start.{migration_path_key} must be an explicit path")
                if (
                    not isinstance(migration_digest, str)
                    or len(migration_digest) != 64
                    or any(character not in "0123456789abcdef" for character in migration_digest)
                ):
                    errors.append(
                        f"warm_start.{migration_hash_key} must be a lowercase SHA-256 digest"
                    )

    # --- safe_v2 is an enforced baseline, not only a collection of defaults ---
    safety = _nested_config(config, ("safety_profile",))
    if safety.get("name") == "safe_v2" and safety.get("enforce", True):
        if config.get("config_schema_version") != 2:
            errors.append("safe_v2 requires config_schema_version: 2")
        if ga.get("mode") != "single_objective":
            errors.append("safe_v2 requires genetic_algorithm.mode: single_objective")
        for path in _SAFE_V2_DISABLED_FEATURES:
            feature = _nested_config(config, path)
            if feature.get("enabled", False):
                errors.append(f"safe_v2 forbids {'.'.join(path)}.enabled: true")
        if bt.get("fee_noise_std", 0.0) != 0.0:
            errors.append("safe_v2 requires backtesting.fee_noise_std: 0.0")
        evaluation_v2 = _nested_config(config, ("evaluation_v2",))
        if not evaluation_v2.get("enabled", False):
            errors.append("safe_v2 requires evaluation_v2.enabled: true")
        if not evaluation_v2.get("mark_to_market", False):
            errors.append("safe_v2 requires evaluation_v2.mark_to_market: true")
    if safety.get("name") == "automation_island_v2" and safety.get("enforce", True):
        if config.get("config_schema_version") != 2:
            errors.append("automation_island_v2 requires config_schema_version: 2")
        if ga.get("mode") != "single_objective":
            errors.append(
                "automation_island_v2 requires genetic_algorithm.mode: single_objective"
            )
        if not safety.get("shadow_mode", False):
            errors.append("automation_island_v2 requires shadow_mode: true")
        if safety.get("automation_eligible", False):
            errors.append(
                "automation_island_v2 forbids strategy automation_eligible: true"
            )
        if max_runtime_minutes is None:
            errors.append(
                "automation_island_v2 requires genetic_algorithm.max_runtime_minutes"
            )
        for path in _AUTOMATION_ISLAND_V2_DISABLED_FEATURES:
            feature = _nested_config(config, path)
            if feature.get("enabled", False):
                errors.append(
                    f"automation_island_v2 forbids {'.'.join(path)}.enabled: true"
                )
        generic_island = _nested_config(config, ("generic_island_model",))
        if not generic_island.get("enabled", False):
            errors.append(
                "automation_island_v2 requires generic_island_model.enabled: true"
            )
        if generic_island.get("parallel_islands", False):
            errors.append("automation_island_v2 requires parallel_islands: false")
        if generic_island.get("specialization", {}).get("pair_rotation", False):
            errors.append("automation_island_v2 requires pair_rotation: false")
        pair_validation = _nested_config(config, ("pair_validation",))
        if not pair_validation.get("enabled", False):
            errors.append("automation_island_v2 requires pair_validation.enabled: true")
        if pair_validation.get("validate_top_n_only", 0) != 0:
            errors.append(
                "automation_island_v2 requires pair_validation.validate_top_n_only: 0"
            )
        if bt.get("fee_noise_std", 0.0) != 0.0:
            errors.append("automation_island_v2 requires backtesting.fee_noise_std: 0.0")
        evaluation_v2 = _nested_config(config, ("evaluation_v2",))
        if not evaluation_v2.get("enabled", False):
            errors.append("automation_island_v2 requires evaluation_v2.enabled: true")
        if not evaluation_v2.get("mark_to_market", False):
            errors.append("automation_island_v2 requires evaluation_v2.mark_to_market: true")
        if not _nested_config(config, ("promotion_v2",)).get("enabled", False):
            errors.append("automation_island_v2 requires promotion_v2.enabled: true")
    if (
        safety.get("name") == "automation_island_child_v2"
        and safety.get("enforce", True)
    ):
        if config.get("config_schema_version") != 2:
            errors.append("automation_island_child_v2 requires config_schema_version: 2")
        if ga.get("mode") != "single_objective":
            errors.append(
                "automation_island_child_v2 requires genetic_algorithm.mode: "
                "single_objective"
            )
        if not safety.get("shadow_mode", False):
            errors.append("automation_island_child_v2 requires shadow_mode: true")
        if safety.get("automation_eligible", False):
            errors.append(
                "automation_island_child_v2 forbids strategy automation_eligible: true"
            )
        if max_runtime_minutes is None:
            errors.append(
                "automation_island_child_v2 requires genetic_algorithm.max_runtime_minutes"
            )
        for path in _AUTOMATION_ISLAND_V2_DISABLED_FEATURES:
            feature = _nested_config(config, path)
            if feature.get("enabled", False):
                errors.append(
                    f"automation_island_child_v2 forbids {'.'.join(path)}.enabled: true"
                )
        if _nested_config(config, ("generic_island_model",)).get("enabled", False):
            errors.append(
                "automation_island_child_v2 forbids recursive "
                "generic_island_model.enabled: true"
            )
        pair_validation = _nested_config(config, ("pair_validation",))
        if not pair_validation.get("enabled", False):
            errors.append(
                "automation_island_child_v2 requires pair_validation.enabled: true"
            )
        if pair_validation.get("validate_top_n_only", 0) != 0:
            errors.append(
                "automation_island_child_v2 requires "
                "pair_validation.validate_top_n_only: 0"
            )

    evaluation_v2 = _nested_config(config, ("evaluation_v2",))
    annual_risk_free_rate = evaluation_v2.get("annual_risk_free_rate", 0.0)
    if (
        not isinstance(annual_risk_free_rate, (int, float))
        or isinstance(annual_risk_free_rate, bool)
        or not math.isfinite(annual_risk_free_rate)
        or annual_risk_free_rate <= -1.0
    ):
        errors.append(
            "evaluation_v2.annual_risk_free_rate must be a finite decimal greater than -1"
        )
    if evaluation_v2.get("enabled", False):
        samples = evaluation_v2.get("bootstrap_samples", 0)
        block_days = evaluation_v2.get("bootstrap_block_days", 0)
        block_trades = evaluation_v2.get("trade_bootstrap_block", 0)
        cluster_days = evaluation_v2.get("expectancy_cluster_days", 0)
        confidence = evaluation_v2.get("bootstrap_confidence", 0.0)
        if not isinstance(samples, int) or isinstance(samples, bool) or samples < 100:
            errors.append("evaluation_v2.bootstrap_samples must be an integer >= 100")
        if not isinstance(block_days, int) or isinstance(block_days, bool) or block_days < 1:
            errors.append("evaluation_v2.bootstrap_block_days must be a positive integer")
        if not isinstance(block_trades, int) or isinstance(block_trades, bool) or block_trades < 1:
            errors.append("evaluation_v2.trade_bootstrap_block must be a positive integer")
        if (
            not isinstance(cluster_days, int)
            or isinstance(cluster_days, bool)
            or cluster_days < 1
        ):
            errors.append("evaluation_v2.expectancy_cluster_days must be a positive integer")
        if (
            not isinstance(confidence, (int, float))
            or isinstance(confidence, bool)
            or not 0.5 < confidence < 1.0
        ):
            errors.append("evaluation_v2.bootstrap_confidence must be between 0.5 and 1.0")

    promotion_v2 = _nested_config(config, ("promotion_v2",))
    if promotion_v2.get("enabled", False):
        try:
            from genetic_algorithm.orchestration.promotion_policy_v2 import (
                shadow_gate_policy_from_config,
            )

            shadow_gate_policy_from_config(config)
        except (TypeError, ValueError) as exc:
            errors.append(f"invalid promotion_v2 policy: {exc}")

    invariant_errors, invariant_warnings = validate_runtime_invariants(config)
    errors.extend(invariant_errors)
    warnings.extend(invariant_warnings)

    return sorted(set(errors)), sorted(set(warnings))


# ---------------------------------------------------------------------------
# Main entry: resolve_config / load_config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfigResolution:
    """Detached resolved config plus its canonical validation report."""

    config: Dict[str, Any]
    errors: Tuple[str, ...]
    warnings: Tuple[str, ...]

    def require_valid(self) -> Dict[str, Any]:
        if self.errors:
            msg = "Config validation failed:\n" + "\n".join(f"  - {error}" for error in self.errors)
            raise ValueError(msg)
        return copy.deepcopy(self.config)


def resolve_config_data(
    raw: Dict[str, Any],
    *,
    overrides: Optional[Dict[str, Any]] = None,
) -> ConfigResolution:
    """Resolve and validate an in-memory YAML mapping through the canonical path."""

    if not isinstance(raw, dict):
        raise ValueError("Config root must be a YAML mapping")

    resolved = resolve_preset(raw)
    config = deep_merge(DEFAULTS, resolved)

    if overrides is not None:
        if not isinstance(overrides, dict):
            raise ValueError("Config overrides must be a mapping")
        config = deep_merge(config, overrides)

    errors, warnings_ = validate_config(config)
    return ConfigResolution(
        config=copy.deepcopy(config),
        errors=tuple(errors),
        warnings=tuple(warnings_),
    )


def resolve_config(
    path: str | Path,
    *,
    overrides: Optional[Dict[str, Any]] = None,
) -> ConfigResolution:
    """Read, resolve, merge, and validate a config exactly once."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as fh:
        raw = yaml.safe_load(fh)
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("Config root must be a YAML mapping")
    return resolve_config_data(raw, overrides=overrides)


def validate_config_or_raise(config: Dict[str, Any]) -> None:
    """Apply the canonical contract to an already-resolved runtime config."""
    errors, warnings_ = validate_config(config)
    for warning in warnings_:
        logger.warning("Config warning: %s", warning)
    if errors:
        msg = "Config validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ValueError(msg)


def _missing_default_paths(
    config: Any,
    defaults: Any = DEFAULTS,
    prefix: str = "",
) -> List[str]:
    """Return DEFAULTS-backed paths absent from an allegedly resolved config."""
    if not isinstance(defaults, dict) or not isinstance(config, dict):
        return []
    missing: List[str] = []
    for key in sorted(defaults, key=str):
        path = _join_path(prefix, key)
        if key not in config:
            missing.append(path)
        else:
            missing.extend(_missing_default_paths(config[key], defaults[key], path))
    return missing


def validate_resolved_config_v2_or_raise(config: Dict[str, Any]) -> None:
    """Require a complete, fail-closed schema-V2 config at an artifact boundary.

    ``validate_config_or_raise`` intentionally keeps schema V1 readable for
    legacy runners.  Immutable V2 manifests/workers must not inherit that
    compatibility mode and must receive a config on which DEFAULTS have
    already been resolved.
    """
    if config.get("config_schema_version") != 2:
        raise ValueError("V2 artifact requires config_schema_version: 2")
    missing = _missing_default_paths(config)
    if missing:
        raise ValueError(
            "V2 artifact requires a fully resolved config; missing paths: " + ", ".join(missing)
        )
    validate_config_or_raise(config)


def load_config(
    path: str | Path,
    *,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Load a GA configuration from *path*.

    1. Read YAML file.
    2. If ``preset`` key present, load that preset and merge on top.
    3. Deep-merge onto :data:`DEFAULTS`.
    4. Apply *overrides* (CLI flags, env vars, etc.).
    5. Validate; log warnings, raise on errors.

    Returns the final merged config dict.
    """
    resolution = resolve_config(path, overrides=overrides)
    for warning in resolution.warnings:
        logger.warning("Config warning: %s", warning)
    return resolution.require_valid()
