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
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

from genetic_algorithm.config.invariants import validate_runtime_invariants
from genetic_algorithm.utils.timerange import validate_walk_forward_config


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
    # === Guarded unattended controller ===
    "automation_controller": {
        "root_seeds": [1001],
        "max_waves": 50,
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
        "seed_known_archetypes": True,
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
    # === Fitness normalization bounds ===
    "fitness_bounds": {
        "profit_min": -10.0,
        "profit_max": 10.0,
        "sharpe_min": -5.0,
        "sharpe_max": 10.0,
        "sortino_min": -5.0,
        "sortino_max": 12.0,
        "profit_factor_max": 10.0,
        "profit_factor_normalization": 3.0,
        # Zero preserves the legacy 1-drawdown normalization. Audited profiles
        # may opt into a smooth target-relative scale.
        "drawdown_normalization_target": 0.0,
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
        # Zero preserves the historical all-AND initial population.  Search
        # profiles may opt into evolvable disjunctive signal topologies.
        "initial_or_probability": 0.0,
    },
    # === Strategy constraints ===
    "strategy_constraints": {
        "timeframes": ["5m", "15m", "1h"],
        "startup_candle_floor": None,
        "startup_candle_cap": None,
        "canonicalize_executable_genome": False,
        "stoploss_range": [-0.20, -0.05],
        "roi_range": [0.01, 0.10],
        "trailing_stop_positive_range": [0.01, 0.03],
        "trailing_stop_offset_addition_range": [0.01, 0.03],
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
    "genetic_algorithm": {
        # Optional deterministic salt used to separate an arm's evolutionary
        # search RNG from the paired V2 evaluation/bootstrap seed.
        "search_seed_salt": 0,
    },
    "fitness_weights": {
        "drawdown_duration": 0.0,
        "consecutive_losses": 0.0,
    },
    "fitness_bounds": {
        "profit_factor_break_even_normalization": False,
    },
    "fitness_penalties": {
        "target_trades_per_active_month": 5.0,
        "pair_trade_coverage_exponent": 1.0,
    },
    "fitness_policy_v3": {
        "enabled": True,
        "policy_version": "fitness-v3.0-feasibility-first",
        "min_trades": 12,
        "min_net_profit": 0.0,
        "min_profit_factor": 1.0,
        "min_profitable_pair_ratio": 0.6666666666666666,
        "max_worst_pair_loss": -5.0,
        "max_drawdown": 0.20,
        "max_drawdown_duration_days": 120.0,
        "median_hold_hours_target": 24.0,
        "p90_hold_hours_target": 72.0,
        "holding_soft_weight": 0.10,
    },
    # V3 qualification is an additive shadow contract.  Keeping it outside
    # DEFAULTS prevents legacy/V2 runs from changing behavior merely because
    # the schema learned how to validate the new section.
    "qualification_v3": {
        "enabled": False,
        "schema_version": "3.0",
        "policy_version": "qualification-v3.0-shadow",
        "required_scenarios": [
            {
                "scenario_id": "scenario",
                "pair": "BTC/USDT",
                "pair_group": "development",
                "timeframe": "1h",
                "role": "TRAIN",
                "period_start": "2025-01-01",
                "period_end_exclusive": "2025-02-01",
                "cost_multiplier": 1.0,
            }
        ],
        "pair_groups": [
            {
                "group_id": "development",
                "pairs": ["BTC/USDT"],
                "min_profitable_pairs": 1,
                "min_group_expectancy_lcb": 0.0,
                "min_group_annual_return_lcb": 0.0,
                "min_worst_pair_scenario_return": -0.05,
            }
        ],
        "development_evidence": {
            "min_effective_sample_size": 30.0,
            "min_active_months": 12,
        },
        "final_test_evidence": {
            "min_effective_sample_size": 20.0,
            "min_active_months": 1,
        },
        "max_drawdown_ucb": 0.20,
        "max_daily_es5_ucb": 0.05,
        "max_consecutive_losses": 12,
        "max_drawdown_duration_days": 120.0,
        "require_final_test": True,
        "soft_targets": {
            "target_trades_per_day_min": 0.5,
            "target_trades_per_day_max": 1.5,
            "median_hold_hours_target": 24.0,
            "p90_hold_hours_target": 72.0,
        },
    },
    # Search-only score used by the hardcore multi-pair campaign.  This is a
    # deliberately raw contract: no confidence bounds, ESS, Sharpe-family
    # statistic, gate state or deferred validation may influence selection.
    "raw_multipair_score": {
        "enabled": False,
        "policy_version": "raw-multipair-score-v4",
        "development_pairs": ["BTC/USDT", "SOL/USDT", "XRP/USDT"],
        "validation_pairs": ["BNB/USDT", "ETH/USDT", "PEPE/USDT"],
        "period_start": "2023-05-09",
        "period_end": "2026-03-26",
        "policy": {
            "annual_return_scale": 0.20,
            "expectancy_scale": 0.005,
            "profit_factor_cap": 3.0,
            "profit_factor_scale": 0.5,
            "profit_factor_full_credit_trades": 30,
            "activity_target_trades_per_month_15m": 17.0,
            "activity_target_trades_per_month_1h": 10.0,
            "activity_target_active_month_ratio": 0.75,
            "holding_median_target_hours_15m": 8.0,
            "holding_median_target_hours_1h": 12.0,
            "holding_p90_target_hours_15m": 24.0,
            "holding_p90_target_hours_1h": 36.0,
            "max_drawdown_scale": 0.20,
            "loss_streak_scale": 10.0,
            "overtrading_free_trades_per_month": 60.0,
            "overtrading_scale_trades_per_month": 60.0,
            "development_group_weight": 0.40,
            "validation_group_weight": 0.60,
            "worst_pair_weight": 0.60,
            "group_median_weight": 0.40,
            "edge_return_weight": 0.50,
            "edge_expectancy_weight": 0.30,
            "edge_profit_factor_weight": 0.20,
            "score_edge_weight": 0.60,
            "score_productive_frequency_weight": 0.30,
            "score_holding_weight": 0.02,
            "score_drawdown_weight": -0.040,
            "score_drawdown_duration_weight": -0.020,
            "score_loss_streak_weight": -0.015,
            "score_overtrading_weight": -0.005,
        },
    },
    "safety_profile": {
        "name": "safe_v2",
        "enforce": True,
        "shadow_mode": True,
        "automation_eligible": False,
        "canary": False,
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
        "evaluation_mode": "joint",
        "training_pairs": ["BTC/USDT"],
        "validation_pairs": ["ETH/USDT"],
        "weight_train": 0.6,
        "weight_val": 0.4,
        "min_val_fitness": 0.0,
        "validate_top_n_only": 0,
        "worst_pair_weight": 0.5,
        "worst_split_weight": 0.0,
        "min_profitable_pair_ratio": 0.0,
        "profitable_pair_penalty_floor": 0.1,
        "max_pair_loss_pct": 0.0,
        "worst_pair_loss_penalty_floor": 0.1,
    },
    "generic_island_model": {
        "parallel_islands": False,
        "graceful_stop_marker": "",
        "common_panel_replay": {
            "enabled": False,
            "interval": 3,
            "top_n_per_island": 3,
            "early_stop_patience": 4,
            "min_improvement": 0.002,
            "relative_min_improvement": 0.0,
            "min_generation": 1,
            "incumbent_score": None,
        },
        "diversity_recovery": {
            "enabled": False,
            "island_threshold_count": 1,
            "consecutive_checks_before_recovery": 2,
            "consecutive_checks_after_recovery": 2,
            "duplicate_fraction_threshold": 0.75,
            "genetic_diversity_threshold": 0.10,
            "replacement_fraction": 0.25,
            "recovery_mutation_rate": 0.35,
        },
        "archive_seeding": {
            "enabled": False,
            "island_names": ["island-1"],
            "max_seeds_per_island": 4,
            "assignments": {},
            "cross_niche_offspring_per_generation": 0,
        },
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
    "generic_island_model.common_panel_replay.incumbent_score": float,
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

_HARDCORE_MULTIPAIR_V1_DISABLED_FEATURES: Tuple[Tuple[str, ...], ...] = (
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
    ("qualification_v3",),
    ("fitness_policy_v3",),
    ("nsga2",),
)

_HARDCORE_RECURSIVE_FEATURE_NAMES = frozenset(
    {
        "walk_forward",
        "holdout_validation",
        "holdout_monitoring",
        "monte_carlo",
        "deflated_sharpe",
        "surrogate",
        "regime_aware",
        "cpcv",
        "pbo",
        "sis",
        "llm",
        "short_selling",
        "multi_timeframe",
        "qualification_v3",
        "fitness_policy_v3",
        "nsga2",
    }
)


def _nested_config(config: Dict[str, Any], path: Sequence[str]) -> Dict[str, Any]:
    """Return a nested config mapping, or an empty mapping when absent."""
    value: Any = config
    for key in path:
        if not isinstance(value, dict):
            return {}
        value = value.get(key, {})
    return value if isinstance(value, dict) else {}


def _recursively_enabled_forbidden_features(
    value: Any,
    *,
    prefix: str,
) -> List[str]:
    """Return forbidden feature paths enabled inside an island subtree.

    Island configuration is intentionally inspected recursively so a child
    cannot re-enable a robustness feature hidden below an otherwise safe
    parent.  Both ``feature: {enabled: true}`` and the historical
    ``feature_enabled: true`` spellings are rejected.
    """

    violations: List[str] = []
    if isinstance(value, dict):
        for raw_key, child in value.items():
            key = str(raw_key)
            path = _join_path(prefix, key)
            if key in _HARDCORE_RECURSIVE_FEATURE_NAMES:
                if child is True or (
                    isinstance(child, dict) and child.get("enabled", False) is True
                ):
                    violations.append(path)
            elif key.endswith("_enabled"):
                feature = key[: -len("_enabled")]
                if feature in _HARDCORE_RECURSIVE_FEATURE_NAMES and child is True:
                    violations.append(path)
            violations.extend(
                _recursively_enabled_forbidden_features(child, prefix=path)
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            violations.extend(
                _recursively_enabled_forbidden_features(
                    child,
                    prefix=f"{prefix}[{index}]",
                )
            )
    return violations


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
        if prefix == "generic_island_model.archive_seeding.assignments":
            # Island names are materialized dynamically from the immutable run
            # request. Row structure is validated by validate_config().
            return []
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


def _resolve_preset_chain(
    config: Dict[str, Any],
    *,
    seen: Tuple[str, ...],
) -> Dict[str, Any]:
    """Resolve inherited presets and reject recursive inheritance.

    Each child config overrides its preset, which in turn overrides any
    parent preset. Defaults are merged later by the normal config resolver.
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
    if preset_name in seen:
        chain = " -> ".join((*seen, preset_name))
        raise ValueError(f"Recursive preset inheritance: {chain}")

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

    resolved_preset = _resolve_preset_chain(
        preset_data,
        seen=(*seen, preset_name),
    )
    # deepest preset overrides defaults, then each child overrides its parent
    merged = deep_merge(resolved_preset, config)
    return merged


def resolve_preset(config: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve all inherited presets into one detached config mapping."""

    return _resolve_preset_chain(config, seen=())


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

    # Keep configuration acceptance aligned with the runtime implementation.
    # This prevents expensive queued runs from failing only after startup.
    try:
        validate_walk_forward_config(_nested_config(config, ("walk_forward",)))
    except ValueError as exc:
        errors.append(f"invalid walk_forward configuration: {exc}")

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
    automation = _nested_config(config, ("automation_controller",))
    root_seeds = automation.get("root_seeds", [])
    if (
        not root_seeds
        or any(type(seed) is not int for seed in root_seeds)
        or root_seeds != sorted(set(root_seeds))
    ):
        errors.append(
            "automation_controller.root_seeds must be a non-empty, sorted list "
            "of unique integers"
        )
    max_waves = automation.get("max_waves")
    if type(max_waves) is not int or max_waves < 1:
        errors.append("automation_controller.max_waves must be a positive integer")
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

    indicators = _nested_config(config, ("indicators",))
    initial_or_probability = indicators.get("initial_or_probability", 0.0)
    if (
        not isinstance(initial_or_probability, (int, float))
        or isinstance(initial_or_probability, bool)
        or not math.isfinite(initial_or_probability)
        or not 0.0 <= float(initial_or_probability) <= 1.0
    ):
        errors.append("indicators.initial_or_probability must be between 0 and 1")

    pair_validation = _nested_config(config, ("pair_validation",))
    if pair_validation.get("enabled", False):
        evaluation_mode = pair_validation.get("evaluation_mode", "joint")
        if evaluation_mode not in {"joint", "independent_pairs"}:
            errors.append(
                "pair_validation.evaluation_mode must be joint or independent_pairs"
            )
        for key in (
            "worst_pair_weight",
            "worst_split_weight",
            "min_profitable_pair_ratio",
            "profitable_pair_penalty_floor",
            "worst_pair_loss_penalty_floor",
        ):
            value = pair_validation.get(key, 0.0)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or not 0.0 <= float(value) <= 1.0
            ):
                errors.append(f"pair_validation.{key} must be between 0 and 1")
        max_pair_loss_pct = pair_validation.get("max_pair_loss_pct", 0.0)
        if (
            not isinstance(max_pair_loss_pct, (int, float))
            or isinstance(max_pair_loss_pct, bool)
            or not math.isfinite(max_pair_loss_pct)
            or max_pair_loss_pct < 0.0
        ):
            errors.append(
                "pair_validation.max_pair_loss_pct must be a finite non-negative number"
            )

    raw_score = _nested_config(config, ("raw_multipair_score",))
    if raw_score.get("enabled", False):
        if raw_score.get("policy_version") != "raw-multipair-score-v4":
            errors.append(
                "raw_multipair_score.policy_version must be raw-multipair-score-v4"
            )
        development_pairs = raw_score.get("development_pairs", [])
        validation_pairs = raw_score.get("validation_pairs", [])
        if (
            not isinstance(development_pairs, list)
            or len(development_pairs) != 3
            or any(not isinstance(pair, str) or not pair for pair in development_pairs)
            or len(set(development_pairs)) != 3
        ):
            errors.append(
                "raw_multipair_score.development_pairs must contain three unique pairs"
            )
        if (
            not isinstance(validation_pairs, list)
            or len(validation_pairs) != 3
            or any(not isinstance(pair, str) or not pair for pair in validation_pairs)
            or len(set(validation_pairs)) != 3
        ):
            errors.append(
                "raw_multipair_score.validation_pairs must contain three unique pairs"
            )
        if (
            isinstance(development_pairs, list)
            and isinstance(validation_pairs, list)
            and all(isinstance(pair, str) for pair in development_pairs)
            and all(isinstance(pair, str) for pair in validation_pairs)
        ):
            overlap = set(development_pairs) & set(validation_pairs)
            if overlap:
                errors.append(
                    "raw_multipair_score pair groups must be disjoint: "
                    + ", ".join(sorted(overlap))
                )
        for key in ("period_start", "period_end"):
            raw_value = raw_score.get(key)
            try:
                parsed = date.fromisoformat(raw_value) if isinstance(raw_value, str) else None
            except ValueError:
                parsed = None
            if parsed is None:
                errors.append(f"raw_multipair_score.{key} must be an ISO date")
        try:
            if date.fromisoformat(str(raw_score.get("period_end"))) < date.fromisoformat(
                str(raw_score.get("period_start"))
            ):
                errors.append(
                    "raw_multipair_score.period_end must not precede period_start"
                )
        except ValueError:
            pass
        try:
            from genetic_algorithm.evaluation.raw_multipair_score import (
                RawMultiPairPolicyV4,
            )

            RawMultiPairPolicyV4.from_mapping(raw_score.get("policy"))
        except (TypeError, ValueError) as exc:
            errors.append(f"invalid raw_multipair_score.policy: {exc}")

    qualification_v3 = _nested_config(config, ("qualification_v3",))
    if qualification_v3.get("enabled", False):
        try:
            from genetic_algorithm.orchestration.promotion_policy_v3 import (
                qualification_policy_v3_from_config,
            )

            qualification_policy_v3_from_config(config)
        except ValueError as exc:
            errors.append(f"invalid qualification_v3: {exc}")

    fitness_policy_v3 = _nested_config(config, ("fitness_policy_v3",))
    if fitness_policy_v3.get("enabled", False):
        try:
            from genetic_algorithm.evaluation.fitness_policy_v3 import (
                fitness_policy_v3_from_config,
            )

            fitness_policy_v3_from_config(config)
        except ValueError as exc:
            errors.append(f"invalid fitness_policy_v3: {exc}")

    common_replay = _nested_config(
        config, ("generic_island_model", "common_panel_replay")
    )
    if common_replay.get("enabled", False):
        for key in (
            "interval",
            "top_n_per_island",
            "early_stop_patience",
            "min_generation",
        ):
            value = common_replay.get(key, 1) if key == "min_generation" else common_replay.get(key)
            if type(value) is not int or value < 1:
                errors.append(
                    f"generic_island_model.common_panel_replay.{key} "
                    "must be a positive integer"
                )
        min_improvement = common_replay.get("min_improvement")
        if (
            isinstance(min_improvement, bool)
            or not isinstance(min_improvement, (int, float))
            or not math.isfinite(float(min_improvement))
            or float(min_improvement) < 0
        ):
            errors.append(
                "generic_island_model.common_panel_replay.min_improvement "
                "must be a finite non-negative number"
            )
        relative_min_improvement = common_replay.get(
            "relative_min_improvement",
            0.0,
        )
        if (
            isinstance(relative_min_improvement, bool)
            or not isinstance(relative_min_improvement, (int, float))
            or not math.isfinite(float(relative_min_improvement))
            or float(relative_min_improvement) < 0
        ):
            errors.append(
                "generic_island_model.common_panel_replay."
                "relative_min_improvement must be a finite non-negative number"
            )
        incumbent_score = common_replay.get("incumbent_score")
        if incumbent_score is not None and (
            isinstance(incumbent_score, bool)
            or not isinstance(incumbent_score, (int, float))
            or not math.isfinite(float(incumbent_score))
        ):
            errors.append(
                "generic_island_model.common_panel_replay.incumbent_score "
                "must be null or a finite number"
            )
    diversity_recovery = _nested_config(
        config,
        ("generic_island_model", "diversity_recovery"),
    )
    if diversity_recovery.get("enabled", False):
        for key in (
            "island_threshold_count",
            "consecutive_checks_before_recovery",
            "consecutive_checks_after_recovery",
        ):
            value = diversity_recovery.get(key)
            if type(value) is not int or value < 1:
                errors.append(
                    f"generic_island_model.diversity_recovery.{key} "
                    "must be a positive integer"
                )
        for key in (
            "duplicate_fraction_threshold",
            "genetic_diversity_threshold",
            "replacement_fraction",
            "recovery_mutation_rate",
        ):
            value = diversity_recovery.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                errors.append(
                    f"generic_island_model.diversity_recovery.{key} "
                    "must be between 0 and 1"
                )

    archive_seeding = _nested_config(
        config,
        ("generic_island_model", "archive_seeding"),
    )
    if archive_seeding.get("enabled", False):
        island_names = archive_seeding.get("island_names", [])
        if (
            not isinstance(island_names, list)
            or not island_names
            or any(not isinstance(name, str) or not name for name in island_names)
            or len(island_names) != len(set(island_names))
        ):
            errors.append(
                "generic_island_model.archive_seeding.island_names must be a "
                "non-empty list of unique names"
            )
        max_seeds = archive_seeding.get("max_seeds_per_island")
        if type(max_seeds) is not int or max_seeds < 1:
            errors.append(
                "generic_island_model.archive_seeding.max_seeds_per_island "
                "must be a positive integer"
            )
        assignments = archive_seeding.get("assignments", {})
        if not isinstance(assignments, dict) or any(
            name not in island_names or not isinstance(rows, list)
            for name, rows in (
                assignments.items() if isinstance(assignments, dict) else []
            )
        ):
            errors.append(
                "generic_island_model.archive_seeding.assignments must map "
                "configured archive islands to seed lists"
            )
        elif any(
            not isinstance(row, dict)
            or set(row) != {"candidate_id", "niche"}
            or not isinstance(row.get("candidate_id"), str)
            or row.get("niche") not in {"EDGE", "ACTIVITY", "BALANCED"}
            for rows in assignments.values()
            for row in rows
        ):
            errors.append(
                "archive seed assignments require candidate_id and a valid niche"
            )
        bridge_count = archive_seeding.get(
            "cross_niche_offspring_per_generation", 0
        )
        if type(bridge_count) is not int or not 0 <= bridge_count <= 4:
            errors.append(
                "archive_seeding.cross_niche_offspring_per_generation must be 0..4"
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
        if pair_validation.get("evaluation_mode") != "independent_pairs":
            errors.append(
                "automation_island_v2 requires "
                "pair_validation.evaluation_mode: independent_pairs"
            )
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
    if safety.get("name") == "hardcore_multipair_v1" and safety.get(
        "enforce",
        True,
    ):
        label = "hardcore_multipair_v1"
        if config.get("config_schema_version") != 2:
            errors.append(f"{label} requires config_schema_version: 2")
        if ga.get("mode") != "single_objective":
            errors.append(f"{label} requires genetic_algorithm.mode: single_objective")
        if not safety.get("shadow_mode", False):
            errors.append(f"{label} requires shadow_mode: true")
        if safety.get("automation_eligible", False):
            errors.append(f"{label} forbids automation_eligible: true")
        if max_runtime_minutes is None or max_runtime_minutes > 540:
            errors.append(
                f"{label} requires max_runtime_minutes <= 540 to reserve "
                "watchdog finalization headroom"
            )
        if _nested_config(config, ("storage",)).get("checkpoint_interval") != 1:
            errors.append(f"{label} requires a checkpoint after every generation")
        if ga.get("allow_self_crossover", True):
            errors.append(f"{label} requires allow_self_crossover: false")
        if ga.get("random_immigrants") != 2:
            errors.append(f"{label} requires exactly two random immigrants")
        if ga.get("seed_known_archetypes", True):
            errors.append(
                f"{label} requires fresh islands without built-in archetype seeds"
            )
        if not ga.get("fitness_sharing", False):
            errors.append(f"{label} requires fitness sharing")

        for path in _HARDCORE_MULTIPAIR_V1_DISABLED_FEATURES:
            feature = _nested_config(config, path)
            if feature.get("enabled", False):
                errors.append(f"{label} forbids {'.'.join(path)}.enabled: true")

        generic_island = _nested_config(config, ("generic_island_model",))
        if not generic_island.get("enabled", False):
            errors.append(f"{label} requires generic_island_model.enabled: true")
        if generic_island.get("parallel_islands", False):
            errors.append(f"{label} requires one shared worker pool")
        if generic_island.get("num_islands") != 12:
            errors.append(f"{label} requires generic_island_model.num_islands: 12")
        islands = generic_island.get("islands", [])
        if not isinstance(islands, list) or len(islands) != 12:
            errors.append(f"{label} requires exactly twelve explicit islands")
            islands = []
        island_names = [
            island.get("name") for island in islands if isinstance(island, dict)
        ]
        if len(island_names) != len(set(island_names)) or any(
            not isinstance(name, str) or not name for name in island_names
        ):
            errors.append(f"{label} requires twelve unique island names")
        if not safety.get("canary", False):
            if ga.get("population_size") != 12 or ga.get("generations") != 30:
                errors.append(f"{label} production GA requires population 12 x 30 generations")
            if generic_island.get("population_per_island") != 12:
                errors.append(f"{label} production islands require population size 12")
            if generic_island.get("generations") != 30:
                errors.append(f"{label} production islands require 30 generations")
            for index, island in enumerate(islands):
                if not isinstance(island, dict):
                    continue
                if island.get("population_size") != 12:
                    errors.append(f"{label} island[{index}] requires population_size: 12")
                if island.get("generations") != 30:
                    errors.append(f"{label} island[{index}] requires generations: 30")

        recursive_violations = sorted(
            set(
                _recursively_enabled_forbidden_features(
                    generic_island,
                    prefix="generic_island_model",
                )
            )
        )
        for path in recursive_violations:
            errors.append(f"{label} recursively forbids enabled feature {path}")

        expected_dev = ["BTC/USDT", "SOL/USDT", "XRP/USDT"]
        expected_val = ["BNB/USDT", "ETH/USDT", "PEPE/USDT"]
        expected_pairs = set(expected_dev) | set(expected_val)
        raw_score = _nested_config(config, ("raw_multipair_score",))
        if not raw_score.get("enabled", False):
            errors.append(f"{label} requires raw_multipair_score.enabled: true")
        if raw_score.get("policy_version") != "raw-multipair-score-v4":
            errors.append(f"{label} requires raw-multipair-score-v4")
        if raw_score.get("development_pairs") != expected_dev:
            errors.append(f"{label} requires the fixed development pair group")
        if raw_score.get("validation_pairs") != expected_val:
            errors.append(f"{label} requires the fixed validation pair group")
        if raw_score.get("period_start") != "2023-05-09":
            errors.append(f"{label} requires period_start: 2023-05-09")
        if raw_score.get("period_end") != "2026-03-26":
            errors.append(f"{label} requires period_end: 2026-03-26")

        if set(bt.get("pairs", [])) != expected_pairs or len(bt.get("pairs", [])) != 6:
            errors.append(f"{label} requires the fixed six-pair panel")
        if bt.get("timeframe") not in {"15m", "1h"}:
            errors.append(f"{label} permits only 15m or 1h")
        expected_timerange = {
            "15m": "1683590400-1774568700",
            "1h": "1683590400-1774566000",
        }.get(bt.get("timeframe"))
        if bt.get("timerange") != expected_timerange:
            errors.append(
                f"{label} requires the exact inclusive 2023-05-09 through "
                f"2026-03-26 {bt.get('timeframe')} candle timerange"
            )
        if bt.get("fee") != 0.001 or bt.get("slippage_pct") != 0.0005:
            errors.append(f"{label} requires deterministic 0.1% fee and 0.05% slippage")
        if bt.get("fee_noise_std", 0.0) != 0.0:
            errors.append(f"{label} requires fee_noise_std: 0.0")
        if bt.get("auto_download_data", False):
            errors.append(f"{label} forbids automatic data downloads")
        constraints = _nested_config(config, ("strategy_constraints",))
        if (
            constraints.get("startup_candle_floor") != 75
            or constraints.get("startup_candle_cap") != 75
        ):
            errors.append(
                f"{label} requires the proven 75-candle startup floor/cap"
            )
        if constraints.get("canonicalize_executable_genome") is not True:
            errors.append(f"{label} requires executable-genome canonicalization")
        if indicators.get("min_entry_conditions") != 1:
            errors.append(f"{label} requires min_entry_conditions: 1")
        if indicators.get("initial_or_probability") != 0.5:
            errors.append(f"{label} requires initial_or_probability: 0.5")
        roi_range = constraints.get("roi_range")
        if (
            not isinstance(roi_range, list)
            or len(roi_range) != 2
            or roi_range[0] > 0.001
            or roi_range[1] < 0.10
        ):
            errors.append(f"{label} requires the broad high-turnover ROI search range")

        promotion = _nested_config(config, ("promotion_v2",))
        scenarios = promotion.get("required_scenarios", [])
        scenario_rows = [row for row in scenarios if isinstance(row, dict)]
        expected_roles = {
            **{pair: "TRAIN" for pair in expected_dev},
            **{pair: "PAIR_VALIDATION" for pair in expected_val},
        }
        if not promotion.get("enabled", False) or len(scenario_rows) != 6:
            errors.append(f"{label} requires one replay manifest row per panel pair")
        elif {
            (row.get("pair"), row.get("timeframe")) for row in scenario_rows
        } != {(pair, bt.get("timeframe")) for pair in expected_pairs}:
            errors.append(f"{label} replay manifest must cover the exact timeframe panel")
        for row in scenario_rows:
            pair = row.get("pair")
            if (
                row.get("period_start") != "2023-05-09"
                or row.get("period_end") != "2026-03-26"
                or row.get("cost_multiplier") != 1.0
                or row.get("role") != expected_roles.get(pair)
            ):
                errors.append(
                    f"{label} replay manifest row for {pair!r} differs from "
                    "the fixed raw panel"
                )

        pair_validation = _nested_config(config, ("pair_validation",))
        if not pair_validation.get("enabled", False):
            errors.append(f"{label} requires pair_validation.enabled: true")
        if pair_validation.get("evaluation_mode") != "independent_pairs":
            errors.append(f"{label} requires independent pair evaluation")
        if pair_validation.get("validate_top_n_only") != 0:
            errors.append(f"{label} requires validate_top_n_only: 0")
        if pair_validation.get("training_pairs") != expected_dev:
            errors.append(f"{label} requires the fixed development pair split")
        if pair_validation.get("validation_pairs") != expected_val:
            errors.append(f"{label} requires the fixed validation pair split")
        specialization = generic_island.get("specialization", {})
        if specialization.get("pair_rotation", False):
            errors.append(f"{label} forbids pair rotation")
        for index, island in enumerate(islands):
            if not isinstance(island, dict):
                continue
            if set(island.get("pairs", [])) != expected_pairs:
                errors.append(f"{label} island[{index}] must see all six pairs")
            if island.get("walk_forward_enabled", False):
                errors.append(f"{label} island[{index}] forbids walk forward")

        migration = generic_island.get("migration", {})
        if migration.get("topology") != "ring":
            errors.append(f"{label} requires ring migration")
        if migration.get("interval") != 5 or migration.get("count") != 1:
            errors.append(f"{label} requires one migrant every five generations")
        if migration.get("merge_rounds", False):
            errors.append(f"{label} forbids merge rounds")
        common_replay = generic_island.get("common_panel_replay", {})
        if not common_replay.get("enabled", False):
            errors.append(f"{label} requires common-panel replay")
        if common_replay.get("interval") != 3:
            errors.append(f"{label} requires common-panel interval: 3")
        if common_replay.get("top_n_per_island") != 2:
            errors.append(f"{label} requires top two candidates per island")
        if common_replay.get("min_generation") != 9:
            errors.append(f"{label} requires no economic stop before generation 9")
        if common_replay.get("early_stop_patience") != 4:
            errors.append(f"{label} requires four-check plateau patience")
        if common_replay.get("min_improvement") != 0.25:
            errors.append(f"{label} requires absolute material improvement 0.25")
        if common_replay.get("relative_min_improvement") != 0.005:
            errors.append(f"{label} requires relative material improvement 0.005")

        archive_seeding = generic_island.get("archive_seeding", {})
        archive_names = archive_seeding.get("island_names", [])
        if not archive_seeding.get("enabled", False) or len(archive_names) != 3:
            errors.append(f"{label} requires exactly three archive islands")
        if not set(archive_names).issubset(set(island_names)):
            errors.append(f"{label} archive island names must exist")
        if archive_seeding.get("max_seeds_per_island") != 4:
            errors.append(f"{label} allows at most four archive seeds per island")
        diversity_recovery = generic_island.get("diversity_recovery", {})
        expected_diversity = {
            "enabled": True,
            "island_threshold_count": 9,
            "consecutive_checks_before_recovery": 2,
            "consecutive_checks_after_recovery": 2,
            "duplicate_fraction_threshold": 0.75,
            "genetic_diversity_threshold": 0.10,
            "replacement_fraction": 0.25,
            "recovery_mutation_rate": 0.35,
        }
        for key, expected in expected_diversity.items():
            if diversity_recovery.get(key) != expected:
                errors.append(
                    f"{label} requires diversity_recovery.{key}: {expected}"
                )
    if (
        safety.get("name") == "hardcore_multipair_child_v1"
        and safety.get("enforce", True)
    ):
        label = "hardcore_multipair_child_v1"
        expected_dev = ["BTC/USDT", "SOL/USDT", "XRP/USDT"]
        expected_val = ["BNB/USDT", "ETH/USDT", "PEPE/USDT"]
        expected_pairs = set(expected_dev) | set(expected_val)
        if config.get("config_schema_version") != 2:
            errors.append(f"{label} requires config_schema_version: 2")
        if ga.get("mode") != "single_objective":
            errors.append(f"{label} requires genetic_algorithm.mode: single_objective")
        if not safety.get("shadow_mode", False):
            errors.append(f"{label} requires shadow_mode: true")
        if safety.get("automation_eligible", False):
            errors.append(f"{label} forbids automation_eligible: true")
        if max_runtime_minutes is None or max_runtime_minutes > 540:
            errors.append(f"{label} requires max_runtime_minutes <= 540")
        if ga.get("allow_self_crossover", True):
            errors.append(f"{label} requires allow_self_crossover: false")
        if not ga.get("fitness_sharing", False):
            errors.append(f"{label} requires fitness sharing")
        expected_immigrants = 2
        if ga.get("random_immigrants") != expected_immigrants:
            errors.append(
                f"{label} requires exactly {expected_immigrants} random immigrants"
            )

        for path in _HARDCORE_MULTIPAIR_V1_DISABLED_FEATURES:
            feature = _nested_config(config, path)
            if feature.get("enabled", False):
                errors.append(f"{label} forbids {'.'.join(path)}.enabled: true")
        generic_island = _nested_config(config, ("generic_island_model",))
        if generic_island.get("enabled", False):
            errors.append(f"{label} forbids recursive generic_island_model.enabled: true")
        for path in sorted(
            set(
                _recursively_enabled_forbidden_features(
                    generic_island,
                    prefix="generic_island_model",
                )
            )
        ):
            errors.append(f"{label} recursively forbids enabled feature {path}")
        if _nested_config(config, ("parallel_evaluation",)).get("enabled", False):
            errors.append(f"{label} requires the coordinator's shared worker pool")

        raw_score = _nested_config(config, ("raw_multipair_score",))
        if (
            not raw_score.get("enabled", False)
            or raw_score.get("policy_version") != "raw-multipair-score-v4"
            or raw_score.get("development_pairs") != expected_dev
            or raw_score.get("validation_pairs") != expected_val
            or raw_score.get("period_start") != "2023-05-09"
            or raw_score.get("period_end") != "2026-03-26"
        ):
            errors.append(f"{label} requires the immutable raw-multipair-score-v4 panel")

        timeframe = bt.get("timeframe")
        expected_timerange = {
            "15m": "1683590400-1774568700",
            "1h": "1683590400-1774566000",
        }.get(timeframe)
        if set(bt.get("pairs", [])) != expected_pairs or len(bt.get("pairs", [])) != 6:
            errors.append(f"{label} requires the fixed six-pair panel")
        if expected_timerange is None or bt.get("timerange") != expected_timerange:
            errors.append(f"{label} requires the exact timeframe panel timerange")
        if bt.get("fee") != 0.001 or bt.get("slippage_pct") != 0.0005:
            errors.append(f"{label} requires deterministic fee and slippage")
        if bt.get("fee_noise_std", 0.0) != 0.0:
            errors.append(f"{label} requires fee_noise_std: 0.0")
        if bt.get("auto_download_data", False):
            errors.append(f"{label} forbids automatic data downloads")
        constraints = _nested_config(config, ("strategy_constraints",))
        if (
            constraints.get("startup_candle_floor") != 75
            or constraints.get("startup_candle_cap") != 75
        ):
            errors.append(f"{label} requires the proven 75-candle startup bound")
        if constraints.get("canonicalize_executable_genome") is not True:
            errors.append(f"{label} requires executable-genome canonicalization")

        pair_validation = _nested_config(config, ("pair_validation",))
        if not pair_validation.get("enabled", False):
            errors.append(f"{label} requires pair_validation.enabled: true")
        if pair_validation.get("evaluation_mode") != "independent_pairs":
            errors.append(f"{label} requires independent pair evaluation")
        if pair_validation.get("validate_top_n_only") != 0:
            errors.append(f"{label} requires validate_top_n_only: 0")
        if pair_validation.get("training_pairs") != expected_dev:
            errors.append(f"{label} requires the fixed development pair split")
        if pair_validation.get("validation_pairs") != expected_val:
            errors.append(f"{label} requires the fixed validation pair split")
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
        if pair_validation.get("evaluation_mode") != "independent_pairs":
            errors.append(
                "automation_island_child_v2 requires "
                "pair_validation.evaluation_mode: independent_pairs"
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
