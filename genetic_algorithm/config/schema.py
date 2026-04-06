"""
Config schema — sensible defaults, validation, and preset merging.

Design:
  - Every GA config key has a documented default in ``DEFAULTS``.
  - Users only need to specify what they want to *change*.
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
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults — every supported key with its default value
# ---------------------------------------------------------------------------

DEFAULTS: Dict[str, Any] = {
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
    },

    # === Genetic Algorithm ===
    "genetic_algorithm": {
        "random_seed": None,
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
    },

    # === Strategy constraints ===
    "strategy_constraints": {
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
        "iterations": 50,
    },

    # === Deflated Sharpe ===
    "deflated_sharpe": {
        "enabled": True,
    },

    # === Surrogate model ===
    "surrogate": {
        "enabled": False,
        "min_samples": 50,
        "filter_percentile": 0.50,
    },

    # === Regime detection ===
    "regime_aware": {
        "enabled": False,
        "detection_method": "adx_di_hysteresis",
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
            "rate": 0.2,
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


# ---------------------------------------------------------------------------
# Preset resolution
# ---------------------------------------------------------------------------

_PRESETS_DIR = Path(__file__).parent / "presets"


def resolve_preset(config: Dict[str, Any]) -> Dict[str, Any]:
    """If *config* contains a ``preset`` key, load that preset YAML and
    merge *config* on top of it (config overrides preset overrides defaults).
    """
    preset_name = config.pop("preset", None)
    if not preset_name:
        return config

    preset_file = _PRESETS_DIR / f"{preset_name}.yaml"
    if not preset_file.exists():
        available = [p.stem for p in _PRESETS_DIR.glob("*.yaml")]
        raise FileNotFoundError(
            f"Preset '{preset_name}' not found at {preset_file}. "
            f"Available: {available}"
        )

    with open(preset_file) as fh:
        preset_data = yaml.safe_load(fh) or {}

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

    ga = config.get("genetic_algorithm", {})
    bt = config.get("backtesting", {})
    fw = config.get("fitness_weights", {})

    # --- Critical ---
    if not bt.get("pairs"):
        errors.append("backtesting.pairs is empty")

    pop = ga.get("population_size", 0)
    elite = ga.get("elite_size", 0)
    if pop and elite and elite >= pop:
        errors.append(f"elite_size ({elite}) >= population_size ({pop})")

    # --- Weights ---
    if fw:
        total = sum(fw.values())
        if abs(total - 1.0) > 0.05:
            warnings.append(f"fitness_weights sum to {total:.3f} (expected ~1.0)")

    # --- GA params ---
    mut = ga.get("mutation_rate", 0)
    max_mut = ga.get("max_mutation_rate", 1.0)
    if mut > max_mut:
        warnings.append(f"mutation_rate ({mut}) > max_mutation_rate ({max_mut})")

    tourn = ga.get("tournament_size", 3)
    if pop and tourn > pop // 2:
        warnings.append(f"tournament_size ({tourn}) > half of population ({pop})")

    imm = ga.get("random_immigrants", 0)
    if pop and imm > pop * 0.5:
        warnings.append(f"random_immigrants ({imm}) > 50% of population")

    # --- Island model conflicts ---
    gim = config.get("generic_island_model", {})
    im = config.get("island_model", {})
    if gim.get("enabled") and im.get("enabled"):
        errors.append("Both generic_island_model and island_model are enabled — pick one")

    return errors, warnings


# ---------------------------------------------------------------------------
# Main entry: load_config
# ---------------------------------------------------------------------------

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
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}

    # Resolve preset if specified
    raw = resolve_preset(raw)

    # Merge onto defaults
    config = deep_merge(DEFAULTS, raw)

    # Apply programmatic overrides
    if overrides:
        config = deep_merge(config, overrides)

    # Validate
    errors, warnings_ = validate_config(config)
    for w in warnings_:
        logger.warning("Config warning: %s", w)
    if errors:
        msg = "Config validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ValueError(msg)

    return config
