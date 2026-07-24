"""Versioned mechanical invariants for GA runtime configurations.

This module deliberately does not encode claims about which tuning values are
profitable.  Population size, pair count, mutation rate, and similar search
choices need paired out-of-sample experiments.  A config validator may only
block values that are structurally impossible, ignored by the implementation,
or internally contradictory.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import datetime
from typing import Any


CONFIG_INVARIANT_POLICY_VERSION = "ga-config-invariants-v1"

# These are implementation/resource guards, not claims of optimality.
MIN_POPULATION_SIZE = 2
MAX_POPULATION_SIZE = 10_000
MAX_GENERATIONS = 100_000

SELECTION_METHODS = frozenset({"tournament", "roulette", "rank", "nsga2"})
CROSSOVER_METHODS = frozenset({"single_point", "uniform", "component"})
GA_MODES = frozenset({"single_objective", "nsga2"})
MIGRATION_TOPOLOGIES = frozenset(
    {"ring", "fully_connected", "tournament", "hierarchical"}
)


def derive_island_population_slots(population_size: int) -> tuple[int, int]:
    """Return feasible elite and immigrant counts for a derived island config."""

    if not _is_int(population_size) or population_size < MIN_POPULATION_SIZE:
        raise ValueError(
            f"island population_size must be an integer >= {MIN_POPULATION_SIZE}"
        )
    target = max(2, population_size // 10)
    elite_size = min(target, population_size - 1)
    random_immigrants = min(target, population_size - elite_size)
    return elite_size, random_immigrants


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any] | None:
    value = config.get(key)
    return value if isinstance(value, Mapping) else None


def _require_int(
    section: Mapping[str, Any],
    key: str,
    *,
    path: str,
    minimum: int,
    maximum: int | None,
    errors: list[str],
) -> int | None:
    value = section.get(key)
    if not _is_int(value):
        errors.append(f"{path} must be an integer")
        return None
    if value < minimum or (maximum is not None and value > maximum):
        bound = f">= {minimum}" if maximum is None else f"between {minimum} and {maximum}"
        errors.append(f"{path} must be {bound}, got {value}")
        return None
    return value


def _require_probability(
    section: Mapping[str, Any],
    key: str,
    *,
    path: str,
    errors: list[str],
) -> float | None:
    value = section.get(key)
    if not _is_finite_number(value) or not 0.0 <= float(value) <= 1.0:
        errors.append(f"{path} must be a finite number between 0 and 1")
        return None
    return float(value)


def _validate_pair_list(value: Any, path: str, errors: list[str]) -> list[str] | None:
    if not isinstance(value, list) or not value:
        errors.append(f"{path} must be a non-empty list")
        return None
    if any(not isinstance(pair, str) or not pair.strip() for pair in value):
        errors.append(f"{path} must contain non-empty strings")
        return None
    if len(value) != len(set(value)):
        errors.append(f"{path} contains duplicate pairs")
        return None
    return value


def _validate_ga(
    ga: Mapping[str, Any],
    errors: list[str],
    warnings: list[str],
) -> None:
    population = _require_int(
        ga,
        "population_size",
        path="genetic_algorithm.population_size",
        minimum=MIN_POPULATION_SIZE,
        maximum=MAX_POPULATION_SIZE,
        errors=errors,
    )
    _require_int(
        ga,
        "generations",
        path="genetic_algorithm.generations",
        minimum=1,
        maximum=MAX_GENERATIONS,
        errors=errors,
    )
    elite = _require_int(
        ga,
        "elite_size",
        path="genetic_algorithm.elite_size",
        minimum=0,
        maximum=None,
        errors=errors,
    )
    tournament = _require_int(
        ga,
        "tournament_size",
        path="genetic_algorithm.tournament_size",
        minimum=1,
        maximum=None,
        errors=errors,
    )
    immigrants = _require_int(
        ga,
        "random_immigrants",
        path="genetic_algorithm.random_immigrants",
        minimum=0,
        maximum=None,
        errors=errors,
    )

    if population is not None:
        if elite is not None and elite >= population:
            errors.append(
                "genetic_algorithm.elite_size must be smaller than "
                "genetic_algorithm.population_size"
            )
        if tournament is not None and tournament > population:
            errors.append(
                "genetic_algorithm.tournament_size must not exceed "
                "genetic_algorithm.population_size"
            )
        if (
            elite is not None
            and immigrants is not None
            and immigrants > population - elite
        ):
            errors.append(
                "genetic_algorithm.random_immigrants exceeds the non-elite "
                "population slots"
            )

    mutation = _require_probability(
        ga,
        "mutation_rate",
        path="genetic_algorithm.mutation_rate",
        errors=errors,
    )
    max_mutation = _require_probability(
        ga,
        "max_mutation_rate",
        path="genetic_algorithm.max_mutation_rate",
        errors=errors,
    )
    _require_probability(
        ga,
        "crossover_rate",
        path="genetic_algorithm.crossover_rate",
        errors=errors,
    )
    if (
        mutation is not None
        and max_mutation is not None
        and mutation > max_mutation
    ):
        errors.append(
            "genetic_algorithm.mutation_rate must not exceed "
            "genetic_algorithm.max_mutation_rate"
        )

    max_factor = ga.get("max_adaptation_factor")
    if not _is_finite_number(max_factor) or float(max_factor) < 1.0:
        errors.append(
            "genetic_algorithm.max_adaptation_factor must be a finite number >= 1"
        )
    adaptation_step = ga.get("adaptation_step")
    if not _is_finite_number(adaptation_step) or float(adaptation_step) < 0.0:
        errors.append("genetic_algorithm.adaptation_step must be a finite number >= 0")
    cooldown = ga.get("mutation_cooldown_factor")
    if not _is_finite_number(cooldown) or not 0.0 <= float(cooldown) <= 1.0:
        errors.append(
            "genetic_algorithm.mutation_cooldown_factor must be between 0 and 1"
        )

    convergence_patience = ga.get("convergence_patience")
    if not _is_int(convergence_patience) or convergence_patience < 1:
        errors.append("genetic_algorithm.convergence_patience must be an integer >= 1")

    for key in ("sharing_radius", "diversity_threshold", "behavioral_distance_weight"):
        value = ga.get(key)
        if not _is_finite_number(value) or not 0.0 <= float(value) <= 1.0:
            errors.append(f"genetic_algorithm.{key} must be between 0 and 1")

    mode = ga.get("mode")
    if mode not in GA_MODES:
        errors.append(
            "genetic_algorithm.mode must be one of: "
            + ", ".join(sorted(GA_MODES))
        )
    selection = ga.get("selection_method")
    if selection not in SELECTION_METHODS:
        errors.append(
            "genetic_algorithm.selection_method must be one of: "
            + ", ".join(sorted(SELECTION_METHODS))
        )
    crossover = ga.get("crossover_method")
    if crossover not in CROSSOVER_METHODS:
        errors.append(
            "genetic_algorithm.crossover_method must be one of: "
            + ", ".join(sorted(CROSSOVER_METHODS))
        )
    if mode == "nsga2" and selection != "nsga2":
        warnings.append(
            "genetic_algorithm.selection_method is ignored in nsga2 mode; "
            "the engine forces nsga2 selection"
        )
    if mode != "nsga2" and selection == "nsga2":
        errors.append(
            "genetic_algorithm.selection_method=nsga2 requires "
            "genetic_algorithm.mode=nsga2"
        )
    if selection == "tournament" and tournament == 1:
        warnings.append(
            "genetic_algorithm.tournament_size=1 makes tournament parent "
            "selection uniform random; this is valid only when random selection is intended"
        )


def _validate_backtesting(
    backtesting: Mapping[str, Any],
    errors: list[str],
    warnings: list[str],
) -> None:
    _validate_pair_list(backtesting.get("pairs"), "backtesting.pairs", errors)

    timerange = backtesting.get("timerange")
    if timerange == "":
        warnings.append("backtesting.timerange is empty; all available data will be used")
    elif not isinstance(timerange, str):
        errors.append("backtesting.timerange must be a string")
    else:
        try:
            start_raw, end_raw = timerange.split("-", maxsplit=1)
            if (
                start_raw.isdigit()
                and end_raw.isdigit()
                and len(start_raw) in {9, 10}
                and len(end_raw) in {9, 10}
            ):
                start = int(start_raw)
                end = int(end_raw)
            else:
                start = datetime.strptime(start_raw, "%Y%m%d")
                end = datetime.strptime(end_raw, "%Y%m%d")
            if end <= start:
                errors.append("backtesting.timerange end must be after start")
        except ValueError:
            errors.append(
                "backtesting.timerange must use valid YYYYMMDD dates or epoch seconds"
            )

    for key in ("fee", "slippage_pct", "fee_noise_std"):
        value = backtesting.get(key)
        if not _is_finite_number(value) or float(value) < 0.0:
            errors.append(f"backtesting.{key} must be a finite number >= 0")

    timeout = backtesting.get("timeout")
    if not _is_int(timeout) or timeout < 0:
        errors.append("backtesting.timeout must be an integer >= 0")

    if "datadir" in backtesting:
        warnings.append(
            "backtesting.datadir is not a supported runtime key; data location is "
            "resolved centrally and bound by data_manifest.json"
        )


def _validate_fitness_weights(
    config: Mapping[str, Any],
    errors: list[str],
    warnings: list[str],
) -> None:
    weights = _mapping(config, "fitness_weights")
    if weights is None or not weights:
        errors.append("fitness_weights must be a non-empty mapping")
        return

    # Local constants avoid importing the full evaluator during config parsing.
    valid_keys = {
        "profit",
        "sharpe_ratio",
        "sortino_ratio",
        "profit_factor",
        "drawdown",
        "win_rate",
        "trade_frequency",
        "monthly_stability",
        "cross_pair",
        "drawdown_duration",
        "consecutive_losses",
    }
    aliases = {
        "consistency": "monthly_stability",
        "exposure_time": "cross_pair",
    }
    total = 0.0
    for key, value in weights.items():
        if not _is_finite_number(value) or float(value) < 0.0:
            errors.append(f"fitness_weights.{key} must be a finite number >= 0")
            continue
        total += float(value)
        if key in aliases:
            warnings.append(
                f"fitness_weights.{key} is deprecated; use {aliases[key]}"
            )
        elif key not in valid_keys:
            errors.append(f"fitness_weights.{key} is not a recognized fitness input")
    if total <= 0.0:
        errors.append("fitness_weights must contain at least one positive weight")


def _validate_pair_validation(config: Mapping[str, Any], errors: list[str]) -> None:
    pair_validation = _mapping(config, "pair_validation")
    if pair_validation is None or not pair_validation.get("enabled", False):
        return
    training = _validate_pair_list(
        pair_validation.get("training_pairs"),
        "pair_validation.training_pairs",
        errors,
    )
    validation = _validate_pair_list(
        pair_validation.get("validation_pairs"),
        "pair_validation.validation_pairs",
        errors,
    )
    if training is not None and validation is not None:
        overlap = sorted(set(training).intersection(validation))
        if overlap:
            errors.append(
                "pair_validation training and validation pairs must be disjoint: "
                + ", ".join(overlap)
            )
    train_weight = pair_validation.get("weight_train", 0.6)
    validation_weight = pair_validation.get("weight_val", 0.4)
    if (
        not _is_finite_number(train_weight)
        or not _is_finite_number(validation_weight)
        or float(train_weight) < 0.0
        or float(validation_weight) < 0.0
        or float(train_weight) + float(validation_weight) <= 0.0
    ):
        errors.append(
            "pair_validation weights must be finite, non-negative, and have a positive sum"
        )


def _validate_parallel(config: Mapping[str, Any], errors: list[str]) -> None:
    parallel = _mapping(config, "parallel_evaluation")
    if parallel is None:
        errors.append("parallel_evaluation must be a mapping")
        return
    workers = parallel.get("num_workers")
    if workers is not None and (not _is_int(workers) or workers < 1):
        errors.append("parallel_evaluation.num_workers must be null or an integer >= 1")
    timeout = parallel.get("backtest_timeout")
    if not _is_int(timeout) or timeout < 0:
        errors.append("parallel_evaluation.backtest_timeout must be an integer >= 0")


def _validate_islands(config: Mapping[str, Any], errors: list[str]) -> None:
    generic = _mapping(config, "generic_island_model") or {}
    classic = _mapping(config, "island_model") or {}
    generic_enabled = generic.get("enabled", False)
    classic_enabled = classic.get("enabled", False)
    if generic_enabled and classic_enabled:
        errors.append(
            "generic_island_model and island_model cannot both be enabled"
        )

    num_islands = generic.get("num_islands")
    population = generic.get("population_per_island")
    if not _is_int(num_islands) or num_islands < 1:
        errors.append("generic_island_model.num_islands must be an integer >= 1")
    if (
        not _is_int(population)
        or population < MIN_POPULATION_SIZE
        or population > MAX_POPULATION_SIZE
    ):
        errors.append(
            "generic_island_model.population_per_island must be between "
            f"{MIN_POPULATION_SIZE} and {MAX_POPULATION_SIZE}"
        )
        population = None
    island_generations = generic.get("generations")
    if island_generations is not None and (
        not _is_int(island_generations)
        or not 1 <= island_generations <= MAX_GENERATIONS
    ):
        errors.append(
            "generic_island_model.generations must be null or an integer between "
            f"1 and {MAX_GENERATIONS}"
        )

    migration = generic.get("migration")
    if not isinstance(migration, Mapping):
        errors.append("generic_island_model.migration must be a mapping")
    else:
        topology = migration.get("topology")
        if topology not in MIGRATION_TOPOLOGIES:
            errors.append(
                "generic_island_model.migration.topology must be one of: "
                + ", ".join(sorted(MIGRATION_TOPOLOGIES))
            )
        interval = migration.get("interval")
        count = migration.get("count")
        if not _is_int(interval) or interval < 1:
            errors.append(
                "generic_island_model.migration.interval must be an integer >= 1"
            )
        if not _is_int(count) or count < 0:
            errors.append(
                "generic_island_model.migration.count must be an integer >= 0"
            )
        elif population is not None and count > population:
            errors.append(
                "generic_island_model.migration.count must not exceed "
                "population_per_island"
            )

    # The classic engine overwrites walk-forward to disabled inside each
    # island, so accepting both would persist a setting that is not executed.
    walk_forward = _mapping(config, "walk_forward") or {}
    if classic_enabled and walk_forward.get("enabled", False):
        errors.append(
            "island_model and walk_forward cannot both be enabled because the "
            "classic island engine disables walk-forward"
        )


def validate_runtime_invariants(
    config: Mapping[str, Any],
) -> tuple[list[str], list[str]]:
    """Return deterministic mechanical errors and factual semantic warnings."""

    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(config, Mapping):
        return ["config root must be a mapping"], []

    ga = _mapping(config, "genetic_algorithm")
    if ga is None:
        errors.append("genetic_algorithm must be a mapping")
    else:
        _validate_ga(ga, errors, warnings)

    backtesting = _mapping(config, "backtesting")
    if backtesting is None:
        errors.append("backtesting must be a mapping")
    else:
        _validate_backtesting(backtesting, errors, warnings)

    _validate_fitness_weights(config, errors, warnings)
    _validate_pair_validation(config, errors)
    _validate_parallel(config, errors)
    _validate_islands(config, errors)

    return sorted(set(errors)), sorted(set(warnings))
