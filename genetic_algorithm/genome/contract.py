"""Semantic contract for persisted ``strategy-gene-v2`` genomes.

Structural round-tripping alone is insufficient at a persistence boundary:
unknown indicators, misspelled parameter names and invalid risk values can all
round-trip while generating a different or empty strategy.  This module is the
single fail-closed semantic validator used by native artifacts and migrations.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from genetic_algorithm.genome.gene import StrategyGene, is_higher_timeframe
from genetic_algorithm.genome.indicators import (
    CDL_TYPES,
    get_all_indicator_types,
    is_valid_operator,
)


GENOME_SCHEMA_VERSION = "strategy-gene-v2"


class GenomeContractError(ValueError):
    """Raised when a genome is structurally canonical but semantically invalid."""


_TIMEFRAMES = {
    "1m",
    "3m",
    "5m",
    "15m",
    "30m",
    "1h",
    "2h",
    "4h",
    "6h",
    "8h",
    "12h",
    "1d",
    "3d",
    "1w",
    "1M",
}
_KNOWN_INDICATORS = set(get_all_indicator_types()) | set(CDL_TYPES)
_PERIOD_ONLY = {
    "RSI",
    "EMA",
    "SMA",
    "ATR",
    "ADX",
    "CCI",
    "MFI",
    "WILLR",
    "ROC",
    "TEMA",
    "KAMA",
    "AROON",
    "DONCHIAN",
    "VWAP",
    "CMF",
    "VROC",
}
_NO_PARAMETERS = {
    "OBV",
    "CDL_ENGULFING",
    "CDL_HAMMER",
    "CDL_DOJI",
    "CDL_SHOOTINGSTAR",
    "CDL_HARAMI",
    "CDL_PIERCING",
    "CDL_DARKCLOUD",
    "CDL_3WHITESOLDIERS",
    "CDL_3BLACKCROWS",
}
_STAR_PARAMETERS = {"CDL_MORNINGSTAR", "CDL_EVENINGSTAR"}
_PARAMETER_KEYS = {
    **{name: {"period"} for name in _PERIOD_ONLY},
    **{name: set() for name in _NO_PARAMETERS},
    **{name: {"penetration"} for name in _STAR_PARAMETERS},
    "MACD": {"fast_period", "slow_period", "signal_period"},
    "BBANDS": {"period", "std_dev"},
    "STOCH": {"k_period", "d_period"},
    "SAR": {"acceleration", "maximum"},
    "PSAR": {"acceleration", "maximum"},
    "SUPERTREND": {"period", "multiplier"},
    "ICHIMOKU": {"tenkan_period", "kijun_period", "senkou_b_period"},
}
_PARAMETER_DEFAULTS = {
    **{name: {"period": 14} for name in _PERIOD_ONLY},
    **{name: {} for name in _NO_PARAMETERS},
    **{name: {"penetration": 0.0} for name in _STAR_PARAMETERS},
    "MACD": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
    "BBANDS": {"period": 20, "std_dev": 2.0},
    "STOCH": {"k_period": 14, "d_period": 3},
    "SAR": {"acceleration": 0.02, "maximum": 0.2},
    "PSAR": {"acceleration": 0.02, "maximum": 0.2},
    "SUPERTREND": {"period": 10, "multiplier": 3.0},
    "ICHIMOKU": {"tenkan_period": 9, "kijun_period": 26, "senkou_b_period": 52},
}
_INTEGER_PARAMETERS = {
    "period",
    "fast_period",
    "slow_period",
    "signal_period",
    "k_period",
    "d_period",
    "tenkan_period",
    "kijun_period",
    "senkou_b_period",
}


def _finite_number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GenomeContractError(f"{path} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise GenomeContractError(f"{path} must be finite")
    return number


def _integer(
    value: Any,
    path: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise GenomeContractError(f"{path} must be an integer")
    if value < minimum:
        raise GenomeContractError(f"{path} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise GenomeContractError(f"{path} must be <= {maximum}")
    return value


def _validate_parameters(indicator_type: str, parameters: Any, path: str) -> None:
    if not isinstance(parameters, Mapping):
        raise GenomeContractError(f"{path} must be an object")
    expected = _PARAMETER_KEYS[indicator_type]
    actual = set(parameters)
    if not actual <= expected:
        raise GenomeContractError(
            f"{path} keys for {indicator_type} must be a subset of "
            f"{sorted(expected)}, got {sorted(actual)}"
        )
    for name, value in parameters.items():
        parameter_path = f"{path}.{name}"
        if name in _INTEGER_PARAMETERS:
            _integer(value, parameter_path, minimum=1, maximum=10_000)
        else:
            _finite_number(value, parameter_path)

    resolved = {**_PARAMETER_DEFAULTS[indicator_type], **parameters}
    if indicator_type == "MACD":
        if resolved["fast_period"] >= resolved["slow_period"]:
            raise GenomeContractError(f"{path} requires fast_period < slow_period")
    elif indicator_type == "BBANDS":
        if not 0.0 < resolved["std_dev"] <= 100.0:
            raise GenomeContractError(f"{path}.std_dev must be in (0, 100]")
    elif indicator_type in {"SAR", "PSAR"}:
        if not 0.0 < resolved["acceleration"] <= 1.0:
            raise GenomeContractError(f"{path}.acceleration must be in (0, 1]")
        if resolved["maximum"] > 10.0:
            raise GenomeContractError(f"{path}.maximum must be <= 10")
        if resolved["maximum"] < resolved["acceleration"]:
            raise GenomeContractError(f"{path} requires maximum >= acceleration")
    elif indicator_type == "SUPERTREND":
        if not 0.0 < resolved["multiplier"] <= 100.0:
            raise GenomeContractError(f"{path}.multiplier must be in (0, 100]")
    elif indicator_type == "ICHIMOKU":
        if not (resolved["tenkan_period"] < resolved["kijun_period"] < resolved["senkou_b_period"]):
            raise GenomeContractError(
                f"{path} requires tenkan_period < kijun_period < senkou_b_period"
            )
    elif indicator_type in _STAR_PARAMETERS:
        penetration = resolved["penetration"]
        if not 0.0 <= penetration <= 1.0:
            raise GenomeContractError(f"{path}.penetration must be in [0, 1]")


def _validate_parameter_bounds(
    bounds: Any,
    parameters: Mapping[str, Any],
    path: str,
) -> None:
    if bounds is None:
        return
    if not isinstance(bounds, Mapping) or set(bounds) != set(parameters):
        raise GenomeContractError(
            f"{path} must be null or define exactly the indicator parameter keys"
        )
    for name, interval in bounds.items():
        interval_path = f"{path}.{name}"
        if not isinstance(interval, (list, tuple)) or len(interval) != 2:
            raise GenomeContractError(f"{interval_path} must be [minimum, maximum]")
        lower = _finite_number(interval[0], f"{interval_path}[0]")
        upper = _finite_number(interval[1], f"{interval_path}[1]")
        value = _finite_number(parameters[name], f"parameters.{name}")
        if lower > upper or not lower <= value <= upper:
            raise GenomeContractError(
                f"{interval_path} must be ordered and contain the current value"
            )


def validate_strategy_gene_semantics(gene: StrategyGene) -> None:
    """Validate executable semantics for a canonical persisted genome."""

    _integer(gene.generation, "generation", minimum=0)
    _integer(gene.individual_id, "individual_id", minimum=0)
    if gene.timeframe not in _TIMEFRAMES:
        raise GenomeContractError(f"unknown base timeframe: {gene.timeframe!r}")
    if len(gene.informative_timeframes) != len(set(gene.informative_timeframes)):
        raise GenomeContractError("informative_timeframes contains duplicates")
    for timeframe in gene.informative_timeframes:
        if timeframe not in _TIMEFRAMES or not is_higher_timeframe(timeframe, gene.timeframe):
            raise GenomeContractError(
                f"informative timeframe {timeframe!r} must be higher than {gene.timeframe!r}"
            )

    indicator_by_id = {}
    for index, indicator in enumerate(gene.indicators):
        path = f"indicators[{index}]"
        if indicator.type not in _KNOWN_INDICATORS:
            raise GenomeContractError(f"{path}.type is unknown: {indicator.type!r}")
        if not isinstance(indicator.instance_id, str) or not indicator.instance_id:
            raise GenomeContractError(f"{path}.instance_id must be present")
        if indicator.instance_id in indicator_by_id:
            raise GenomeContractError(
                f"{path}.instance_id is duplicated: {indicator.instance_id!r}"
            )
        indicator_by_id[indicator.instance_id] = indicator
        _validate_parameters(indicator.type, indicator.parameters, f"{path}.parameters")
        _validate_parameter_bounds(
            indicator.param_bounds,
            indicator.parameters,
            f"{path}.param_bounds",
        )
        weight = _finite_number(indicator.weight, f"{path}.weight")
        if not 0.0 <= weight <= 1.0:
            raise GenomeContractError(f"{path}.weight must be in [0, 1]")
        if indicator.timeframe is not None:
            if indicator.timeframe not in gene.informative_timeframes:
                raise GenomeContractError(
                    f"{path}.timeframe is not declared in informative_timeframes"
                )
            if not is_higher_timeframe(indicator.timeframe, gene.timeframe):
                raise GenomeContractError(
                    f"{path}.timeframe must be higher than the base timeframe"
                )

    condition_groups = (
        ("entry_conditions", gene.entry_conditions),
        ("exit_conditions", gene.exit_conditions),
        ("short_entry_conditions", gene.short_entry_conditions),
        ("short_exit_conditions", gene.short_exit_conditions),
    )
    for group_name, conditions in condition_groups:
        for index, condition in enumerate(conditions):
            path = f"{group_name}[{index}]"
            indicator = indicator_by_id.get(condition.indicator)
            if indicator is None:
                raise GenomeContractError(
                    f"{path}.indicator is unresolved: {condition.indicator!r}"
                )
            if not is_valid_operator(indicator.type, condition.operator):
                raise GenomeContractError(
                    f"{path}.operator {condition.operator!r} is invalid for {indicator.type}"
                )
            if condition.logic not in {"AND", "OR"}:
                raise GenomeContractError(f"{path}.logic must be AND or OR")
            threshold = _finite_number(condition.threshold, f"{path}.threshold")
            upper = _finite_number(condition.threshold_upper, f"{path}.threshold_upper")
            if abs(threshold) > 1e12 or abs(upper) > 1e12:
                raise GenomeContractError(f"{path} thresholds exceed hard bounds")
            _integer(condition.lookback, f"{path}.lookback", minimum=1, maximum=10_000)
            if condition.operator == "between" and upper <= threshold:
                raise GenomeContractError(
                    f"{path} requires threshold_upper > threshold for between"
                )

    stoploss = _finite_number(gene.stoploss, "stoploss")
    if not -1.0 < stoploss < 0.0:
        raise GenomeContractError("stoploss must be strictly between -1 and 0")
    _integer(gene.max_open_trades, "max_open_trades", minimum=1, maximum=1_000)
    if not isinstance(gene.minimal_roi, Mapping) or not gene.minimal_roi:
        raise GenomeContractError("minimal_roi must be a non-empty object")
    previous_time = -1
    previous_value = math.inf
    if any(
        not isinstance(raw_time, str) or not raw_time.isdigit() for raw_time in gene.minimal_roi
    ):
        raise GenomeContractError("minimal_roi keys must be non-negative integer strings")
    for raw_time, raw_value in sorted(
        gene.minimal_roi.items(),
        key=lambda item: int(item[0]),
    ):
        time_value = int(raw_time)
        roi_value = _finite_number(raw_value, f"minimal_roi.{raw_time}")
        if time_value > 10_000_000 or roi_value > 10.0:
            raise GenomeContractError("minimal_roi exceeds hard time/value bounds")
        if time_value <= previous_time:
            raise GenomeContractError("minimal_roi time keys must be unique and increasing")
        if roi_value < 0.0 or roi_value > previous_value:
            raise GenomeContractError(
                "minimal_roi values must be non-negative and monotonically decreasing"
            )
        previous_time = time_value
        previous_value = roi_value

    if not isinstance(gene.trailing_stop, bool):
        raise GenomeContractError("trailing_stop must be boolean")
    if gene.trailing_stop:
        positive = _finite_number(
            gene.trailing_stop_positive,
            "trailing_stop_positive",
        )
        offset = _finite_number(
            gene.trailing_stop_positive_offset,
            "trailing_stop_positive_offset",
        )
        if positive <= 0.0 or offset <= positive:
            raise GenomeContractError("trailing stop requires 0 < positive < positive_offset")
    elif gene.trailing_stop_positive is not None or gene.trailing_stop_positive_offset is not None:
        raise GenomeContractError("disabled trailing_stop cannot carry positive/offset values")

    if not isinstance(gene.can_short, bool):
        raise GenomeContractError("can_short must be boolean")
    if gene.preferred_regime not in {None, "bullish", "bearish", "sideways", "volatile"}:
        raise GenomeContractError("preferred_regime is unknown")
    if gene.regime_mode not in {"generalist", "specialist", "exclusive"}:
        raise GenomeContractError("regime_mode is unknown")
    if gene.regime_mode != "generalist" and gene.preferred_regime is None:
        raise GenomeContractError("specialist/exclusive regime mode requires a preferred regime")

    if gene.regime_gene is not None:
        regime = gene.regime_gene
        if not isinstance(regime.enabled, bool):
            raise GenomeContractError("regime_gene.enabled must be boolean")
        if len(regime.regime_timeframes) != len(set(regime.regime_timeframes)):
            raise GenomeContractError("regime_gene.regime_timeframes contains duplicates")
        for timeframe in regime.regime_timeframes:
            if timeframe not in _TIMEFRAMES or not is_higher_timeframe(
                timeframe,
                gene.timeframe,
            ):
                raise GenomeContractError(
                    "regime timeframes must be known and higher than the base timeframe"
                )
        lower = _finite_number(regime.entry_trend_min, "regime_gene.entry_trend_min")
        upper = _finite_number(regime.entry_trend_max, "regime_gene.entry_trend_max")
        if not -1.0 <= lower <= upper <= 1.0:
            raise GenomeContractError("regime trend bounds must satisfy -1 <= min <= max <= 1")
        if regime.combination not in {"hierarchical", "weighted_voting"}:
            raise GenomeContractError("regime_gene.combination is unknown")
        if not isinstance(regime.exit_on_regime_change, bool) or not isinstance(
            regime.micro_regime,
            bool,
        ):
            raise GenomeContractError("regime_gene flags must be boolean")

    if gene.self_mutation_rate is not None:
        rate = _finite_number(gene.self_mutation_rate, "self_mutation_rate")
        if not 0.0 <= rate <= 1.0:
            raise GenomeContractError("self_mutation_rate must be in [0, 1]")
    if gene.self_crossover_pref not in {None, "single_point", "uniform", "component"}:
        raise GenomeContractError("self_crossover_pref is unknown")
