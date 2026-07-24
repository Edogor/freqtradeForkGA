"""Explicit, fail-closed spatial and temporal evaluation split contract."""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import date, datetime, timedelta
from typing import Any

from pydantic import Field, model_validator

from genetic_algorithm.orchestration.artifact_store_v2 import canonical_config_hash
from genetic_algorithm.orchestration.promotion_policy_v2 import ShadowGatePolicyV2
from genetic_algorithm.orchestration.result_contract import ScenarioRole, StrictV2Model


class SplitContractError(ValueError):
    """Raised when evaluation roles overlap or cannot be proven from config."""


class SplitScenarioV2(StrictV2Model):
    scenario_id: str = Field(min_length=1)
    pair: str = Field(min_length=3)
    timeframe: str = Field(min_length=1)
    role: ScenarioRole
    period_start: date
    period_end: date

    @model_validator(mode="after")
    def _valid_period(self) -> SplitScenarioV2:
        if self.period_end < self.period_start:
            raise ValueError("period_end precedes period_start")
        return self

    @property
    def exclusive_end(self) -> date:
        return self.period_end + timedelta(days=1)


class EvaluationSplitPlanV2(StrictV2Model):
    schema_version: str = "2.0"
    exchange: str = Field(min_length=1)
    evolution_pairs: list[str] = Field(min_length=1)
    evolution_period_start: date
    evolution_period_end_exclusive: date
    min_embargo_days: int = Field(default=0, ge=0)
    scenarios: list[SplitScenarioV2] = Field(min_length=1)

    @model_validator(mode="after")
    def _canonical_order(self) -> EvaluationSplitPlanV2:
        if self.evolution_pairs != sorted(set(self.evolution_pairs)):
            raise ValueError("evolution_pairs must be sorted and unique")
        if self.evolution_period_end_exclusive <= self.evolution_period_start:
            raise ValueError("evolution period is empty")
        scenario_ids = [item.scenario_id for item in self.scenarios]
        if len(scenario_ids) != len(set(scenario_ids)):
            raise ValueError("split scenario IDs must be unique")
        if scenario_ids != sorted(scenario_ids):
            raise ValueError("split scenarios must be sorted")
        return self

    @property
    def split_hash(self) -> str:
        return canonical_config_hash(self.model_dump(mode="json"))


_TIMERANGE_PATTERN = re.compile(r"^(\d{8})-(\d{8})$")


def _parse_closed_config_timerange(value: object) -> tuple[date, date]:
    if not isinstance(value, str):
        raise SplitContractError("backtesting.timerange must be an explicit closed timerange")
    match = _TIMERANGE_PATTERN.fullmatch(value)
    if match is None:
        raise SplitContractError(
            "backtesting.timerange must use YYYYMMDD-YYYYMMDD with both bounds"
        )
    try:
        start = datetime.strptime(match.group(1), "%Y%m%d").date()
        exclusive_end = datetime.strptime(match.group(2), "%Y%m%d").date()
    except ValueError as exc:
        raise SplitContractError("backtesting.timerange contains an invalid date") from exc
    if exclusive_end <= start:
        raise SplitContractError("backtesting.timerange must contain at least one day")
    return start, exclusive_end


def _overlaps(
    first_start: date,
    first_end_exclusive: date,
    second_start: date,
    second_end_exclusive: date,
) -> bool:
    return max(first_start, second_start) < min(first_end_exclusive, second_end_exclusive)


def _validate_role_semantics(
    scenario: SplitScenarioV2,
    *,
    evolution_pairs: set[str],
    evolution_start: date,
    evolution_end: date,
    embargo: timedelta,
) -> None:
    if scenario.role == ScenarioRole.INNER_VALIDATION:
        raise SplitContractError(
            f"{scenario.scenario_id} uses ambiguous INNER_VALIDATION; declare "
            "PAIR_VALIDATION or TEMPORAL_VALIDATION"
        )
    if scenario.role == ScenarioRole.TRAIN:
        if scenario.pair not in evolution_pairs:
            raise SplitContractError(
                f"TRAIN scenario {scenario.scenario_id} uses a non-evolution pair"
            )
        if scenario.period_start < evolution_start or scenario.exclusive_end > evolution_end:
            raise SplitContractError(
                f"TRAIN scenario {scenario.scenario_id} escapes the evolution timerange"
            )
    elif scenario.role == ScenarioRole.PAIR_VALIDATION:
        if scenario.pair in evolution_pairs:
            raise SplitContractError(
                f"PAIR_VALIDATION scenario {scenario.scenario_id} reuses an evolution pair"
            )
    elif scenario.role == ScenarioRole.TEMPORAL_VALIDATION:
        if scenario.pair not in evolution_pairs:
            raise SplitContractError(
                f"TEMPORAL_VALIDATION scenario {scenario.scenario_id} must use an evolution pair"
            )
        earliest = evolution_end + embargo
        if scenario.period_start < earliest:
            raise SplitContractError(
                f"TEMPORAL_VALIDATION scenario {scenario.scenario_id} overlaps/precedes "
                "evolution or violates embargo"
            )


def _validate_cross_role_observations(
    scenarios: list[SplitScenarioV2],
    *,
    evolution_pairs: set[str],
    evolution_start: date,
    evolution_end: date,
    embargo: timedelta,
) -> None:
    non_final_by_pair: dict[str, list[tuple[date, date, str]]] = {
        pair: [(evolution_start, evolution_end, "EVOLUTION")] for pair in evolution_pairs
    }
    for scenario in scenarios:
        if scenario.role != ScenarioRole.FINAL_TEST:
            non_final_by_pair.setdefault(scenario.pair, []).append(
                (scenario.period_start, scenario.exclusive_end, scenario.scenario_id)
            )

    for index, first in enumerate(scenarios):
        for second in scenarios[index + 1 :]:
            if first.pair != second.pair or first.role == second.role:
                continue
            if _overlaps(
                first.period_start,
                first.exclusive_end,
                second.period_start,
                second.exclusive_end,
            ):
                raise SplitContractError(
                    f"scenarios {first.scenario_id} ({first.role.value}) and "
                    f"{second.scenario_id} ({second.role.value}) reuse the same observations"
                )

    for final in (item for item in scenarios if item.role == ScenarioRole.FINAL_TEST):
        prior = non_final_by_pair.get(final.pair, [])
        if not prior:
            continue
        latest_non_final_end = max(end for _, end, _ in prior)
        if final.period_start < latest_non_final_end + embargo:
            sources = sorted(source for _, _, source in prior)
            raise SplitContractError(
                f"FINAL_TEST scenario {final.scenario_id} is not strictly after prior "
                f"same-pair observations/embargo: {sources}"
            )


def build_evaluation_split_plan(
    config: Mapping[str, Any],
    policy: ShadowGatePolicyV2,
) -> EvaluationSplitPlanV2:
    """Build and validate a canonical split plan from resolved config and policy."""

    backtesting = config.get("backtesting")
    if not isinstance(backtesting, Mapping):
        raise SplitContractError("resolved config lacks backtesting settings")
    raw_pairs = backtesting.get("pairs")
    if (
        not isinstance(raw_pairs, list)
        or not raw_pairs
        or not all(isinstance(pair, str) and pair for pair in raw_pairs)
    ):
        raise SplitContractError("backtesting.pairs must be a non-empty explicit list")
    configured_pairs = sorted(set(raw_pairs))
    if len(configured_pairs) != len(raw_pairs):
        raise SplitContractError("backtesting.pairs contains duplicates")
    pair_validation = config.get("pair_validation", {})
    if not isinstance(pair_validation, Mapping):
        raise SplitContractError("pair_validation must be a mapping")
    if pair_validation.get("enabled", False):
        training_pairs = pair_validation.get("training_pairs")
        validation_pairs = pair_validation.get("validation_pairs")
        if (
            not isinstance(training_pairs, list)
            or not training_pairs
            or not all(isinstance(pair, str) and pair for pair in training_pairs)
            or not isinstance(validation_pairs, list)
            or not validation_pairs
            or not all(isinstance(pair, str) and pair for pair in validation_pairs)
        ):
            raise SplitContractError(
                "enabled pair_validation requires non-empty training/validation pair lists"
            )
        if set(training_pairs).intersection(validation_pairs):
            raise SplitContractError(
                "pair_validation training and validation pairs must be disjoint"
            )
        if set(training_pairs).union(validation_pairs) != set(configured_pairs):
            raise SplitContractError(
                "pair_validation train/validation union must equal backtesting.pairs"
            )
        evolution_pairs = sorted(set(training_pairs))
    else:
        evolution_pairs = configured_pairs
    evolution_start, evolution_end = _parse_closed_config_timerange(backtesting.get("timerange"))
    split_config = config.get("split_v2", {})
    if not isinstance(split_config, Mapping):
        raise SplitContractError("split_v2 must be a mapping")
    min_embargo_days = split_config.get("min_embargo_days", 0)
    if not isinstance(min_embargo_days, int) or isinstance(min_embargo_days, bool):
        raise SplitContractError("split_v2.min_embargo_days must be an integer")
    if min_embargo_days < 0:
        raise SplitContractError("split_v2.min_embargo_days cannot be negative")
    embargo = timedelta(days=min_embargo_days)

    scenarios = sorted(
        [
            SplitScenarioV2(
                scenario_id=item.scenario_id,
                pair=item.pair,
                timeframe=item.timeframe,
                role=item.role,
                period_start=item.period_start,
                period_end=item.period_end,
            )
            for item in policy.required_scenarios
        ],
        key=lambda item: item.scenario_id,
    )
    evolution_pair_set = set(evolution_pairs)
    for scenario in scenarios:
        _validate_role_semantics(
            scenario,
            evolution_pairs=evolution_pair_set,
            evolution_start=evolution_start,
            evolution_end=evolution_end,
            embargo=embargo,
        )
    _validate_cross_role_observations(
        scenarios,
        evolution_pairs=evolution_pair_set,
        evolution_start=evolution_start,
        evolution_end=evolution_end,
        embargo=embargo,
    )
    return EvaluationSplitPlanV2(
        exchange=str(backtesting.get("exchange", "binance")),
        evolution_pairs=evolution_pairs,
        evolution_period_start=evolution_start,
        evolution_period_end_exclusive=evolution_end,
        min_embargo_days=min_embargo_days,
        scenarios=scenarios,
    )
