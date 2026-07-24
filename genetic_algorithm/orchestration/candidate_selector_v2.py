"""Deterministic Pareto ranking and diversity selection for analyzed GA waves."""

from __future__ import annotations

import math
from collections import Counter
from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from genetic_algorithm.orchestration.artifact_store_v2 import canonical_config_hash
from genetic_algorithm.orchestration.result_contract import StrictV2Model
from genetic_algorithm.orchestration.wave_analyzer_v2 import (
    CandidateAnalysisV2,
    CandidateEligibilityStatus,
    WaveAnalysisV2,
)


class CandidateSelectionError(ValueError):
    """Raised when candidates cannot be compared under the selection contract."""


class ParetoObjective(StrEnum):
    WORST_ANNUAL_RETURN_LCB = "WORST_ANNUAL_RETURN_LCB"
    WORST_EXPECTANCY_LCB = "WORST_EXPECTANCY_LCB"
    MAX_DRAWDOWN_UCB = "MAX_DRAWDOWN_UCB"
    DAILY_ES5_UCB = "DAILY_ES5_UCB"


OBJECTIVE_SCHEMA = "return-risk-pareto-v1"
_OBJECTIVES = [
    ParetoObjective.WORST_ANNUAL_RETURN_LCB,
    ParetoObjective.WORST_EXPECTANCY_LCB,
    ParetoObjective.MAX_DRAWDOWN_UCB,
    ParetoObjective.DAILY_ES5_UCB,
]


class CandidateSelectionPolicyV2(StrictV2Model):
    schema_version: Literal["2.0"] = "2.0"
    selection_policy_version: str = Field(min_length=1)
    objective_schema: Literal["return-risk-pareto-v1"] = OBJECTIVE_SCHEMA
    max_selected: int = Field(default=8, ge=1)
    min_selected: int = Field(default=1, ge=1)
    max_per_experiment: int = Field(default=3, ge=1)
    max_pareto_rank: int = Field(default=2, ge=0)
    min_normalized_objective_distance: float = Field(default=0.15, ge=0, le=2)
    dominance_epsilon: float = Field(default=1e-12, ge=0)
    require_common_comparison_panel: bool = True
    require_analysis_planning_allowed: bool = True

    @model_validator(mode="after")
    def _coherent_capacity(self) -> CandidateSelectionPolicyV2:
        if self.min_selected > self.max_selected:
            raise ValueError("min_selected cannot exceed max_selected")
        return self

    @property
    def policy_hash(self) -> str:
        return canonical_config_hash(self.model_dump(mode="json"))


class ParetoAssessmentV2(StrictV2Model):
    experiment_id: str = Field(min_length=1)
    phenotype_hash: str = Field(min_length=8)
    comparison_panel_hash: str = Field(min_length=64, max_length=64)
    pareto_rank: int | None = Field(default=None, ge=0)
    selected: bool
    selection_order: int | None = Field(default=None, ge=0)
    min_distance_at_selection: float | None = Field(default=None, ge=0)
    objective_values: dict[ParetoObjective, float]
    reason_codes: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def _consistent_selection(self) -> ParetoAssessmentV2:
        if list(self.objective_values) != _OBJECTIVES:
            raise ValueError("assessment objective order differs from schema")
        if self.reason_codes != sorted(set(self.reason_codes)):
            raise ValueError("assessment reason_codes must be unique and sorted")
        if self.selected:
            if self.selection_order is None or self.pareto_rank is None:
                raise ValueError("selected candidate lacks order or Pareto rank")
            if "SELECTED" not in self.reason_codes:
                raise ValueError("selected candidate lacks SELECTED reason")
        elif self.selection_order is not None or self.min_distance_at_selection is not None:
            raise ValueError("unselected candidate contains selection metadata")
        return self

    @property
    def candidate_key(self) -> str:
        return canonical_config_hash(
            {
                "experiment_id": self.experiment_id,
                "phenotype_hash": self.phenotype_hash,
            }
        )


class CandidateSelectionV2(StrictV2Model):
    schema_version: Literal["2.0"] = "2.0"
    wave_id: str = Field(min_length=1)
    created_at: datetime
    analysis_hash: str = Field(min_length=64, max_length=64)
    analysis_snapshot_hash: str = Field(min_length=64, max_length=64)
    selection_policy: CandidateSelectionPolicyV2
    selection_policy_hash: str = Field(min_length=64, max_length=64)
    objective_schema: Literal["return-risk-pareto-v1"] = OBJECTIVE_SCHEMA
    comparison_panel_hash: str | None = Field(default=None, min_length=64, max_length=64)
    eligible_candidate_count: int = Field(ge=0)
    ineligible_candidate_count: int = Field(ge=0)
    assessments: list[ParetoAssessmentV2] = Field(default_factory=list)
    selected_candidate_keys: list[str] = Field(default_factory=list)
    planning_allowed: bool
    reason_codes: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def _canonical_selection(self) -> CandidateSelectionV2:
        if self.selection_policy.policy_hash != self.selection_policy_hash:
            raise ValueError("selection policy hash differs from content")
        ordered = sorted(
            self.assessments,
            key=lambda item: (item.experiment_id, item.phenotype_hash),
        )
        if self.assessments != ordered:
            raise ValueError("Pareto assessments must be sorted")
        selected = sorted(
            (item for item in self.assessments if item.selected),
            key=lambda item: int(item.selection_order),
        )
        if self.selected_candidate_keys != [item.candidate_key for item in selected]:
            raise ValueError("selected candidate keys differ from assessments")
        if self.reason_codes != sorted(set(self.reason_codes)):
            raise ValueError("selection reason_codes must be unique and sorted")
        if self.planning_allowed and len(selected) < self.selection_policy.min_selected:
            raise ValueError("planning allowed with too few selected candidates")
        return self

    @property
    def selection_hash(self) -> str:
        return canonical_config_hash(self.model_dump(mode="json"))


def _require_metric(value: float | None, field_name: str) -> float:
    if value is None:
        raise CandidateSelectionError(f"eligible candidate lacks {field_name}")
    return float(value)


def _objectives(candidate: CandidateAnalysisV2) -> dict[ParetoObjective, float]:
    return {
        ParetoObjective.WORST_ANNUAL_RETURN_LCB: _require_metric(
            candidate.worst_annualized_return_lcb,
            "worst_annualized_return_lcb",
        ),
        ParetoObjective.WORST_EXPECTANCY_LCB: _require_metric(
            candidate.worst_net_expectancy_lcb,
            "worst_net_expectancy_lcb",
        ),
        ParetoObjective.MAX_DRAWDOWN_UCB: _require_metric(
            candidate.worst_max_drawdown_ucb,
            "worst_max_drawdown_ucb",
        ),
        ParetoObjective.DAILY_ES5_UCB: _require_metric(
            candidate.worst_daily_es5_ucb,
            "worst_daily_es5_ucb",
        ),
    }


def _maximization_vector(values: dict[ParetoObjective, float]) -> tuple[float, ...]:
    return (
        values[ParetoObjective.WORST_ANNUAL_RETURN_LCB],
        values[ParetoObjective.WORST_EXPECTANCY_LCB],
        -values[ParetoObjective.MAX_DRAWDOWN_UCB],
        -values[ParetoObjective.DAILY_ES5_UCB],
    )


def _dominates(
    left: tuple[float, ...],
    right: tuple[float, ...],
    epsilon: float,
) -> bool:
    no_worse = all(a >= b - epsilon for a, b in zip(left, right, strict=True))
    strictly_better = any(a > b + epsilon for a, b in zip(left, right, strict=True))
    return no_worse and strictly_better


def _pareto_ranks(
    candidates: list[CandidateAnalysisV2],
    values: dict[tuple[str, str], dict[ParetoObjective, float]],
    epsilon: float,
) -> dict[tuple[str, str], int]:
    keys = [(item.experiment_id, item.phenotype_hash) for item in candidates]
    vectors = {key: _maximization_vector(values[key]) for key in keys}
    remaining = set(keys)
    ranks: dict[tuple[str, str], int] = {}
    rank = 0
    while remaining:
        front = {
            key
            for key in remaining
            if not any(
                other != key
                and _dominates(vectors[other], vectors[key], epsilon)
                for other in remaining
            )
        }
        if not front:
            raise CandidateSelectionError("Pareto sorting produced an empty front")
        for key in front:
            ranks[key] = rank
        remaining -= front
        rank += 1
    return ranks


def _normalized_vectors(
    values: dict[tuple[str, str], dict[ParetoObjective, float]],
) -> dict[tuple[str, str], tuple[float, ...]]:
    maximizing = {key: _maximization_vector(item) for key, item in values.items()}
    columns = list(zip(*maximizing.values(), strict=True))
    bounds = [(min(column), max(column)) for column in columns]
    return {
        key: tuple(
            0.0 if high == low else (value - low) / (high - low)
            for value, (low, high) in zip(vector, bounds, strict=True)
        )
        for key, vector in maximizing.items()
    }


def _distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right, strict=True)))


def _conservative_key(
    candidate: CandidateAnalysisV2,
    values: dict[ParetoObjective, float],
) -> tuple[float, float, float, float, str, str]:
    return (
        values[ParetoObjective.MAX_DRAWDOWN_UCB],
        values[ParetoObjective.DAILY_ES5_UCB],
        -values[ParetoObjective.WORST_ANNUAL_RETURN_LCB],
        -values[ParetoObjective.WORST_EXPECTANCY_LCB],
        candidate.experiment_id,
        candidate.phenotype_hash,
    )


def _select_diverse(
    candidates: list[CandidateAnalysisV2],
    values: dict[tuple[str, str], dict[ParetoObjective, float]],
    ranks: dict[tuple[str, str], int],
    policy: CandidateSelectionPolicyV2,
) -> tuple[list[tuple[str, str]], dict[tuple[str, str], float | None]]:
    normalized = _normalized_vectors(values)
    selected: list[tuple[str, str]] = []
    selected_distances: dict[tuple[str, str], float | None] = {}
    experiment_counts: Counter[str] = Counter()
    by_key = {(item.experiment_id, item.phenotype_hash): item for item in candidates}

    for rank in range(policy.max_pareto_rank + 1):
        available = {
            key
            for key, item_rank in ranks.items()
            if item_rank == rank
            and experiment_counts[key[0]] < policy.max_per_experiment
        }
        while available and len(selected) < policy.max_selected:
            if not selected:
                chosen = min(
                    available,
                    key=lambda key: _conservative_key(by_key[key], values[key]),
                )
                minimum_distance = None
            else:
                distances = {
                    key: min(
                        _distance(normalized[key], normalized[selected_key])
                        for selected_key in selected
                    )
                    for key in available
                }
                eligible = {
                    key
                    for key, distance in distances.items()
                    if distance >= policy.min_normalized_objective_distance
                }
                if not eligible:
                    break
                chosen = min(
                    eligible,
                    key=lambda key: (
                        -distances[key],
                        *_conservative_key(by_key[key], values[key]),
                    ),
                )
                minimum_distance = distances[chosen]
            selected.append(chosen)
            selected_distances[chosen] = minimum_distance
            experiment_counts[chosen[0]] += 1
            available.remove(chosen)
            available = {
                key
                for key in available
                if experiment_counts[key[0]] < policy.max_per_experiment
            }
        if len(selected) >= policy.max_selected:
            break
    return selected, selected_distances


def select_wave_candidates(
    analysis: WaveAnalysisV2,
    policy: CandidateSelectionPolicyV2,
) -> CandidateSelectionV2:
    """Rank and select candidates without collapsing return and risk to one score."""

    eligible = [
        item
        for item in analysis.candidates
        if item.eligibility_status == CandidateEligibilityStatus.ELIGIBLE
    ]
    ineligible_count = len(analysis.candidates) - len(eligible)
    reasons: set[str] = set()
    if policy.require_analysis_planning_allowed and not analysis.planning_allowed:
        reasons.add("ANALYSIS_BLOCKED")
    if not eligible:
        reasons.add("NO_ELIGIBLE_CANDIDATES")

    panel_hashes = {item.comparison_panel_hash for item in eligible}
    if None in panel_hashes:
        raise CandidateSelectionError("eligible candidate lacks comparison panel hash")
    common_panel = next(iter(panel_hashes)) if len(panel_hashes) == 1 else None
    if policy.require_common_comparison_panel and len(panel_hashes) > 1:
        reasons.add("INCOMPARABLE_CANDIDATE_PANELS")

    values = {
        (item.experiment_id, item.phenotype_hash): _objectives(item) for item in eligible
    }
    may_compare = bool(eligible) and (
        common_panel is not None or not policy.require_common_comparison_panel
    )
    ranks = (
        _pareto_ranks(eligible, values, policy.dominance_epsilon)
        if may_compare
        else {}
    )
    selected, distances = (
        _select_diverse(eligible, values, ranks, policy)
        if may_compare and "ANALYSIS_BLOCKED" not in reasons
        else ([], {})
    )
    if len(selected) < policy.min_selected:
        reasons.add("INSUFFICIENT_DIVERSE_CANDIDATES")
    if not reasons:
        reasons.add("SELECTION_COMPLETE")

    order = {key: index for index, key in enumerate(selected)}
    assessments: list[ParetoAssessmentV2] = []
    for candidate in eligible:
        key = (candidate.experiment_id, candidate.phenotype_hash)
        candidate_reasons: set[str] = set()
        if key in order:
            candidate_reasons.add("SELECTED")
        elif key not in ranks:
            candidate_reasons.add("INCOMPARABLE_PANEL")
        elif ranks[key] > policy.max_pareto_rank:
            candidate_reasons.add("PARETO_RANK_LIMIT")
        else:
            candidate_reasons.add("DIVERSITY_OR_CAPACITY_LIMIT")
        assessments.append(
            ParetoAssessmentV2(
                experiment_id=candidate.experiment_id,
                phenotype_hash=candidate.phenotype_hash,
                comparison_panel_hash=str(candidate.comparison_panel_hash),
                pareto_rank=ranks.get(key),
                selected=key in order,
                selection_order=order.get(key),
                min_distance_at_selection=distances.get(key),
                objective_values=values[key],
                reason_codes=sorted(candidate_reasons),
            )
        )
    assessments.sort(key=lambda item: (item.experiment_id, item.phenotype_hash))
    selected_assessments = sorted(
        (item for item in assessments if item.selected),
        key=lambda item: int(item.selection_order),
    )
    planning_allowed = len(selected) >= policy.min_selected and not (
        reasons - {"SELECTION_COMPLETE"}
    )
    return CandidateSelectionV2(
        wave_id=analysis.wave_id,
        created_at=analysis.created_at,
        analysis_hash=analysis.analysis_hash,
        analysis_snapshot_hash=analysis.snapshot_hash,
        selection_policy=policy,
        selection_policy_hash=policy.policy_hash,
        comparison_panel_hash=common_panel,
        eligible_candidate_count=len(eligible),
        ineligible_candidate_count=ineligible_count,
        assessments=assessments,
        selected_candidate_keys=[item.candidate_key for item in selected_assessments],
        planning_allowed=planning_allowed,
        reason_codes=sorted(reasons),
    )
