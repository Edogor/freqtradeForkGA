"""Pure, deterministic child-wave planning from analyzed and selected evidence."""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import Field, model_validator

from genetic_algorithm.orchestration.artifact_store_v2 import canonical_config_hash
from genetic_algorithm.orchestration.candidate_selector_v2 import (
    CandidateSelectionPolicyV2,
    CandidateSelectionV2,
    select_wave_candidates,
)
from genetic_algorithm.orchestration.result_contract import StrictV2Model
from genetic_algorithm.orchestration.wave_analyzer_v2 import (
    CandidateAnalysisV2,
    CandidateObservationRefV2,
    WaveAnalysisV2,
)
from genetic_algorithm.orchestration.wave_state_v2 import (
    ExperimentArmType,
    ExperimentSpecV2,
    WaveBudgetV2,
    WaveDecisionType,
    WaveDecisionV2,
)


class WavePlanningError(ValueError):
    """Raised when a child wave cannot be derived without ambiguity or mutation."""


class PlannerSourceMode(StrEnum):
    BASELINE_CONTROL = "BASELINE_CONTROL"
    SELECTED_CANDIDATES = "SELECTED_CANDIDATES"


class WavePlanMode(StrEnum):
    """Execution shape fixed before arm templates are interpreted."""

    EVOLUTION_EXPERIMENT = "EVOLUTION_EXPERIMENT"
    REPLAY_VALIDATION = "REPLAY_VALIDATION"


_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]+$")


class WaveArmTemplateV2(StrictV2Model):
    arm_id: str = Field(min_length=1)
    arm_type: ExperimentArmType
    source_mode: PlannerSourceMode
    hypothesis: str = Field(min_length=1)
    primary_metric: str = Field(min_length=1)
    factor_delta: dict[str, Any] = Field(default_factory=dict)
    seeds: list[int] = Field(min_length=1)
    selected_candidate_limit: int = Field(default=1, ge=1)
    factorial: bool = False

    @model_validator(mode="after")
    def _canonical_arm(self) -> WaveArmTemplateV2:
        if not _SAFE_ID.fullmatch(self.arm_id) or self.arm_id in {".", ".."}:
            raise ValueError("arm_id is not artifact-safe")
        if self.seeds != sorted(set(self.seeds)):
            raise ValueError("arm seeds must be unique and sorted")
        self._validate_source_semantics()
        self._validate_factor_delta()
        return self

    def _validate_source_semantics(self) -> None:
        if self.source_mode == PlannerSourceMode.BASELINE_CONTROL:
            if self.arm_type != ExperimentArmType.CONTROL:
                raise ValueError("baseline source must create CONTROL arm")
            if self.selected_candidate_limit != 1:
                raise ValueError("baseline control cannot select multiple candidates")
            if self.factor_delta:
                raise ValueError("baseline control must not contain factor deltas")
        elif self.arm_type == ExperimentArmType.CONTROL:
            raise ValueError("CONTROL arm must use BASELINE_CONTROL source")
        if self.arm_type == ExperimentArmType.REPLICATION and self.factor_delta:
            raise ValueError("REPLICATION arm must not contain factor deltas")

    def _validate_factor_delta(self) -> None:
        if self.arm_type in {ExperimentArmType.EXPLOIT, ExperimentArmType.EXPLORE}:
            if not self.factor_delta:
                raise ValueError("EXPLOIT/EXPLORE arm requires an explicit factor delta")
            if len(self.factor_delta) > 1 and not self.factorial:
                raise ValueError("multi-factor arm must be explicitly factorial")
        if list(self.factor_delta) != sorted(self.factor_delta):
            raise ValueError("factor_delta paths must be sorted")
        invalid_paths = [
            path
            for path in self.factor_delta
            if not path or path.startswith(".") or path.endswith(".")
        ]
        if invalid_paths:
            raise ValueError("factor_delta contains invalid dotted path")


class WavePlannerPolicyV2(StrictV2Model):
    schema_version: Literal["2.0"] = "2.0"
    planner_policy_version: str = Field(min_length=1)
    child_result_policy_version: str = Field(min_length=1)
    child_search_space_version: str = Field(min_length=1)
    plan_mode: WavePlanMode = WavePlanMode.EVOLUTION_EXPERIMENT
    budget: WaveBudgetV2
    arm_templates: list[WaveArmTemplateV2] = Field(min_length=1)
    allow_control_fallback_when_no_candidates: bool = False
    allow_control_recovery_after_technical_failure: bool = False
    rotate_seeds_per_parent_wave: bool = False

    @model_validator(mode="after")
    def _coherent_templates(self) -> WavePlannerPolicyV2:
        arm_ids = [item.arm_id for item in self.arm_templates]
        if len(arm_ids) != len(set(arm_ids)):
            raise ValueError("planner arm IDs must be unique")
        controls = [
            item
            for item in self.arm_templates
            if item.source_mode == PlannerSourceMode.BASELINE_CONTROL
        ]
        if self.plan_mode == WavePlanMode.EVOLUTION_EXPERIMENT:
            if len(controls) != 1:
                raise ValueError("evolution planner requires exactly one baseline control template")
        else:
            if controls:
                raise ValueError("replay validation cannot contain a baseline control template")
            if any(
                item.arm_type != ExperimentArmType.VALIDATION
                or item.source_mode != PlannerSourceMode.SELECTED_CANDIDATES
                or item.factor_delta
                for item in self.arm_templates
            ):
                raise ValueError(
                    "replay validation requires selected-candidate VALIDATION arms without deltas"
                )
        if (
            (
                self.allow_control_fallback_when_no_candidates
                or self.allow_control_recovery_after_technical_failure
            )
            and self.plan_mode != WavePlanMode.EVOLUTION_EXPERIMENT
        ):
            raise ValueError("control fallback/recovery is only valid for evolution planning")
        paired_seeds = {tuple(item.seeds) for item in self.arm_templates}
        if len(paired_seeds) != 1:
            raise ValueError("all planner arms must use identical paired seeds")
        return self

    @property
    def policy_hash(self) -> str:
        return canonical_config_hash(self.model_dump(mode="json"))


class PlannedSourceV2(StrictV2Model):
    source_mode: PlannerSourceMode
    parent_experiment_id: str = Field(min_length=1)
    parent_config_hash: str = Field(min_length=64, max_length=64)
    selected_candidate_key: str | None = None
    phenotype_hash: str | None = None
    observation_ref: CandidateObservationRefV2 | None = None

    @model_validator(mode="after")
    def _source_shape(self) -> PlannedSourceV2:
        candidate_fields = (
            self.selected_candidate_key,
            self.phenotype_hash,
            self.observation_ref,
        )
        if self.source_mode == PlannerSourceMode.SELECTED_CANDIDATES:
            if any(value is None for value in candidate_fields):
                raise ValueError("selected source lacks candidate provenance")
        elif any(value is not None for value in candidate_fields):
            raise ValueError("baseline source cannot contain candidate provenance")
        return self


class PlannedAttemptV2(StrictV2Model):
    attempt_id: str = Field(min_length=1)
    experiment_id: str = Field(min_length=1)
    seed: int
    ordinal: int = Field(ge=0)
    resolved_config_hash: str = Field(min_length=64, max_length=64)


class PlannedExperimentV2(StrictV2Model):
    experiment_id: str = Field(min_length=1)
    arm_id: str = Field(min_length=1)
    arm_type: ExperimentArmType
    hypothesis: str = Field(min_length=1)
    primary_metric: str = Field(min_length=1)
    source: PlannedSourceV2
    factor_delta: dict[str, Any] = Field(default_factory=dict)
    resolved_config: dict[str, Any]
    resolved_config_hash: str = Field(min_length=64, max_length=64)
    seeds: list[int] = Field(min_length=1)
    attempts: list[PlannedAttemptV2] = Field(min_length=1)

    @model_validator(mode="after")
    def _consistent_experiment(self) -> PlannedExperimentV2:
        if canonical_config_hash(self.resolved_config) != self.resolved_config_hash:
            raise ValueError("planned resolved config hash differs from content")
        if self.seeds != sorted(set(self.seeds)):
            raise ValueError("planned experiment seeds must be unique and sorted")
        ordered = sorted(self.attempts, key=lambda item: item.ordinal)
        if self.attempts != ordered:
            raise ValueError("planned attempts must be sorted by ordinal")
        if [item.ordinal for item in ordered] != list(range(len(ordered))):
            raise ValueError("planned attempt ordinals must be contiguous")
        if [item.seed for item in ordered] != self.seeds:
            raise ValueError("planned attempt seeds differ from experiment seeds")
        if any(item.experiment_id != self.experiment_id for item in ordered):
            raise ValueError("planned attempt belongs to another experiment")
        if any(item.resolved_config_hash != self.resolved_config_hash for item in ordered):
            raise ValueError("planned attempt config differs from experiment")
        return self


class ChildWavePlanV2(StrictV2Model):
    schema_version: Literal["2.0"] = "2.0"
    wave_id: str = Field(min_length=1)
    parent_wave_id: str = Field(min_length=1)
    created_at: datetime
    analysis_hash: str = Field(min_length=64, max_length=64)
    selection_hash: str = Field(min_length=64, max_length=64)
    planner_policy: WavePlannerPolicyV2
    planner_policy_hash: str = Field(min_length=64, max_length=64)
    plan_input_hash: str = Field(min_length=64, max_length=64)
    result_policy_version: str = Field(min_length=1)
    search_space_version: str = Field(min_length=1)
    budget: WaveBudgetV2
    experiments: list[PlannedExperimentV2] = Field(default_factory=list)
    planning_allowed: bool
    reason_codes: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def _canonical_plan(self) -> ChildWavePlanV2:
        if self.created_at.utcoffset() is None:
            raise ValueError("plan created_at must be timezone-aware")
        if self.planner_policy.policy_hash != self.planner_policy_hash:
            raise ValueError("planner policy hash differs from content")
        ordered = sorted(self.experiments, key=lambda item: item.experiment_id)
        if self.experiments != ordered:
            raise ValueError("planned experiments must be sorted")
        attempts = [attempt for item in self.experiments for attempt in item.attempts]
        if len(attempts) > self.budget.max_attempts:
            raise ValueError("planned attempts exceed wave budget")
        attempt_ids = [item.attempt_id for item in attempts]
        if len(attempt_ids) != len(set(attempt_ids)):
            raise ValueError("planned attempt IDs must be unique")
        if self.reason_codes != sorted(set(self.reason_codes)):
            raise ValueError("plan reason_codes must be unique and sorted")
        if self.planning_allowed:
            if not self.experiments or tuple(self.reason_codes) not in {
                ("PLAN_COMPLETE",),
                ("PLAN_CONTROL_FALLBACK",),
                ("PLAN_CONTROL_RECOVERY",),
            }:
                raise ValueError("allowed plan is incomplete")
        elif self.experiments:
            raise ValueError("blocked plan cannot contain executable experiments")
        return self

    @property
    def plan_hash(self) -> str:
        return canonical_config_hash(self.model_dump(mode="json"))

    def to_decision(self, *, analysis_decision_hash: str) -> WaveDecisionV2:
        if len(analysis_decision_hash) != 64:
            raise WavePlanningError("analysis_decision_hash must be SHA-256")
        if self.planning_allowed:
            raise WavePlanningError(
                "an allowed plan must be materialized before a PROPOSAL can be created"
            )
        decision_type = WaveDecisionType.BLOCK
        return WaveDecisionV2(
            decision_id=f"{decision_type.value.lower()}-{self.plan_hash[:24]}",
            wave_id=self.parent_wave_id,
            decision_type=decision_type,
            created_at=self.created_at,
            actor="wave-planner-v2",
            reason_codes=self.reason_codes,
            input_hash=analysis_decision_hash,
            payload={
                "child_wave_plan": self.model_dump(mode="json"),
                "plan_hash": self.plan_hash,
                "selection_hash": self.selection_hash,
                "planner_policy_hash": self.planner_policy_hash,
                "planning_allowed": self.planning_allowed,
            },
        )


def _replace_config_value(config: dict[str, Any], dotted_path: str, value: Any) -> None:
    parts = dotted_path.split(".")
    if any(not part for part in parts):
        raise WavePlanningError(f"invalid factor path: {dotted_path}")
    current: dict[str, Any] = config
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            raise WavePlanningError(f"factor path is missing or not a mapping: {dotted_path}")
        current = child
    leaf = parts[-1]
    if leaf not in current:
        raise WavePlanningError(f"factor path does not exist in base config: {dotted_path}")
    previous = current[leaf]
    if isinstance(previous, bool) != isinstance(value, bool):
        raise WavePlanningError(f"factor value changes boolean type: {dotted_path}")
    if previous is not None and value is not None:
        numeric = isinstance(previous, (int, float)) and isinstance(value, (int, float))
        if not numeric and type(previous) is not type(value):
            raise WavePlanningError(f"factor value changes field type: {dotted_path}")
    current[leaf] = copy.deepcopy(value)


def _resolved_variant(
    base_config: Mapping[str, Any],
    factor_delta: Mapping[str, Any],
) -> dict[str, Any]:
    resolved = copy.deepcopy(dict(base_config))
    for path in sorted(factor_delta):
        _replace_config_value(resolved, path, factor_delta[path])
    canonical_config_hash(resolved)
    return resolved


def _selected_candidates(
    analysis: WaveAnalysisV2,
    selection: CandidateSelectionV2,
) -> list[CandidateAnalysisV2]:
    by_key = {
        canonical_config_hash(
            {
                "experiment_id": item.experiment_id,
                "phenotype_hash": item.phenotype_hash,
            }
        ): item
        for item in analysis.candidates
    }
    missing = [key for key in selection.selected_candidate_keys if key not in by_key]
    if missing:
        raise WavePlanningError(f"selection references unknown candidates: {missing}")
    return [by_key[key] for key in selection.selected_candidate_keys]


def _source_for_candidate(
    candidate: CandidateAnalysisV2,
    experiment: ExperimentSpecV2,
) -> PlannedSourceV2:
    observation = min(
        candidate.observation_refs,
        key=lambda item: (item.seed, item.attempt_id, item.candidate_id),
    )
    candidate_key = canonical_config_hash(
        {
            "experiment_id": candidate.experiment_id,
            "phenotype_hash": candidate.phenotype_hash,
        }
    )
    return PlannedSourceV2(
        source_mode=PlannerSourceMode.SELECTED_CANDIDATES,
        parent_experiment_id=experiment.experiment_id,
        parent_config_hash=experiment.resolved_config_hash,
        selected_candidate_key=candidate_key,
        phenotype_hash=candidate.phenotype_hash,
        observation_ref=observation,
    )


def _plan_input_hash(
    analysis: WaveAnalysisV2,
    selection: CandidateSelectionV2,
    experiments: list[ExperimentSpecV2],
    policy: WavePlannerPolicyV2,
) -> str:
    return canonical_config_hash(
        {
            "parent_wave_id": analysis.wave_id,
            "analysis_hash": analysis.analysis_hash,
            "selection_hash": selection.selection_hash,
            "planner_policy_hash": policy.policy_hash,
            "experiment_spec_hashes": [item.spec_hash for item in experiments],
        }
    )


def _planned_experiment(
    *,
    child_wave_id: str,
    template: WaveArmTemplateV2,
    source: PlannedSourceV2,
    base_config: Mapping[str, Any],
) -> PlannedExperimentV2:
    resolved_config = _resolved_variant(base_config, template.factor_delta)
    resolved_hash = canonical_config_hash(resolved_config)
    experiment_hash = canonical_config_hash(
        {
            "wave_id": child_wave_id,
            "arm_id": template.arm_id,
            "source": source.model_dump(mode="json"),
            "resolved_config_hash": resolved_hash,
            "seeds": template.seeds,
        }
    )
    experiment_id = f"experiment-{experiment_hash[:24]}"
    attempts = [
        PlannedAttemptV2(
            attempt_id=_planned_attempt_id(experiment_id, seed),
            experiment_id=experiment_id,
            seed=seed,
            ordinal=index,
            resolved_config_hash=resolved_hash,
        )
        for index, seed in enumerate(template.seeds)
    ]
    return PlannedExperimentV2(
        experiment_id=experiment_id,
        arm_id=template.arm_id,
        arm_type=template.arm_type,
        hypothesis=template.hypothesis,
        primary_metric=template.primary_metric,
        source=source,
        factor_delta=template.factor_delta,
        resolved_config=resolved_config,
        resolved_config_hash=resolved_hash,
        seeds=template.seeds,
        attempts=attempts,
    )


def _planned_attempt_id(experiment_id: str, seed: int) -> str:
    identity_hash = canonical_config_hash(
        {"experiment_id": experiment_id, "seed": seed}
    )
    return f"attempt-{identity_hash[:24]}"


def _parent_experiment_context(
    analysis: WaveAnalysisV2,
    selection: CandidateSelectionV2,
    parent_experiments: list[ExperimentSpecV2],
    policy: WavePlannerPolicyV2,
) -> tuple[list[ExperimentSpecV2], dict[str, ExperimentSpecV2]]:
    if selection.analysis_hash != analysis.analysis_hash:
        raise WavePlanningError("selection belongs to another analysis")
    if policy.child_result_policy_version != analysis.result_policy_version:
        raise WavePlanningError("child result policy differs from analyzed result policy")
    experiments = sorted(parent_experiments, key=lambda item: item.experiment_id)
    experiment_map = {item.experiment_id: item for item in experiments}
    if len(experiment_map) != len(experiments):
        raise WavePlanningError("duplicate parent experiment")
    if any(item.wave_id != analysis.wave_id for item in experiments):
        raise WavePlanningError("parent experiment belongs to another wave")
    return experiments, experiment_map


def _blocked_child_plan(
    analysis: WaveAnalysisV2,
    selection: CandidateSelectionV2,
    policy: WavePlannerPolicyV2,
    *,
    input_hash: str,
    child_wave_id: str,
) -> ChildWavePlanV2:
    return ChildWavePlanV2(
        wave_id=child_wave_id,
        parent_wave_id=analysis.wave_id,
        created_at=analysis.created_at,
        analysis_hash=analysis.analysis_hash,
        selection_hash=selection.selection_hash,
        planner_policy=policy,
        planner_policy_hash=policy.policy_hash,
        plan_input_hash=input_hash,
        result_policy_version=policy.child_result_policy_version,
        search_space_version=policy.child_search_space_version,
        budget=policy.budget,
        planning_allowed=False,
        reason_codes=["SELECTION_BLOCKED"],
    )


def _template_sources(
    template: WaveArmTemplateV2,
    control: ExperimentSpecV2 | None,
    selected: list[CandidateAnalysisV2],
    experiment_map: dict[str, ExperimentSpecV2],
) -> list[PlannedSourceV2]:
    if template.source_mode == PlannerSourceMode.BASELINE_CONTROL:
        if control is None:
            raise WavePlanningError("baseline template lacks a parent CONTROL experiment")
        return [
            PlannedSourceV2(
                source_mode=PlannerSourceMode.BASELINE_CONTROL,
                parent_experiment_id=control.experiment_id,
                parent_config_hash=control.resolved_config_hash,
            )
        ]
    sources = [
        _source_for_candidate(candidate, experiment_map[candidate.experiment_id])
        for candidate in selected[: template.selected_candidate_limit]
    ]
    if not sources:
        raise WavePlanningError(f"arm has no selected candidate source: {template.arm_id}")
    return sources


def _materialize_experiments(
    *,
    child_wave_id: str,
    templates: list[WaveArmTemplateV2],
    control: ExperimentSpecV2 | None,
    selected: list[CandidateAnalysisV2],
    experiment_map: dict[str, ExperimentSpecV2],
    resolved_configs: Mapping[str, Mapping[str, Any]],
) -> list[PlannedExperimentV2]:
    planned: list[PlannedExperimentV2] = []
    for template in templates:
        sources = _template_sources(template, control, selected, experiment_map)
        for source in sources:
            base_config = resolved_configs.get(source.parent_config_hash)
            if base_config is None:
                raise WavePlanningError(
                    f"missing resolved parent config: {source.parent_config_hash}"
                )
            if canonical_config_hash(base_config) != source.parent_config_hash:
                raise WavePlanningError("resolved parent config hash differs from content")
            planned.append(
                _planned_experiment(
                    child_wave_id=child_wave_id,
                    template=template,
                    source=source,
                    base_config=base_config,
                )
            )
    return sorted(planned, key=lambda item: item.experiment_id)


def _rotate_template_seeds(
    templates: list[WaveArmTemplateV2],
    *,
    parent_wave_id: str,
    policy: WavePlannerPolicyV2,
) -> list[WaveArmTemplateV2]:
    """Derive one deterministic paired seed panel for each child wave."""

    if not policy.rotate_seeds_per_parent_wave:
        return templates
    base_seeds = templates[0].seeds
    rotated = [
        int(
            canonical_config_hash(
                {
                    "contract": "CHILD_WAVE_SEED_V1",
                    "parent_wave_id": parent_wave_id,
                    "base_seed": seed,
                    "ordinal": ordinal,
                }
            )[:8],
            16,
        )
        for ordinal, seed in enumerate(base_seeds)
    ]
    if len(rotated) != len(set(rotated)):
        raise WavePlanningError("derived child-wave seeds collide")
    return [
        WaveArmTemplateV2.model_validate(
            {**template.model_dump(mode="python"), "seeds": rotated}
        )
        for template in templates
    ]


def plan_child_wave(
    analysis: WaveAnalysisV2,
    selection: CandidateSelectionV2,
    parent_experiments: list[ExperimentSpecV2],
    resolved_configs: Mapping[str, Mapping[str, Any]],
    policy: WavePlannerPolicyV2,
) -> ChildWavePlanV2:
    """Create a shadow plan; this function performs no filesystem or queue writes."""

    experiments, experiment_map = _parent_experiment_context(
        analysis, selection, parent_experiments, policy
    )
    input_hash = _plan_input_hash(analysis, selection, experiments, policy)
    child_wave_id = f"wave-{input_hash[:24]}"

    fallback_reasons = {
        "NO_ELIGIBLE_CANDIDATES",
        "NO_CONTINUATION_CANDIDATES",
        "INSUFFICIENT_DIVERSE_CANDIDATES",
    }
    use_control_fallback = (
        policy.allow_control_fallback_when_no_candidates
        and policy.plan_mode == WavePlanMode.EVOLUTION_EXPERIMENT
        and analysis.planning_allowed
        and not selection.planning_allowed
        and bool(set(selection.reason_codes).intersection(fallback_reasons))
        and not (set(selection.reason_codes) - fallback_reasons)
    )
    recovery_analysis_reasons = {
        "EXPERIMENT_HEALTH_BLOCKED",
        "NO_ELIGIBLE_CANDIDATES",
    }
    recovery_selection_reasons = {
        "ANALYSIS_BLOCKED",
        "NO_ELIGIBLE_CANDIDATES",
        "NO_CONTINUATION_CANDIDATES",
        "INSUFFICIENT_DIVERSE_CANDIDATES",
    }
    use_control_recovery = (
        policy.allow_control_recovery_after_technical_failure
        and policy.plan_mode == WavePlanMode.EVOLUTION_EXPERIMENT
        and not analysis.planning_allowed
        and not analysis.has_eligible_candidates
        and set(analysis.reason_codes).issubset(recovery_analysis_reasons)
        and "EXPERIMENT_HEALTH_BLOCKED" in analysis.reason_codes
        and set(selection.reason_codes).issubset(recovery_selection_reasons)
        and not selection.selected_candidate_keys
    )
    if (
        not selection.planning_allowed
        and not use_control_fallback
        and not use_control_recovery
    ):
        return _blocked_child_plan(
            analysis,
            selection,
            policy,
            input_hash=input_hash,
            child_wave_id=child_wave_id,
        )

    controls = [item for item in experiments if item.arm_type == ExperimentArmType.CONTROL]
    control: ExperimentSpecV2 | None = None
    if policy.plan_mode == WavePlanMode.EVOLUTION_EXPERIMENT:
        if len(controls) != 1:
            raise WavePlanningError(
                "evolution planner requires exactly one parent CONTROL experiment"
            )
        control = controls[0]
    use_control_only = use_control_fallback or use_control_recovery
    selected = [] if use_control_only else _selected_candidates(analysis, selection)
    templates = (
        [
            template
            for template in policy.arm_templates
            if template.source_mode == PlannerSourceMode.BASELINE_CONTROL
        ]
        if use_control_only
        else policy.arm_templates
    )
    templates = _rotate_template_seeds(
        templates,
        parent_wave_id=analysis.wave_id,
        policy=policy,
    )
    planned = _materialize_experiments(
        child_wave_id=child_wave_id,
        templates=templates,
        control=control,
        selected=selected,
        experiment_map=experiment_map,
        resolved_configs=resolved_configs,
    )
    attempt_count = sum(len(item.attempts) for item in planned)
    if attempt_count > policy.budget.max_attempts:
        raise WavePlanningError(
            f"planned {attempt_count} attempts exceed budget {policy.budget.max_attempts}"
        )
    return ChildWavePlanV2(
        wave_id=child_wave_id,
        parent_wave_id=analysis.wave_id,
        created_at=analysis.created_at,
        analysis_hash=analysis.analysis_hash,
        selection_hash=selection.selection_hash,
        planner_policy=policy,
        planner_policy_hash=policy.policy_hash,
        plan_input_hash=input_hash,
        result_policy_version=policy.child_result_policy_version,
        search_space_version=policy.child_search_space_version,
        budget=policy.budget,
        experiments=planned,
        planning_allowed=True,
        reason_codes=[
            "PLAN_CONTROL_RECOVERY"
            if use_control_recovery
            else "PLAN_CONTROL_FALLBACK"
            if use_control_fallback
            else "PLAN_COMPLETE"
        ],
    )


def select_and_plan_child_wave(
    analysis: WaveAnalysisV2,
    selection_policy: CandidateSelectionPolicyV2,
    parent_experiments: list[ExperimentSpecV2],
    resolved_configs: Mapping[str, Mapping[str, Any]],
    planner_policy: WavePlannerPolicyV2,
    *,
    excluded_continuation_phenotype_hashes: list[str] | None = None,
) -> tuple[CandidateSelectionV2, ChildWavePlanV2]:
    selection = select_wave_candidates(
        analysis,
        selection_policy,
        excluded_continuation_phenotype_hashes=(
            excluded_continuation_phenotype_hashes
        ),
    )
    plan = plan_child_wave(
        analysis,
        selection,
        parent_experiments,
        resolved_configs,
        planner_policy,
    )
    return selection, plan
