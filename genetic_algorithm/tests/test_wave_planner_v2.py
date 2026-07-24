"""Tests for deterministic, typed and non-executing child-wave plans."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from genetic_algorithm.orchestration.artifact_store_v2 import canonical_config_hash
from genetic_algorithm.orchestration.candidate_selector_v2 import (
    CandidateSelectionPolicyV2,
    select_wave_candidates,
)
from genetic_algorithm.orchestration.wave_analyzer_v2 import (
    AnalysisHealthStatus,
    CandidateAnalysisV2,
    CandidateEligibilityStatus,
    CandidateObservationRefV2,
    ExperimentAnalysisV2,
    WaveAnalysisV2,
    WaveAnalyzerPolicyV2,
)
from genetic_algorithm.orchestration.wave_planner_v2 import (
    PlannerSourceMode,
    WaveArmTemplateV2,
    WavePlanMode,
    WavePlannerPolicyV2,
    WavePlanningError,
    plan_child_wave,
)
from genetic_algorithm.orchestration.wave_state_v2 import (
    AttemptExpectationV2,
    ExperimentArmType,
    ExperimentSpecV2,
    WaveBudgetV2,
)


NOW = datetime(2026, 7, 21, 22, 0, tzinfo=UTC)
WAVE_ID = "wave-planner-parent"
EXPERIMENT_ID = "control-planner-parent"
PANEL_HASH = "p" * 64
BASE_CONFIG = {
    "ga": {"population_size": 100, "mutation_rate": 0.20},
    "features": {"sis_enabled": False},
}
BASE_CONFIG_HASH = canonical_config_hash(BASE_CONFIG)


def _candidate() -> CandidateAnalysisV2:
    return CandidateAnalysisV2(
        experiment_id=EXPERIMENT_ID,
        phenotype_hash="f" * 64,
        candidate_ids=["candidate-1", "candidate-2"],
        seeds=[11, 22],
        observation_refs=[
            CandidateObservationRefV2(
                attempt_id="attempt-parent-11",
                candidate_id="candidate-1",
                seed=11,
                result_path="/artifacts/planner/11/result.json",
                result_sha256="1" * 64,
            ),
            CandidateObservationRefV2(
                attempt_id="attempt-parent-22",
                candidate_id="candidate-2",
                seed=22,
                result_path="/artifacts/planner/22/result.json",
                result_sha256="2" * 64,
            ),
        ],
        observation_count=2,
        valid_observation_count=2,
        eligibility_status=CandidateEligibilityStatus.ELIGIBLE,
        reason_codes=["CANDIDATE_ELIGIBLE"],
        comparison_panel_hash=PANEL_HASH,
        median_robust_score=0.02,
        worst_annualized_return_lcb=0.05,
        median_annualized_return_lcb=0.07,
        worst_max_drawdown_ucb=0.12,
        worst_daily_es5_ucb=0.02,
        worst_net_expectancy_lcb=0.001,
        median_profit_factor=1.5,
        median_win_rate=0.6,
        min_effective_sample_size=40,
        min_trades_per_active_month=10,
        scenario_trade_count_sum=120,
    )


def _analysis(
    *,
    planning_allowed: bool = True,
    with_candidate: bool = True,
) -> WaveAnalysisV2:
    analyzer_policy = WaveAnalyzerPolicyV2(
        analysis_policy_version="analysis-planner-test",
        required_result_policy_version="result-planner-test",
    )
    candidate = _candidate()
    return WaveAnalysisV2(
        wave_id=WAVE_ID,
        created_at=NOW,
        snapshot_hash="s" * 64,
        result_policy_version="result-planner-test",
        analyzer_policy=analyzer_policy,
        analyzer_policy_hash=analyzer_policy.policy_hash,
        experiments=[
            ExperimentAnalysisV2(
                experiment_id=EXPERIMENT_ID,
                arm_type=ExperimentArmType.CONTROL,
                expected_attempt_count=2,
                verified_result_count=2,
                successful_attempt_count=2,
                documented_abort_count=0,
                candidate_observation_count=2,
                abort_fraction=0,
                non_success_fraction=0,
                health_status=AnalysisHealthStatus.HEALTHY,
                reason_codes=["EXPERIMENT_HEALTHY"],
            )
        ],
        candidates=[candidate] if with_candidate else [],
        planning_allowed=planning_allowed,
        has_eligible_candidates=with_candidate,
        reason_codes=(
            ["ANALYSIS_COMPLETE"]
            if planning_allowed and with_candidate
            else ["NO_ELIGIBLE_CANDIDATES"]
            if planning_allowed
            else ["ANALYSIS_BLOCKED"]
        ),
    )


def _parent_experiment() -> ExperimentSpecV2:
    return ExperimentSpecV2(
        experiment_id=EXPERIMENT_ID,
        wave_id=WAVE_ID,
        arm_type=ExperimentArmType.CONTROL,
        hypothesis="Parent control anchor.",
        primary_metric="annualized_net_return_lcb",
        seeds=[11, 22],
        resolved_config_hash=BASE_CONFIG_HASH,
        expected_attempts=[
            AttemptExpectationV2(
                attempt_id="attempt-parent-11",
                manifest_hash="a" * 64,
                seed=11,
                ordinal=0,
            ),
            AttemptExpectationV2(
                attempt_id="attempt-parent-22",
                manifest_hash="b" * 64,
                seed=22,
                ordinal=1,
            ),
        ],
        created_at=NOW,
    )


def _selection(analysis: WaveAnalysisV2):
    return select_wave_candidates(
        analysis,
        CandidateSelectionPolicyV2(
            selection_policy_version="selection-planner-test",
            max_selected=1,
            min_selected=1,
            max_per_experiment=1,
            min_normalized_objective_distance=0,
        ),
    )


def _planner_policy(
    *,
    replication_delta: dict | None = None,
    max_attempts: int = 4,
) -> WavePlannerPolicyV2:
    delta = replication_delta or {"ga.mutation_rate": 0.25}
    return WavePlannerPolicyV2(
        planner_policy_version="planner-policy-test",
        child_result_policy_version="result-planner-test",
        child_search_space_version="search-space-child-test",
        budget=WaveBudgetV2(
            max_attempts=max_attempts,
            max_parallel=min(2, max_attempts),
            max_wallclock_seconds=7200,
        ),
        arm_templates=[
            WaveArmTemplateV2(
                arm_id="control",
                arm_type=ExperimentArmType.CONTROL,
                source_mode=PlannerSourceMode.BASELINE_CONTROL,
                hypothesis="Unchanged paired control.",
                primary_metric="annualized_net_return_lcb",
                seeds=[101, 102],
            ),
            WaveArmTemplateV2(
                arm_id="exploit",
                arm_type=ExperimentArmType.EXPLOIT,
                source_mode=PlannerSourceMode.SELECTED_CANDIDATES,
                hypothesis="Test one explicit delta around the selected phenotype.",
                primary_metric="annualized_net_return_lcb",
                factor_delta=delta,
                seeds=[101, 102],
            ),
        ],
    )


def _plan(*, policy: WavePlannerPolicyV2 | None = None):
    analysis = _analysis()
    selection = _selection(analysis)
    return plan_child_wave(
        analysis,
        selection,
        [_parent_experiment()],
        {BASE_CONFIG_HASH: BASE_CONFIG},
        policy or _planner_policy(),
    )


def test_plan_is_deterministic_and_uses_paired_seed_attempts():
    first = _plan()
    second = _plan()

    assert first == second
    assert first.plan_hash == second.plan_hash
    assert first.wave_id == second.wave_id
    assert first.planning_allowed is True
    assert len(first.experiments) == 2
    assert sum(len(item.attempts) for item in first.experiments) == 4
    assert all(item.seeds == [101, 102] for item in first.experiments)
    assert len(
        {attempt.attempt_id for item in first.experiments for attempt in item.attempts}
    ) == 4


def test_control_is_unchanged_and_selected_arm_has_exact_declared_delta():
    plan = _plan()
    by_arm = {item.arm_id: item for item in plan.experiments}

    assert by_arm["control"].resolved_config == BASE_CONFIG
    assert by_arm["control"].resolved_config_hash == BASE_CONFIG_HASH
    assert by_arm["exploit"].resolved_config["ga"]["mutation_rate"] == 0.25
    assert by_arm["exploit"].resolved_config["ga"]["population_size"] == 100
    assert BASE_CONFIG["ga"]["mutation_rate"] == 0.20
    assert by_arm["exploit"].source.phenotype_hash == "f" * 64
    assert by_arm["exploit"].source.observation_ref is not None
    assert by_arm["exploit"].source.observation_ref.seed == 11


def test_unknown_or_type_changing_factor_delta_is_rejected():
    with pytest.raises(WavePlanningError, match="does not exist"):
        _plan(policy=_planner_policy(replication_delta={"ga.unknown": 1}))

    with pytest.raises(WavePlanningError, match="changes field type"):
        _plan(policy=_planner_policy(replication_delta={"ga.mutation_rate": "high"}))


def test_missing_or_tampered_parent_config_is_rejected():
    analysis = _analysis()
    selection = _selection(analysis)
    experiment = _parent_experiment()

    with pytest.raises(WavePlanningError, match="missing resolved parent config"):
        plan_child_wave(analysis, selection, [experiment], {}, _planner_policy())
    with pytest.raises(WavePlanningError, match="hash differs"):
        plan_child_wave(
            analysis,
            selection,
            [experiment],
            {BASE_CONFIG_HASH: {"ga": {"mutation_rate": 0.99}}},
            _planner_policy(),
        )


def test_attempt_budget_is_enforced_before_proposal():
    with pytest.raises(WavePlanningError, match="exceed budget"):
        _plan(policy=_planner_policy(max_attempts=3))


def test_blocked_selection_creates_non_executable_block_decision():
    analysis = _analysis(planning_allowed=False)
    selection = _selection(analysis)
    plan = plan_child_wave(
        analysis,
        selection,
        [_parent_experiment()],
        {},
        _planner_policy(),
    )
    decision = plan.to_decision(analysis_decision_hash="d" * 64)

    assert plan.planning_allowed is False
    assert not plan.experiments
    assert decision.decision_type.value == "BLOCK"
    assert decision.payload["planning_allowed"] is False


def test_control_fallback_keeps_healthy_search_running_without_candidates():
    analysis = _analysis(with_candidate=False)
    selection = _selection(analysis)
    policy = _planner_policy().model_copy(
        update={"allow_control_fallback_when_no_candidates": True}
    )

    plan = plan_child_wave(
        analysis,
        selection,
        [_parent_experiment()],
        {BASE_CONFIG_HASH: BASE_CONFIG},
        policy,
    )

    assert plan.planning_allowed is True
    assert plan.reason_codes == ["PLAN_CONTROL_FALLBACK"]
    assert len(plan.experiments) == 1
    assert plan.experiments[0].arm_type == ExperimentArmType.CONTROL
    assert plan.experiments[0].source.source_mode == PlannerSourceMode.BASELINE_CONTROL


def test_control_fallback_rotates_seed_panel_per_parent_wave():
    analysis = _analysis(with_candidate=False)
    policy = _planner_policy().model_copy(
        update={
            "allow_control_fallback_when_no_candidates": True,
            "rotate_seeds_per_parent_wave": True,
        }
    )
    first = plan_child_wave(
        analysis,
        _selection(analysis),
        [_parent_experiment()],
        {BASE_CONFIG_HASH: BASE_CONFIG},
        policy,
    )
    repeated = plan_child_wave(
        analysis,
        _selection(analysis),
        [_parent_experiment()],
        {BASE_CONFIG_HASH: BASE_CONFIG},
        policy,
    )
    next_analysis = analysis.model_copy(update={"wave_id": "wave-planner-next"})
    next_parent = _parent_experiment().model_copy(
        update={"wave_id": "wave-planner-next"}
    )
    next_wave = plan_child_wave(
        next_analysis,
        _selection(next_analysis),
        [next_parent],
        {BASE_CONFIG_HASH: BASE_CONFIG},
        policy,
    )

    assert first == repeated
    assert first.experiments[0].seeds != [101, 102]
    assert first.experiments[0].seeds != next_wave.experiments[0].seeds
    assert all(0 <= seed <= 2**32 - 1 for seed in first.experiments[0].seeds)


def test_control_recovery_restarts_scratch_after_bounded_technical_failure():
    analysis = _analysis(
        planning_allowed=False,
        with_candidate=False,
    ).model_copy(
        update={
            "reason_codes": [
                "EXPERIMENT_HEALTH_BLOCKED",
                "NO_ELIGIBLE_CANDIDATES",
            ]
        }
    )
    selection = _selection(analysis)
    policy = _planner_policy().model_copy(
        update={"allow_control_recovery_after_technical_failure": True}
    )

    plan = plan_child_wave(
        analysis,
        selection,
        [_parent_experiment()],
        {BASE_CONFIG_HASH: BASE_CONFIG},
        policy,
    )

    assert plan.planning_allowed is True
    assert plan.reason_codes == ["PLAN_CONTROL_RECOVERY"]
    assert len(plan.experiments) == 1
    assert plan.experiments[0].arm_type == ExperimentArmType.CONTROL


def test_allowed_plan_requires_materialization_before_proposal_decision():
    plan = _plan()

    with pytest.raises(WavePlanningError, match="must be materialized"):
        plan.to_decision(analysis_decision_hash="d" * 64)


def test_replay_validation_mode_has_no_synthetic_control_and_uses_selected_source():
    analysis = _analysis()
    selection = _selection(analysis)
    policy = WavePlannerPolicyV2(
        planner_policy_version="replay-validation-planner-test",
        child_result_policy_version="result-planner-test",
        child_search_space_version="replay-panel-test",
        plan_mode=WavePlanMode.REPLAY_VALIDATION,
        budget=WaveBudgetV2(
            max_attempts=2,
            max_parallel=1,
            max_wallclock_seconds=3600,
        ),
        arm_templates=[
            WaveArmTemplateV2(
                arm_id="cross-pair-validation",
                arm_type=ExperimentArmType.VALIDATION,
                source_mode=PlannerSourceMode.SELECTED_CANDIDATES,
                hypothesis="Replay the frozen candidate on a declared new panel.",
                primary_metric="annualized_net_return_lcb",
                seeds=[201, 202],
            )
        ],
    )

    plan = plan_child_wave(
        analysis,
        selection,
        [_parent_experiment()],
        {BASE_CONFIG_HASH: BASE_CONFIG},
        policy,
    )

    assert plan.planning_allowed is True
    assert len(plan.experiments) == 1
    assert plan.experiments[0].arm_type == ExperimentArmType.VALIDATION
    assert plan.experiments[0].source.source_mode == PlannerSourceMode.SELECTED_CANDIDATES


def test_planner_policy_requires_paired_seeds_across_arms():
    policy = _planner_policy().model_dump(mode="python")
    policy["arm_templates"][1]["seeds"] = [201, 202]

    with pytest.raises(ValidationError, match="identical paired seeds"):
        WavePlannerPolicyV2.model_validate(policy)
