"""Tests for non-compensating Pareto ranking and deterministic diversity."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from genetic_algorithm.orchestration.candidate_selector_v2 import (
    CandidateSelectionPolicyV2,
    ParetoObjective,
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
from genetic_algorithm.orchestration.wave_state_v2 import ExperimentArmType


NOW = datetime(2026, 7, 21, 21, 0, tzinfo=UTC)
PANEL = "1" * 64


def _candidate(
    marker: str,
    *,
    experiment_id: str = "control",
    annual_return_lcb: float,
    expectancy_lcb: float,
    drawdown_ucb: float,
    es_ucb: float,
    panel_hash: str = PANEL,
    eligible: bool = True,
) -> CandidateAnalysisV2:
    phenotype_hash = marker * 64
    candidate_ids = [f"candidate-{marker}-1", f"candidate-{marker}-2"]
    refs = [
        CandidateObservationRefV2(
            attempt_id=f"attempt-{marker}-{seed}",
            candidate_id=candidate_id,
            seed=seed,
            result_path=f"/artifacts/selector/{marker}/{seed}/result.json",
            result_sha256=str(seed) * 64,
        )
        for seed, candidate_id in zip((1, 2), candidate_ids, strict=True)
    ]
    return CandidateAnalysisV2(
        experiment_id=experiment_id,
        phenotype_hash=phenotype_hash,
        candidate_ids=candidate_ids,
        seeds=[1, 2],
        observation_refs=refs,
        observation_count=2,
        valid_observation_count=2,
        eligibility_status=(
            CandidateEligibilityStatus.ELIGIBLE
            if eligible
            else CandidateEligibilityStatus.INELIGIBLE
        ),
        reason_codes=(
            ["CANDIDATE_ELIGIBLE"] if eligible else ["RISK_GATE_FAILED"]
        ),
        failed_gate_reason_codes=[] if eligible else ["RISK_GATE_FAILED"],
        comparison_panel_hash=panel_hash,
        median_robust_score=annual_return_lcb - drawdown_ucb - es_ucb,
        worst_annualized_return_lcb=annual_return_lcb,
        median_annualized_return_lcb=annual_return_lcb,
        worst_max_drawdown_ucb=drawdown_ucb,
        worst_daily_es5_ucb=es_ucb,
        worst_net_expectancy_lcb=expectancy_lcb,
        median_profit_factor=1.5,
        median_win_rate=0.6,
        min_effective_sample_size=40,
        min_trades_per_active_month=10,
        scenario_trade_count_sum=120,
    )


def _analysis(
    candidates: list[CandidateAnalysisV2],
    *,
    planning_allowed: bool = True,
) -> WaveAnalysisV2:
    experiment_ids = sorted({item.experiment_id for item in candidates} or {"control"})
    experiments = [
        ExperimentAnalysisV2(
            experiment_id=experiment_id,
            arm_type=(
                ExperimentArmType.CONTROL
                if experiment_id == "control"
                else ExperimentArmType.EXPLORE
            ),
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
        for experiment_id in experiment_ids
    ]
    analyzer_policy = WaveAnalyzerPolicyV2(
        analysis_policy_version="analysis-selector-test",
        required_result_policy_version="result-selector-test",
    )
    ordered = sorted(candidates, key=lambda item: (item.experiment_id, item.phenotype_hash))
    has_eligible = any(
        item.eligibility_status == CandidateEligibilityStatus.ELIGIBLE
        for item in ordered
    )
    return WaveAnalysisV2(
        wave_id="wave-selector-test",
        created_at=NOW,
        snapshot_hash="s" * 64,
        result_policy_version="result-selector-test",
        analyzer_policy=analyzer_policy,
        analyzer_policy_hash=analyzer_policy.policy_hash,
        experiments=experiments,
        candidates=ordered,
        planning_allowed=planning_allowed,
        has_eligible_candidates=has_eligible,
        reason_codes=["ANALYSIS_COMPLETE"] if planning_allowed else ["ANALYSIS_BLOCKED"],
    )


def _policy(**updates) -> CandidateSelectionPolicyV2:
    values = {
        "selection_policy_version": "selector-policy-test",
        "max_selected": 4,
        "min_selected": 1,
        "max_per_experiment": 4,
        "max_pareto_rank": 2,
        "min_normalized_objective_distance": 0,
    }
    values.update(updates)
    return CandidateSelectionPolicyV2(**values)


def test_dominated_candidate_gets_lower_rank_without_scalar_fitness():
    safe = _candidate(
        "a",
        annual_return_lcb=0.10,
        expectancy_lcb=0.003,
        drawdown_ucb=0.10,
        es_ucb=0.01,
    )
    dominated = _candidate(
        "b",
        annual_return_lcb=0.08,
        expectancy_lcb=0.002,
        drawdown_ucb=0.15,
        es_ucb=0.02,
    )
    aggressive = _candidate(
        "c",
        annual_return_lcb=0.14,
        expectancy_lcb=0.004,
        drawdown_ucb=0.20,
        es_ucb=0.03,
    )

    selection = select_wave_candidates(
        _analysis([safe, dominated, aggressive]),
        _policy(max_selected=2),
    )
    assessments = {item.phenotype_hash: item for item in selection.assessments}

    assert assessments[safe.phenotype_hash].pareto_rank == 0
    assert assessments[aggressive.phenotype_hash].pareto_rank == 0
    assert assessments[dominated.phenotype_hash].pareto_rank == 1
    assert assessments[safe.phenotype_hash].selection_order == 0
    assert assessments[aggressive.phenotype_hash].selected is True
    assert selection.planning_allowed is True


def test_high_return_does_not_compensate_higher_tail_risk_in_dominance():
    safe = _candidate(
        "a",
        annual_return_lcb=0.08,
        expectancy_lcb=0.002,
        drawdown_ucb=0.08,
        es_ucb=0.01,
    )
    aggressive = _candidate(
        "b",
        annual_return_lcb=0.20,
        expectancy_lcb=0.006,
        drawdown_ucb=0.24,
        es_ucb=0.05,
    )

    selection = select_wave_candidates(_analysis([safe, aggressive]), _policy())

    assert {item.pareto_rank for item in selection.assessments} == {0}
    assert selection.assessments[0].objective_values[
        ParetoObjective.MAX_DRAWDOWN_UCB
    ] == pytest.approx(0.08)


def test_selection_is_byte_deterministic():
    candidates = [
        _candidate(
            marker,
            annual_return_lcb=0.08 + index * 0.01,
            expectancy_lcb=0.002 + index * 0.001,
            drawdown_ucb=0.08 + index * 0.02,
            es_ucb=0.01 + index * 0.005,
        )
        for index, marker in enumerate(("a", "b", "c"))
    ]
    analysis = _analysis(candidates)

    first = select_wave_candidates(analysis, _policy(max_selected=2))
    second = select_wave_candidates(analysis, _policy(max_selected=2))

    assert first == second
    assert first.selection_hash == second.selection_hash


def test_candidates_from_different_comparison_panels_are_not_ranked_together():
    first = _candidate(
        "a",
        annual_return_lcb=0.10,
        expectancy_lcb=0.003,
        drawdown_ucb=0.10,
        es_ucb=0.01,
    )
    second = _candidate(
        "b",
        annual_return_lcb=0.12,
        expectancy_lcb=0.004,
        drawdown_ucb=0.12,
        es_ucb=0.02,
        panel_hash="2" * 64,
    )

    selection = select_wave_candidates(_analysis([first, second]), _policy())

    assert selection.planning_allowed is False
    assert "INCOMPARABLE_CANDIDATE_PANELS" in selection.reason_codes
    assert all(item.pareto_rank is None for item in selection.assessments)
    assert not selection.selected_candidate_keys


def test_blocked_analysis_cannot_select_candidates():
    candidate = _candidate(
        "a",
        annual_return_lcb=0.10,
        expectancy_lcb=0.003,
        drawdown_ucb=0.10,
        es_ucb=0.01,
    )

    selection = select_wave_candidates(
        _analysis([candidate], planning_allowed=False),
        _policy(),
    )

    assert selection.planning_allowed is False
    assert "ANALYSIS_BLOCKED" in selection.reason_codes
    assert not selection.selected_candidate_keys


def test_diversity_selection_enforces_per_experiment_limit():
    candidates = [
        _candidate(
            "a",
            experiment_id="control",
            annual_return_lcb=0.10,
            expectancy_lcb=0.003,
            drawdown_ucb=0.10,
            es_ucb=0.01,
        ),
        _candidate(
            "b",
            experiment_id="control",
            annual_return_lcb=0.15,
            expectancy_lcb=0.005,
            drawdown_ucb=0.20,
            es_ucb=0.03,
        ),
        _candidate(
            "c",
            experiment_id="explore",
            annual_return_lcb=0.12,
            expectancy_lcb=0.004,
            drawdown_ucb=0.14,
            es_ucb=0.02,
        ),
    ]

    selection = select_wave_candidates(
        _analysis(candidates),
        _policy(max_selected=3, max_per_experiment=1),
    )
    selected_experiments = [
        item.experiment_id for item in selection.assessments if item.selected
    ]

    assert sorted(selected_experiments) == ["control", "explore"]


def test_ineligible_candidates_never_enter_pareto_pool():
    eligible = _candidate(
        "a",
        annual_return_lcb=0.08,
        expectancy_lcb=0.002,
        drawdown_ucb=0.10,
        es_ucb=0.01,
    )
    failed_gate = _candidate(
        "b",
        annual_return_lcb=0.50,
        expectancy_lcb=0.02,
        drawdown_ucb=0.01,
        es_ucb=0.001,
        eligible=False,
    )

    selection = select_wave_candidates(_analysis([eligible, failed_gate]), _policy())

    assert selection.eligible_candidate_count == 1
    assert selection.ineligible_candidate_count == 1
    assert len(selection.assessments) == 1
    assert selection.assessments[0].phenotype_hash == eligible.phenotype_hash
