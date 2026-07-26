"""Tests for deterministic and fail-closed parent-wave analysis."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from genetic_algorithm.orchestration.artifact_store_v2 import (
    ArtifactIntegrityError,
    V2ArtifactStore,
    canonical_config_hash,
)
from genetic_algorithm.orchestration.attempt_state_v2 import AttemptLifecycleStatus
from genetic_algorithm.orchestration.result_contract import (
    AttemptManifestV2,
    AttemptResultV2,
    AttemptStatus,
    BacktestRecordV2,
    CandidateEvaluationV2,
    EvaluationStatus,
    GateResultV2,
    ScenarioMetricsV2,
    ScenarioRole,
)
from genetic_algorithm.orchestration.wave_analyzer_v2 import (
    AnalysisHealthStatus,
    CandidateEligibilityStatus,
    WaveAnalysisError,
    WaveAnalysisV2,
    WaveAnalyzerPolicyV2,
    WaveAnalyzerV2,
    analyze_wave_snapshot,
)
from genetic_algorithm.orchestration.wave_comparison_v2 import (
    WaveComparisonError,
    build_verified_wave_comparison,
)
from genetic_algorithm.orchestration.wave_state_v2 import (
    AttemptEvidenceType,
    AttemptExpectationV2,
    ExperimentArmType,
    ExperimentSpecV2,
    WaveAttemptSnapshotV2,
    WaveBudgetV2,
    WaveLifecycleStatus,
    WaveResultSnapshotV2,
    WaveSpecV2,
    WaveStateStoreV2,
)
from genetic_algorithm.scripts.wave_comparison import (
    format_verified_comparison,
)
from genetic_algorithm.scripts.wave_comparison import (
    main as wave_comparison_main,
)
from genetic_algorithm.tests.v2_metric_fixtures import expectancy_metrics


NOW = datetime(2026, 7, 21, 20, 0, tzinfo=UTC)
WAVE_ID = "wave-analyzer-test"
EXPERIMENT_ID = "control-analyzer-test"
RESULT_POLICY = "result-policy-analyzer-test"
CONFIG = {"experiment": EXPERIMENT_ID, "mode": "shadow"}
PHENOTYPE_HASH = "p" * 64


def _policy(**updates) -> WaveAnalyzerPolicyV2:
    values = {
        "analysis_policy_version": "analysis-policy-test",
        "required_result_policy_version": RESULT_POLICY,
        "min_candidate_seed_count": 2,
    }
    values.update(updates)
    return WaveAnalyzerPolicyV2(**values)


def _metrics(seed: int) -> ScenarioMetricsV2:
    annual_lcb = 0.04 if seed == 11 else 0.06
    return ScenarioMetricsV2(
        scenario_id="temporal-btc",
        pair="BTC/USDT",
        timeframe="1h",
        role=ScenarioRole.TEMPORAL_VALIDATION,
        period_start=date(2025, 1, 1),
        period_end=date(2025, 7, 1),
        cost_multiplier=1.0,
        status=EvaluationStatus.VALID,
        success=True,
        net_return=0.08,
        annualized_net_return=0.12,
        annualized_net_return_lcb=annual_lcb,
        max_drawdown=0.10,
        max_drawdown_ucb=0.15,
        daily_expected_shortfall_5=0.01,
        daily_expected_shortfall_5_ucb=0.02,
        net_expectancy=0.003,
        net_expectancy_lcb=0.001,
        profit_factor=1.5,
        profit_factor_censored=False,
        profit_factor_contract_version="right-censored-profit-factor-v1",
        win_rate=0.6,
        trade_count=60,
        effective_sample_size=40.0,
        **expectancy_metrics(
            trade_count=60,
            mean_return=0.003,
            lower_confidence_bound=0.001,
            effective_sample_size=40.0,
        ),
        active_months=6,
        max_consecutive_losses=3,
        max_drawdown_duration_days=20.0,
    )


def _persist_success(
    tmp_path: Path,
    *,
    attempt_id: str,
    seed: int,
    ordinal: int,
    gate_passed: bool = True,
) -> tuple[AttemptManifestV2, AttemptResultV2, WaveAttemptSnapshotV2]:
    root = tmp_path / "attempts" / attempt_id
    manifest = AttemptManifestV2(
        attempt_id=attempt_id,
        wave_id=WAVE_ID,
        experiment_id=EXPERIMENT_ID,
        created_at=NOW + timedelta(seconds=1),
        config_hash=canonical_config_hash(CONFIG),
        code_version="analyzer-test-commit",
        data_manifest_hash="d" * 64,
        split_manifest_hash="s" * 64,
        fitness_policy_version=RESULT_POLICY,
        seeds=[seed],
        worker_count=1,
        resolved_config_path=str(root / V2ArtifactStore.CONFIG_NAME),
        artifact_root=str(root),
    )
    artifacts = V2ArtifactStore(root)
    artifacts.write_manifest(manifest, CONFIG)
    candidate_id = f"candidate-{seed}"
    trades = [
        {"pair": "BTC/USDT", "profit_ratio": 0.01 if index % 3 else -0.005} for index in range(60)
    ]
    record = BacktestRecordV2(
        attempt_id=attempt_id,
        wave_id=WAVE_ID,
        experiment_id=EXPERIMENT_ID,
        candidate_id=candidate_id,
        config_hash=manifest.config_hash,
        phenotype_hash=PHENOTYPE_HASH,
        code_version=manifest.code_version,
        data_manifest_hash=manifest.data_manifest_hash,
        fitness_policy_version=RESULT_POLICY,
        seed=seed,
        worker_count=1,
        fee_rate=0.001,
        slippage_rate=0.0005,
        spread_rate=0.0,
        funding_rate=0.0,
        equity_method="MARK_TO_MARKET",
        metrics=_metrics(seed),
        daily_net_returns=[0.01, -0.005, 0.004],
        equity_curve=[100.0, 101.0, 100.495, 100.897],
        trades=trades,
    )
    gate = GateResultV2(
        gate_id="RISK_RETURN_GATE",
        passed=gate_passed,
        status=EvaluationStatus.VALID,
        observed=0.04,
        threshold=0.0,
        operator=">=",
        reason_code="PASS" if gate_passed else "RETURN_LCB_TOO_LOW",
    )
    candidate = CandidateEvaluationV2(
        candidate_id=candidate_id,
        phenotype_hash=PHENOTYPE_HASH,
        fitness_policy_version=RESULT_POLICY,
        status=EvaluationStatus.VALID,
        scenarios=[record],
        gates=[gate],
        robust_score=0.01 if seed == 11 else 0.03,
    )
    artifacts.write_backtest(record)
    artifacts.write_candidate(candidate)
    started_at = NOW + timedelta(seconds=5 + ordinal * 3)
    result = artifacts.finalize(
        {
            "attempt_id": attempt_id,
            "status": AttemptStatus.SUCCEEDED,
            "started_at": started_at,
            "finished_at": started_at + timedelta(milliseconds=200),
            "manifest": manifest.model_dump(mode="json"),
            "candidate_evaluations": [candidate.model_dump(mode="json")],
        }
    )
    result_path = root / V2ArtifactStore.RESULT_NAME
    evidence = WaveAttemptSnapshotV2(
        attempt_id=attempt_id,
        experiment_id=EXPERIMENT_ID,
        manifest_hash=canonical_config_hash(manifest.model_dump(mode="json")),
        seed=seed,
        ordinal=ordinal,
        lifecycle_status=AttemptLifecycleStatus.SUCCEEDED,
        evidence=AttemptEvidenceType.VERIFIED_RESULT,
        result_status=AttemptStatus.SUCCEEDED,
        result_path=str(result_path.resolve()),
        result_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest(),
        result_finished_at=result.finished_at,
        attempt_state_version=7,
    )
    return manifest, result, evidence


def _inputs(
    tmp_path: Path,
    *,
    second_gate_passed: bool = True,
    arm_type: ExperimentArmType = ExperimentArmType.CONTROL,
) -> tuple[
    WaveResultSnapshotV2,
    ExperimentSpecV2,
    list[AttemptManifestV2],
]:
    first_manifest, _, first = _persist_success(
        tmp_path,
        attempt_id="attempt-a",
        seed=11,
        ordinal=0,
    )
    second_manifest, _, second = _persist_success(
        tmp_path,
        attempt_id="attempt-b",
        seed=22,
        ordinal=1,
        gate_passed=second_gate_passed,
    )
    snapshot = WaveResultSnapshotV2(
        wave_id=WAVE_ID,
        created_at=NOW + timedelta(seconds=10),
        policy_version=RESULT_POLICY,
        search_space_version="search-space-analyzer-test",
        attempt_results=[first, second],
    )
    experiment = ExperimentSpecV2(
        experiment_id=EXPERIMENT_ID,
        wave_id=WAVE_ID,
        arm_type=arm_type,
        hypothesis="The control remains a stable comparison anchor.",
        primary_metric="annualized_net_return_lcb",
        seeds=[11, 22],
        resolved_config_hash=canonical_config_hash(CONFIG),
        expected_attempts=[
            AttemptExpectationV2(
                attempt_id=first.attempt_id,
                manifest_hash=first.manifest_hash,
                seed=first.seed,
                ordinal=first.ordinal,
            ),
            AttemptExpectationV2(
                attempt_id=second.attempt_id,
                manifest_hash=second.manifest_hash,
                seed=second.seed,
                ordinal=second.ordinal,
            ),
        ],
        created_at=NOW + timedelta(seconds=1),
    )
    return snapshot, experiment, [first_manifest, second_manifest]


def test_analysis_is_deterministic_and_aggregates_profit_risk_and_activity(tmp_path: Path):
    snapshot, experiment, _ = _inputs(tmp_path)

    first = analyze_wave_snapshot(snapshot, [experiment], _policy())
    second = analyze_wave_snapshot(snapshot, [experiment], _policy())

    assert first == second
    assert first.analysis_hash == second.analysis_hash
    assert first.planning_allowed is True
    assert first.has_eligible_candidates is True
    candidate = first.candidates[0]
    assert candidate.eligibility_status == CandidateEligibilityStatus.ELIGIBLE
    assert candidate.seeds == [11, 22]
    assert candidate.median_robust_score == pytest.approx(0.02)
    assert candidate.worst_annualized_return_lcb == pytest.approx(0.04)
    assert candidate.median_annualized_return_lcb == pytest.approx(0.05)
    assert candidate.worst_max_drawdown_ucb == pytest.approx(0.15)
    assert candidate.worst_daily_es5_ucb == pytest.approx(0.02)
    assert candidate.median_profit_factor == pytest.approx(1.5)
    assert candidate.median_win_rate == pytest.approx(0.6)
    assert candidate.min_scenario_net_return == pytest.approx(0.08)
    assert candidate.profitable_scenario_ratio == pytest.approx(1.0)
    assert candidate.max_drawdown_duration_days == pytest.approx(20.0)
    assert candidate.median_gate_alignment_score == pytest.approx(1.0)
    assert candidate.failed_gate_count == 0
    assert candidate.scenario_trade_count_sum == 120
    assert first.to_decision() == second.to_decision()
    assert first.to_decision().input_hash == snapshot.snapshot_hash


def test_result_mutation_after_reconciliation_blocks_analysis(tmp_path: Path):
    snapshot, experiment, _ = _inputs(tmp_path)
    result_path = Path(snapshot.attempt_results[0].result_path or "")
    result_path.write_bytes(result_path.read_bytes() + b"tampered")

    with pytest.raises(ArtifactIntegrityError, match="hash differs from file"):
        analyze_wave_snapshot(snapshot, [experiment], _policy())


def test_documented_abort_is_counted_and_blocks_strict_health_policy(tmp_path: Path):
    snapshot, experiment, _ = _inputs(tmp_path)
    second = snapshot.attempt_results[1]
    aborted = WaveAttemptSnapshotV2(
        attempt_id=second.attempt_id,
        experiment_id=second.experiment_id,
        manifest_hash=second.manifest_hash,
        seed=second.seed,
        ordinal=second.ordinal,
        lifecycle_status=AttemptLifecycleStatus.FAILED,
        evidence=AttemptEvidenceType.DOCUMENTED_ABORT,
        abort_decision_hash="a" * 64,
        attempt_state_version=8,
    )
    with_abort = snapshot.model_copy(
        update={"attempt_results": [snapshot.attempt_results[0], aborted]}
    )

    analysis = analyze_wave_snapshot(with_abort, [experiment], _policy())

    health = analysis.experiments[0]
    assert health.health_status == AnalysisHealthStatus.BLOCKED
    assert health.abort_fraction == pytest.approx(0.5)
    assert "ABORT_FRACTION_EXCEEDED" in health.reason_codes
    assert analysis.planning_allowed is False
    assert "EXPERIMENT_HEALTH_BLOCKED" in analysis.reason_codes


def test_failed_candidate_gate_cannot_be_compensated_by_good_metrics(tmp_path: Path):
    snapshot, experiment, _ = _inputs(tmp_path, second_gate_passed=False)

    analysis = analyze_wave_snapshot(
        snapshot,
        [experiment],
        _policy(require_eligible_candidate_for_planning=True),
    )

    candidate = analysis.candidates[0]
    assert candidate.eligibility_status == CandidateEligibilityStatus.INELIGIBLE
    assert "RETURN_LCB_TOO_LOW" in candidate.failed_gate_reason_codes
    assert candidate.valid_observation_count == 2
    assert candidate.median_annualized_return_lcb == pytest.approx(0.05)
    assert candidate.median_gate_alignment_score == pytest.approx(0.5)
    assert candidate.failed_gate_count == 1
    assert analysis.has_eligible_candidates is False
    assert analysis.planning_allowed is False
    assert "NO_ELIGIBLE_CANDIDATES" in analysis.reason_codes


def test_missing_control_arm_blocks_planning(tmp_path: Path):
    snapshot, experiment, _ = _inputs(tmp_path, arm_type=ExperimentArmType.EXPLORE)

    analysis = analyze_wave_snapshot(snapshot, [experiment], _policy())

    assert analysis.candidates[0].eligibility_status == CandidateEligibilityStatus.ELIGIBLE
    assert analysis.planning_allowed is False
    assert "CONTROL_ARM_MISSING" in analysis.reason_codes


def test_result_policy_mismatch_is_rejected_before_artifact_analysis(tmp_path: Path):
    snapshot, experiment, _ = _inputs(tmp_path)

    with pytest.raises(WaveAnalysisError, match="result policy differs"):
        analyze_wave_snapshot(
            snapshot,
            [experiment],
            _policy(required_result_policy_version="another-policy"),
        )


def test_experiment_config_hash_is_reverified(tmp_path: Path):
    snapshot, experiment, _ = _inputs(tmp_path)
    mismatched = experiment.model_copy(update={"resolved_config_hash": "f" * 64})

    with pytest.raises(WaveAnalysisError, match="result config differs"):
        analyze_wave_snapshot(snapshot, [mismatched], _policy())


def _reconciled_store(
    tmp_path: Path,
) -> WaveStateStoreV2:
    snapshot, experiment, manifests = _inputs(tmp_path)
    store = WaveStateStoreV2(tmp_path / "orchestration.sqlite3")
    store.register_wave(
        WaveSpecV2(
            wave_id=WAVE_ID,
            policy_version=RESULT_POLICY,
            search_space_version=snapshot.search_space_version,
            budget=WaveBudgetV2(
                max_attempts=2,
                max_parallel=1,
                max_wallclock_seconds=3600,
            ),
            created_at=NOW,
        )
    )
    for manifest in manifests:
        store.register(manifest)
        store.transition(
            manifest.attempt_id,
            AttemptLifecycleStatus.VALIDATED,
            occurred_at=NOW + timedelta(seconds=2),
            actor="validator",
            reason="VALIDATED",
        )
        store.transition(
            manifest.attempt_id,
            AttemptLifecycleStatus.QUEUED,
            occurred_at=NOW + timedelta(seconds=3),
            actor="controller",
            reason="QUEUED",
        )
    store.register_experiment(experiment)
    store.begin_collection(WAVE_ID, started_at=NOW + timedelta(seconds=4))
    for index, manifest in enumerate(manifests):
        claimed_at = NOW + timedelta(seconds=5 + index * 3)
        claimed = store.claim_next(
            worker_id="analyzer-test-worker",
            claimed_at=claimed_at,
            lease_seconds=60,
        )
        assert claimed is not None
        assert claimed.attempt_id == manifest.attempt_id
        log_path = Path(manifest.artifact_root) / "runtime" / "attempt.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("analyzer integration\n")
        store.prepare_execution(
            manifest.attempt_id,
            claim_token=str(claimed.claim_token),
            command_hash="e" * 64,
            working_directory=tmp_path,
            log_path=log_path,
            prepared_at=claimed_at + timedelta(milliseconds=100),
        )
        token = f"process-{index}"
        store.mark_running(
            manifest.attempt_id,
            claim_token=str(claimed.claim_token),
            pid=1200 + index,
            process_start_token=token,
            started_at=claimed_at + timedelta(milliseconds=200),
            lease_seconds=60,
        )
        store.finalize_from_artifacts(
            manifest.attempt_id,
            claim_token=str(claimed.claim_token),
            process_start_token=token,
            finalized_at=claimed_at + timedelta(seconds=1),
        )
    reconciled = store.reconcile_wave(
        WAVE_ID,
        reconciled_at=NOW + timedelta(seconds=10),
    )
    assert reconciled.status == WaveLifecycleStatus.RECONCILED
    return store


def test_store_adapter_records_one_idempotent_hash_chained_analysis(tmp_path: Path):
    store = _reconciled_store(tmp_path)
    analyzer = WaveAnalyzerV2(store, _policy())

    first_analysis, first_state = analyzer.analyze_and_record(WAVE_ID)
    second_analysis, second_state = analyzer.analyze_and_record(WAVE_ID)

    assert first_analysis == second_analysis
    assert first_state == second_state
    assert first_state.status == WaveLifecycleStatus.ANALYZED
    analysis_decisions = [
        item for item in store.decisions(WAVE_ID) if item.decision_type.value == "ANALYSIS"
    ]
    assert len(analysis_decisions) == 1
    assert analysis_decisions[0].input_hash == first_state.result_snapshot_hash
    assert analysis_decisions[0].payload["analysis_hash"] == first_analysis.analysis_hash


def _analyzed_store(tmp_path: Path) -> WaveStateStoreV2:
    store = _reconciled_store(tmp_path)
    WaveAnalyzerV2(store, _policy()).analyze_and_record(WAVE_ID)
    return store


def test_verified_comparison_rebuilds_the_recorded_structured_analysis(tmp_path: Path):
    store = _analyzed_store(tmp_path)

    comparison = build_verified_wave_comparison(store, WAVE_ID)

    assert comparison.lifecycle_status == WaveLifecycleStatus.ANALYZED
    assert comparison.analysis.wave_id == WAVE_ID
    assert comparison.analysis.candidates[0].median_profit_factor == pytest.approx(1.5)
    assert comparison.snapshot_hash == store.get_wave(WAVE_ID).result_snapshot_hash
    assert len(comparison.comparison_hash) == 64


def test_log_lines_cannot_change_verified_wave_comparison(tmp_path: Path):
    store = _analyzed_store(tmp_path)
    before = build_verified_wave_comparison(store, WAVE_ID)
    for log_path in (tmp_path / "attempts").glob("*/runtime/attempt.log"):
        log_path.write_text(
            log_path.read_text()
            + "EVOLUTION COMPLETE profit=999999% sharpe=999 drawdown=0 trades=999999\n"
            + "another candidate profit=-999999% trades=1\n"
        )

    after = build_verified_wave_comparison(store, WAVE_ID)

    assert after == before
    assert after.comparison_hash == before.comparison_hash


def test_result_mutation_after_recorded_analysis_blocks_comparison(tmp_path: Path):
    store = _analyzed_store(tmp_path)
    snapshot = store.get_wave(WAVE_ID).result_snapshot
    assert snapshot is not None
    result_path = Path(snapshot.attempt_results[0].result_path or "")
    result_path.write_bytes(result_path.read_bytes() + b"tampered")

    with pytest.raises(ArtifactIntegrityError, match="hash differs from file"):
        build_verified_wave_comparison(store, WAVE_ID)


def test_self_consistent_forged_analysis_is_rejected_by_result_rebuild(tmp_path: Path):
    store = _analyzed_store(tmp_path)
    decision = next(
        item for item in store.decisions(WAVE_ID) if item.decision_type.value == "ANALYSIS"
    )
    analysis_payload = dict(decision.payload["analysis"])
    candidate_payloads = [dict(item) for item in analysis_payload["candidates"]]
    candidate_payloads[0]["median_profit_factor"] = 999.0
    analysis_payload["candidates"] = candidate_payloads
    forged_analysis = WaveAnalysisV2.model_validate(analysis_payload)
    forged_decision = forged_analysis.to_decision()
    with sqlite3.connect(tmp_path / "orchestration.sqlite3") as connection:
        # Simulate a fully compromised external DB writer which bypassed the
        # normal immutable-trigger boundary and recomputed the row hash.
        connection.execute("DROP TRIGGER trg_wave_decisions_no_update")
        connection.execute(
            """
            UPDATE wave_decisions
            SET decision_id = ?, decision_json = ?, decision_hash = ?
            WHERE decision_id = ?
            """,
            (
                forged_decision.decision_id,
                json.dumps(
                    forged_decision.model_dump(mode="json"),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                forged_decision.decision_hash,
                decision.decision_id,
            ),
        )

    with pytest.raises(WaveComparisonError, match="differs from current verified"):
        build_verified_wave_comparison(store, WAVE_ID)


def test_comparison_without_recorded_analysis_fails_closed(tmp_path: Path):
    store = _reconciled_store(tmp_path)

    with pytest.raises(WaveComparisonError, match="exactly one"):
        build_verified_wave_comparison(store, WAVE_ID)


def test_wave_comparison_cli_exports_only_verified_analysis(
    tmp_path: Path,
    capsys,
):
    store = _analyzed_store(tmp_path)
    comparison = build_verified_wave_comparison(store, WAVE_ID)

    exit_code = wave_comparison_main(
        [
            WAVE_ID,
            "--state-path",
            str(tmp_path / "orchestration.sqlite3"),
            "--json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["comparison_hash"] == comparison.comparison_hash
    assert payload["analysis"]["snapshot_hash"] == comparison.snapshot_hash
    assert payload["analysis"]["candidates"][0]["median_profit_factor"] == pytest.approx(1.5)
    assert "log_path" not in json.dumps(payload)


def test_wave_comparison_text_names_metric_uncertainty_bounds(tmp_path: Path):
    comparison = build_verified_wave_comparison(_analyzed_store(tmp_path), WAVE_ID)

    report = format_verified_comparison(comparison)

    assert "Worst LCB" in report
    assert "DD UCB" in report
    assert "ES5 UCB" in report
    assert "VERIFIED WAVE COMPARISON" in report
