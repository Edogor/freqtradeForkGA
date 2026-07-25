"""Reference tests for V2 panel gates and immutable terminal artifacts."""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from genetic_algorithm.orchestration.artifact_store_v2 import (
    ArtifactIntegrityError,
    V2ArtifactStore,
    canonical_config_hash,
)
from genetic_algorithm.orchestration.promotion_policy_v2 import (
    ScenarioRequirementV2,
    ShadowGatePolicyV2,
    evaluate_candidate_shadow,
    make_shadow_promotion_decision,
    shadow_gate_policy_from_config,
)
from genetic_algorithm.orchestration.result_contract import (
    AttemptManifestV2,
    BacktestRecordV2,
)
from genetic_algorithm.orchestration.shadow_attempt_v2 import ShadowAttemptRecorderV2
from genetic_algorithm.tests.v2_metric_fixtures import expectancy_metrics


def _record(
    scenario_id: str,
    role: str,
    *,
    max_drawdown_ucb: float = 0.15,
    committed_expectancy_lcb: float = 0.001,
) -> BacktestRecordV2:
    trades = [
        {"pair": "BTC/USDT", "profit_ratio": 0.01 if index % 3 else -0.005} for index in range(60)
    ]
    return BacktestRecordV2(
        attempt_id="attempt-001",
        wave_id="wave-001",
        experiment_id="experiment-001",
        candidate_id="candidate-001",
        config_hash="config-hash-001",
        phenotype_hash="phenotype-hash-001",
        code_version="commit-001",
        data_manifest_hash="data-hash-001",
        fitness_policy_version="fitness-v2-test",
        seed=42,
        worker_count=1,
        fee_rate=0.001,
        slippage_rate=0.0005,
        spread_rate=0.0,
        funding_rate=0.0,
        equity_method="MARK_TO_MARKET",
        metrics={
            "scenario_id": scenario_id,
            "pair": "BTC/USDT",
            "timeframe": "1h",
            "role": role,
            "period_start": "2025-01-01",
            "period_end": "2025-06-30",
            "cost_multiplier": 1.0,
            "status": "VALID",
            "success": True,
            "no_trades": False,
            "net_return": 0.10,
            "annualized_net_return": 0.20,
            "annualized_net_return_lcb": 0.05,
            "max_drawdown": 0.10,
            "max_drawdown_ucb": max_drawdown_ucb,
            "daily_expected_shortfall_5": 0.01,
            "daily_expected_shortfall_5_ucb": 0.02,
            "net_expectancy": 0.003,
                "net_expectancy_lcb": 0.001,
                "profit_factor": 1.5,
                "profit_factor_censored": False,
                "profit_factor_contract_version": (
                    "right-censored-profit-factor-v1"
                ),
                "win_rate": 0.60,
            "trade_count": len(trades),
            "effective_sample_size": 40.0,
            **expectancy_metrics(
                trade_count=len(trades),
                mean_return=0.003,
                lower_confidence_bound=committed_expectancy_lcb,
                effective_sample_size=40.0,
            ),
            "active_months": 6,
            "max_consecutive_losses": 3,
            "max_drawdown_duration_days": 20.0,
        },
        daily_net_returns=[0.01, -0.005, 0.004],
        equity_curve=[100.0, 101.0, 100.495, 100.897],
        trades=trades,
    )


def _policy() -> ShadowGatePolicyV2:
    requirements = [
        ScenarioRequirementV2(
            scenario_id="inner-btc",
            pair="BTC/USDT",
            timeframe="1h",
            role="INNER_VALIDATION",
            period_start=date(2025, 1, 1),
            period_end=date(2025, 6, 30),
            cost_multiplier=1.0,
        ),
        ScenarioRequirementV2(
            scenario_id="final-btc",
            pair="BTC/USDT",
            timeframe="1h",
            role="FINAL_TEST",
            period_start=date(2025, 1, 1),
            period_end=date(2025, 6, 30),
            cost_multiplier=1.0,
        ),
    ]
    return ShadowGatePolicyV2(
        policy_version="promotion-v2-test",
        required_scenarios=requirements,
    )


def test_complete_panel_passes_every_non_compensable_shadow_gate():
    candidate = evaluate_candidate_shadow(
        [_record("inner-btc", "INNER_VALIDATION"), _record("final-btc", "FINAL_TEST")],
        _policy(),
    )

    assert candidate.status.value == "VALID"
    assert candidate.robust_score == pytest.approx(0.05 - 0.15 - 0.02)
    assert candidate.gates
    assert all(gate.status.value == "VALID" for gate in candidate.gates)
    assert all(gate.passed is True for gate in candidate.gates)

    created = datetime(2026, 7, 21, 12, 0, tzinfo=timezone.utc)
    first = make_shadow_promotion_decision(
        candidate,
        wave_id="wave-001",
        policy_version="promotion-v2-test",
        created_at=created,
    )
    second = make_shadow_promotion_decision(
        candidate,
        wave_id="wave-001",
        policy_version="promotion-v2-test",
        created_at=created,
    )
    assert first == second
    assert first.outcome == "WOULD_PASS"
    assert first.promotion_authorized is False


def test_promotion_policy_requires_explicit_enable_and_scenario_matrix():
    with pytest.raises(ValueError, match="explicitly enabled"):
        shadow_gate_policy_from_config({"promotion_v2": {"enabled": False}})

    payload = _policy().model_dump(mode="python")
    loaded = shadow_gate_policy_from_config(
        {"promotion_v2": {"enabled": True, **payload}}
    )
    assert loaded == _policy()

    missing_final = payload.copy()
    missing_final["required_scenarios"] = [payload["required_scenarios"][0]]
    with pytest.raises(ValueError, match="FINAL_TEST"):
        shadow_gate_policy_from_config(
            {"promotion_v2": {"enabled": True, **missing_final}}
        )


def test_failed_risk_gate_does_not_make_measurement_invalid_or_authorize_promotion():
    candidate = evaluate_candidate_shadow(
        [
            _record("inner-btc", "INNER_VALIDATION", max_drawdown_ucb=0.40),
            _record("final-btc", "FINAL_TEST"),
        ],
        _policy(),
    )
    failed = {gate.gate_id: gate for gate in candidate.gates if gate.passed is False}

    assert candidate.status.value == "VALID"
    assert failed["MAX_DRAWDOWN_UCB"].reason_code == "MAX_DRAWDOWN_UCB_TOO_HIGH"
    decision = make_shadow_promotion_decision(
        candidate,
        wave_id="wave-001",
        policy_version="promotion-v2-test",
        created_at=datetime(2026, 7, 21, tzinfo=timezone.utc),
    )
    assert decision.outcome == "FAIL"
    assert decision.promotion_authorized is False


def test_negative_committed_capital_edge_cannot_be_hidden_by_trade_expectancy():
    candidate = evaluate_candidate_shadow(
        [
            _record(
                "inner-btc",
                "INNER_VALIDATION",
                committed_expectancy_lcb=-0.002,
            ),
            _record("final-btc", "FINAL_TEST"),
        ],
        _policy(),
    )

    expectancy_gate = next(
        gate for gate in candidate.gates if gate.gate_id == "EXPECTANCY_LCB"
    )
    assert candidate.status.value == "VALID"
    assert expectancy_gate.passed is False
    assert expectancy_gate.observed == pytest.approx(-0.002)


def test_missing_required_scenario_is_inconclusive_instead_of_ignored():
    candidate = evaluate_candidate_shadow(
        [_record("inner-btc", "INNER_VALIDATION")],
        _policy(),
    )

    assert candidate.status.value == "INCONCLUSIVE"
    assert candidate.gates[0].reason_code == "INCOMPLETE_SCENARIO_MATRIX"
    assert "final-btc" in (candidate.gates[0].detail or "")


def test_candidate_aggregation_rejects_mixed_phenotypes():
    first = _record("inner-btc", "INNER_VALIDATION")
    second_payload = _record("final-btc", "FINAL_TEST").model_dump(mode="python")
    second_payload["phenotype_hash"] = "different-phenotype-hash"

    with pytest.raises(ValueError, match="frozen candidate"):
        evaluate_candidate_shadow(
            [first, BacktestRecordV2.model_validate(second_payload)], _policy()
        )


def _manifest(root: Path, config: dict) -> AttemptManifestV2:
    return AttemptManifestV2(
        attempt_id="attempt-001",
        wave_id="wave-001",
        parent_wave_id=None,
        experiment_id="experiment-001",
        created_at=datetime(2026, 7, 21, 12, 0, tzinfo=timezone.utc),
        config_hash=canonical_config_hash(config),
        code_version="commit-001",
        dirty_patch_hash=None,
        data_manifest_hash="data-hash-001",
        fitness_policy_version="fitness-v2-test",
        seeds=[42],
        worker_count=1,
        resolved_config_path=str(root / "resolved_config.yaml"),
        artifact_root=str(root),
    )


def _record_for_manifest(
    manifest: AttemptManifestV2,
    scenario_id: str,
    role: str,
) -> BacktestRecordV2:
    payload = _record(scenario_id, role).model_dump(mode="python")
    for field in (
        "attempt_id",
        "wave_id",
        "experiment_id",
        "config_hash",
        "code_version",
        "data_manifest_hash",
        "fitness_policy_version",
        "worker_count",
    ):
        payload[field] = getattr(manifest, field)
    payload["seed"] = manifest.seeds[0]
    return BacktestRecordV2.model_validate(payload)


def test_terminal_artifact_roundtrip_is_atomic_hashed_and_tamper_evident(tmp_path: Path):
    root = tmp_path / "attempt-001"
    config = {"config_schema_version": 2, "evaluation_v2": {"enabled": True}}
    manifest = _manifest(root, config)
    records = [
        _record_for_manifest(manifest, "inner-btc", "INNER_VALIDATION"),
        _record_for_manifest(manifest, "final-btc", "FINAL_TEST"),
    ]
    candidate = evaluate_candidate_shadow(records, _policy())
    decision = make_shadow_promotion_decision(
        candidate,
        wave_id="wave-001",
        policy_version="promotion-v2-test",
        created_at=datetime(2026, 7, 21, 12, 1, tzinfo=timezone.utc),
    )
    store = V2ArtifactStore(root)
    store.write_manifest(manifest, config)
    store.write_policy(_policy())
    for record in records:
        store.write_backtest(record)
    store.write_candidate(candidate)
    store.write_decision(decision)

    draft = {
        "attempt_id": manifest.attempt_id,
        "status": "SUCCEEDED",
        "started_at": manifest.created_at,
        "finished_at": datetime(2026, 7, 21, 12, 2, tzinfo=timezone.utc),
        "manifest": manifest.model_dump(mode="python"),
        "candidate_evaluations": [candidate.model_dump(mode="python")],
    }
    finalized = store.finalize(draft)
    verified = store.read_verified_result()

    assert verified == finalized
    assert "manifest.json" in verified.artifact_hashes
    assert "resolved_config.yaml" in verified.artifact_hashes
    assert "candidates/candidate-001/decision.json" in verified.artifact_hashes
    assert (root / "result.json.sha256").exists()

    candidate_path = root / "candidates/candidate-001/candidate.json"
    candidate_path.write_bytes(candidate_path.read_bytes() + b" ")
    with pytest.raises(ArtifactIntegrityError, match="artifact manifest mismatch"):
        store.read_verified_result()


def test_raw_trade_datetime_roundtrips_through_canonical_artifacts(tmp_path: Path):
    """Engine-native timestamps must not look like artifact corruption."""
    root = tmp_path / "attempt-runtime-types"
    config = {"config_schema_version": 2, "evaluation_v2": {"enabled": True}}
    manifest = _manifest(root, config).model_copy(
        update={
            "attempt_id": "attempt-runtime-types",
            "artifact_root": str(root),
            "resolved_config_path": str(root / "resolved_config.yaml"),
        }
    )
    payload = _record_for_manifest(
        manifest, "inner-btc", "INNER_VALIDATION"
    ).model_dump(mode="python")
    payload["trades"][0]["open_date"] = datetime(
        2026, 7, 1, 12, 0, tzinfo=timezone.utc
    )
    payload["trades"][0]["orders"] = ({"side": "buy"},)
    record = BacktestRecordV2.model_validate(payload)
    recorder = ShadowAttemptRecorderV2(manifest, config, _policy())

    recorder.add_backtest(record)
    recorder.add_backtest(
        _record_for_manifest(manifest, "final-btc", "FINAL_TEST")
    )
    result = recorder.finalize(
        finished_at=datetime(2026, 7, 21, 12, 2, tzinfo=timezone.utc)
    )

    assert result.status.value == "SUCCEEDED"
    persisted = V2ArtifactStore(root).read_verified_result()
    inner = next(
        scenario
        for scenario in persisted.candidate_evaluations[0].scenarios
        if scenario.metrics.scenario_id == "inner-btc"
    )
    trade = inner.trades[0]
    assert trade["open_date"] == "2026-07-01T12:00:00Z"
    assert trade["orders"] == [{"side": "buy"}]


def test_artifacts_are_immutable_and_partial_attempt_has_no_terminal_result(tmp_path: Path):
    root = tmp_path / "attempt-001"
    config = {"config_schema_version": 2}
    store = V2ArtifactStore(root)
    store.write_manifest(_manifest(root, config), config)
    manifest = _manifest(root, config)
    original = _record_for_manifest(manifest, "inner-btc", "INNER_VALIDATION")
    store.write_backtest(original)

    changed = original.model_dump(mode="python")
    changed["metrics"]["net_return"] = 0.2
    with pytest.raises(ArtifactIntegrityError, match="immutable artifact"):
        store.write_backtest(BacktestRecordV2.model_validate(changed))
    with pytest.raises(ArtifactIntegrityError, match="no complete terminal result"):
        store.read_verified_result()


def test_manifest_rejects_config_hash_mismatch(tmp_path: Path):
    root = tmp_path / "attempt-001"
    config = {"config_schema_version": 2}
    manifest_payload = _manifest(root, config).model_dump(mode="python")
    manifest_payload["config_hash"] = "wrong-hash-value"

    with pytest.raises(ArtifactIntegrityError, match="config hash"):
        V2ArtifactStore(root).write_manifest(
            AttemptManifestV2.model_validate(manifest_payload),
            config,
        )


def test_shadow_attempt_recorder_commits_real_records_and_is_idempotent(tmp_path: Path):
    root = tmp_path / "attempt-001"
    config = {"config_schema_version": 2, "evaluation_v2": {"enabled": True}}
    manifest = _manifest(root, config)
    recorder = ShadowAttemptRecorderV2(manifest, config, _policy())
    recorder.add_backtest(
        _record_for_manifest(manifest, "inner-btc", "INNER_VALIDATION")
    )
    recorder.add_backtest(_record_for_manifest(manifest, "final-btc", "FINAL_TEST"))
    finished = datetime(2026, 7, 21, 12, 2, tzinfo=timezone.utc)

    first = recorder.finalize(finished_at=finished)
    second = recorder.finalize(finished_at=finished)

    assert first == second
    assert first.status.value == "SUCCEEDED"
    assert first.candidate_evaluations[0].status.value == "VALID"
    assert V2ArtifactStore(root).read_verified_result() == first
    with pytest.raises(ArtifactIntegrityError, match="after terminal result"):
        recorder.add_backtest(
            _record_for_manifest(manifest, "inner-btc", "INNER_VALIDATION")
        )


def test_shadow_attempt_without_records_commits_invalid_terminal_result(tmp_path: Path):
    root = tmp_path / "attempt-empty"
    config = {"config_schema_version": 2}
    manifest = _manifest(root, config).model_copy(
        update={
            "attempt_id": "attempt-empty",
            "artifact_root": str(root),
            "resolved_config_path": str(root / "resolved_config.yaml"),
        }
    )
    recorder = ShadowAttemptRecorderV2(manifest, config, _policy())

    result = recorder.finalize(
        finished_at=datetime(2026, 7, 21, 12, 1, tzinfo=timezone.utc)
    )

    assert result.status.value == "INVALID_RESULT"
    assert result.error_code == "NO_SCENARIO_RECORDS"
    assert result.candidate_evaluations == []
