"""Tests for fail-closed Wave/Experiment/Decision reconciliation."""

from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
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
)
from genetic_algorithm.orchestration.wave_state_v2 import (
    AttemptEvidenceType,
    AttemptExpectationV2,
    ExperimentArmType,
    ExperimentSpecV2,
    WaveBudgetV2,
    WaveDecisionType,
    WaveDecisionV2,
    WaveLifecycleStatus,
    WaveSpecV2,
    WaveStateError,
    WaveStateStoreV2,
)


NOW = datetime(2026, 7, 21, 19, 0, tzinfo=UTC)


def _manifest(
    tmp_path: Path,
    attempt_id: str,
    seed: int,
    *,
    config: dict[str, object] | None = None,
    policy_version: str = "wave-policy-v2-test",
) -> AttemptManifestV2:
    root = tmp_path / "attempts" / attempt_id
    resolved_config = config or {"experiment": "control-v2-test"}
    manifest = AttemptManifestV2(
        attempt_id=attempt_id,
        wave_id="wave-v2-test",
        experiment_id="control-v2-test",
        created_at=NOW + timedelta(seconds=1),
        config_hash=canonical_config_hash(resolved_config),
        code_version="wave-state-test-commit",
        data_manifest_hash="d" * 64,
        split_manifest_hash="s" * 64,
        fitness_policy_version=policy_version,
        seeds=[seed],
        worker_count=1,
        resolved_config_path=str(root / V2ArtifactStore.CONFIG_NAME),
        artifact_root=str(root),
    )
    V2ArtifactStore(root).write_manifest(manifest, resolved_config)
    return manifest


def _manifest_hash(manifest: AttemptManifestV2) -> str:
    return canonical_config_hash(manifest.model_dump(mode="json"))


def _setup_wave(
    tmp_path: Path,
    *,
    manifests: list[AttemptManifestV2] | None = None,
    expected_hash_override: dict[str, str] | None = None,
) -> tuple[WaveStateStoreV2, list[AttemptManifestV2]]:
    store = WaveStateStoreV2(tmp_path / "orchestration.sqlite3")
    selected = manifests or [
        _manifest(tmp_path, "attempt-a", 11),
        _manifest(tmp_path, "attempt-b", 22),
    ]
    store.register_wave(
        WaveSpecV2(
            wave_id="wave-v2-test",
            policy_version="wave-policy-v2-test",
            search_space_version="search-space-v2-test",
            budget=WaveBudgetV2(
                max_attempts=4,
                max_parallel=2,
                max_wallclock_seconds=3600,
                max_retries=1,
            ),
            created_at=NOW,
        )
    )
    for manifest in selected:
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
    overrides = expected_hash_override or {}
    store.register_experiment(
        ExperimentSpecV2(
            experiment_id="control-v2-test",
            wave_id="wave-v2-test",
            arm_type=ExperimentArmType.CONTROL,
            hypothesis="The unchanged baseline anchors the next-wave comparison.",
            primary_metric="annualized_net_return_lcb",
            seeds=sorted(manifest.seeds[0] for manifest in selected),
            resolved_config_hash=canonical_config_hash(
                {"experiment": "control-v2-test"}
            ),
            expected_attempts=[
                AttemptExpectationV2(
                    attempt_id=manifest.attempt_id,
                    manifest_hash=overrides.get(manifest.attempt_id, _manifest_hash(manifest)),
                    seed=manifest.seeds[0],
                    ordinal=index,
                )
                for index, manifest in enumerate(selected)
            ],
            created_at=NOW + timedelta(seconds=1),
        )
    )
    return store, selected


def _begin(store: WaveStateStoreV2) -> None:
    state = store.begin_collection(
        "wave-v2-test",
        started_at=NOW + timedelta(seconds=4),
    )
    assert state.status == WaveLifecycleStatus.COLLECTING


def _claim(store: WaveStateStoreV2, at_second: int) -> tuple[str, str]:
    claimed = store.claim_next(
        worker_id="wave-test-worker",
        claimed_at=NOW + timedelta(seconds=at_second),
        lease_seconds=60,
    )
    assert claimed is not None
    assert claimed.claim_token is not None
    return claimed.attempt_id, claimed.claim_token


def _finish_with_verified_failure(
    store: WaveStateStoreV2,
    manifest: AttemptManifestV2,
    *,
    at_second: int,
) -> None:
    attempt_id, claim_token = _claim(store, at_second)
    assert attempt_id == manifest.attempt_id
    log_path = Path(manifest.artifact_root) / "runtime" / "attempt.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("verified failure\n")
    store.prepare_execution(
        attempt_id,
        claim_token=claim_token,
        command_hash="e" * 64,
        working_directory=Path(manifest.artifact_root).parent,
        log_path=log_path,
        prepared_at=NOW + timedelta(seconds=at_second, milliseconds=100),
    )
    process_token = f"process-{attempt_id}"
    store.mark_running(
        attempt_id,
        claim_token=claim_token,
        pid=1234,
        process_start_token=process_token,
        started_at=NOW + timedelta(seconds=at_second, milliseconds=200),
        lease_seconds=60,
    )
    result = V2ArtifactStore(manifest.artifact_root).finalize(
        AttemptResultV2(
            attempt_id=attempt_id,
            status=AttemptStatus.FAILED,
            started_at=NOW + timedelta(seconds=at_second, milliseconds=200),
            finished_at=NOW + timedelta(seconds=at_second, milliseconds=300),
            manifest=manifest,
            error_code="BACKTEST_FAILED",
            error_detail="synthetic verified failure",
        )
    )
    assert result.status == AttemptStatus.FAILED
    store.finalize_from_artifacts(
        attempt_id,
        claim_token=claim_token,
        process_start_token=process_token,
        finalized_at=NOW + timedelta(seconds=at_second, milliseconds=400),
    )


def _finish_without_result(
    store: WaveStateStoreV2,
    manifest: AttemptManifestV2,
    *,
    at_second: int,
) -> WaveDecisionV2:
    attempt_id, claim_token = _claim(store, at_second)
    assert attempt_id == manifest.attempt_id
    store.mark_runtime_failure(
        attempt_id,
        claim_token=claim_token,
        process_start_token=None,
        status=AttemptLifecycleStatus.FAILED,
        failed_at=NOW + timedelta(seconds=at_second, milliseconds=100),
        error_code="SPAWN_FAILED",
        error_detail="synthetic missing child",
    )
    decision = WaveDecisionV2(
        decision_id=f"abort-{attempt_id}",
        wave_id="wave-v2-test",
        decision_type=WaveDecisionType.ATTEMPT_ABORT,
        attempt_id=attempt_id,
        created_at=NOW + timedelta(seconds=at_second, milliseconds=200),
        actor="operator",
        reason_codes=["SPAWN_FAILED_CONFIRMED"],
        payload={"reviewed_attempt_error": "SPAWN_FAILED"},
    )
    store.apply_decision(decision)
    return decision


def _complete_mixed_wave(
    store: WaveStateStoreV2,
    manifests: list[AttemptManifestV2],
):
    _begin(store)
    _finish_with_verified_failure(store, manifests[0], at_second=5)
    abort = _finish_without_result(store, manifests[1], at_second=6)
    reconciled = store.reconcile_wave(
        "wave-v2-test",
        reconciled_at=NOW + timedelta(seconds=7),
    )
    return reconciled, abort


def test_reconciliation_freezes_verified_result_and_documented_abort(tmp_path: Path):
    store, manifests = _setup_wave(tmp_path)

    reconciled, abort = _complete_mixed_wave(store, manifests)

    assert reconciled.status == WaveLifecycleStatus.RECONCILED
    assert reconciled.result_snapshot is not None
    assert reconciled.result_snapshot_hash == reconciled.result_snapshot.snapshot_hash
    evidence = reconciled.result_snapshot.attempt_results
    assert [item.attempt_id for item in evidence] == ["attempt-a", "attempt-b"]
    assert evidence[0].evidence == AttemptEvidenceType.VERIFIED_RESULT
    assert evidence[0].result_status == AttemptStatus.FAILED
    assert evidence[1].evidence == AttemptEvidenceType.DOCUMENTED_ABORT
    assert evidence[1].abort_decision_hash == abort.decision_hash
    assert store.reconcile_wave(
        "wave-v2-test", reconciled_at=NOW + timedelta(seconds=8)
    ) == reconciled


def test_missing_abort_blocks_reconciliation_until_documented(tmp_path: Path):
    store, manifests = _setup_wave(tmp_path)
    _begin(store)
    _finish_with_verified_failure(store, manifests[0], at_second=5)
    attempt_id, claim_token = _claim(store, 6)
    store.mark_runtime_failure(
        attempt_id,
        claim_token=claim_token,
        process_start_token=None,
        status=AttemptLifecycleStatus.FAILED,
        failed_at=NOW + timedelta(seconds=6, milliseconds=100),
        error_code="SPAWN_FAILED",
    )

    with pytest.raises(WaveStateError, match="lacks verified result or abort"):
        store.reconcile_wave(
            "wave-v2-test", reconciled_at=NOW + timedelta(seconds=7)
        )
    assert store.get_wave("wave-v2-test").status == WaveLifecycleStatus.COLLECTING

    store.apply_decision(
        WaveDecisionV2(
            decision_id="late-abort",
            wave_id="wave-v2-test",
            decision_type=WaveDecisionType.ATTEMPT_ABORT,
            attempt_id=manifests[1].attempt_id,
            created_at=NOW + timedelta(seconds=8),
            actor="operator",
            reason_codes=["MANUALLY_REVIEWED_FAILURE"],
        )
    )
    assert store.reconcile_wave(
        "wave-v2-test", reconciled_at=NOW + timedelta(seconds=9)
    ).status == WaveLifecycleStatus.RECONCILED


def test_collection_rejects_unexpected_attempt_in_same_wave(tmp_path: Path):
    store, _ = _setup_wave(tmp_path)
    extra = _manifest(tmp_path, "attempt-extra", 33)
    store.register(extra)
    store.transition(
        extra.attempt_id,
        AttemptLifecycleStatus.VALIDATED,
        occurred_at=NOW + timedelta(seconds=2),
        actor="validator",
        reason="VALIDATED",
    )
    store.transition(
        extra.attempt_id,
        AttemptLifecycleStatus.QUEUED,
        occurred_at=NOW + timedelta(seconds=3),
        actor="controller",
        reason="QUEUED",
    )

    with pytest.raises(WaveStateError, match=r"unexpected=\['attempt-extra'\]"):
        _begin(store)


def test_collection_rejects_manifest_hash_drift(tmp_path: Path):
    first = _manifest(tmp_path, "attempt-a", 11)
    second = _manifest(tmp_path, "attempt-b", 22)
    store, _ = _setup_wave(
        tmp_path,
        manifests=[first, second],
        expected_hash_override={"attempt-b": "f" * 64},
    )

    with pytest.raises(WaveStateError, match="differs from experiment expectation"):
        _begin(store)


@pytest.mark.parametrize(
    ("manifest", "message"),
    [
        (
            lambda path: _manifest(
                path,
                "attempt-a",
                11,
                config={"experiment": "wrong-config"},
            ),
            "config hash differs from experiment spec",
        ),
        (
            lambda path: _manifest(
                path,
                "attempt-a",
                11,
                policy_version="wrong-policy",
            ),
            "policy version differs from wave spec",
        ),
    ],
)
def test_collection_binds_config_and_policy_to_attempt_manifest(
    tmp_path: Path,
    manifest,
    message: str,
):
    first = manifest(tmp_path)
    second = _manifest(tmp_path, "attempt-b", 22)
    store, _ = _setup_wave(tmp_path, manifests=[first, second])

    with pytest.raises(WaveStateError, match=message):
        _begin(store)


def test_corrupt_result_cannot_be_hidden_by_abort_decision(tmp_path: Path):
    store, manifests = _setup_wave(tmp_path)
    _begin(store)
    _finish_with_verified_failure(store, manifests[0], at_second=5)
    _finish_without_result(store, manifests[1], at_second=6)
    result_path = Path(manifests[0].artifact_root) / V2ArtifactStore.RESULT_NAME
    result_path.write_bytes(result_path.read_bytes() + b"corrupt")

    with pytest.raises(WaveStateError, match="result evidence cannot be documented as abort"):
        store.apply_decision(
            WaveDecisionV2(
                decision_id="unsafe-abort",
                wave_id="wave-v2-test",
                decision_type=WaveDecisionType.ATTEMPT_ABORT,
                attempt_id=manifests[0].attempt_id,
                created_at=NOW + timedelta(seconds=7),
                actor="operator",
                reason_codes=["TRY_TO_HIDE_CORRUPTION"],
            )
        )
    with pytest.raises(ArtifactIntegrityError, match="hash differs from state"):
        store.reconcile_wave(
            "wave-v2-test", reconciled_at=NOW + timedelta(seconds=8)
        )
    assert store.get_wave("wave-v2-test").status == WaveLifecycleStatus.COLLECTING


def test_decisions_gate_analysis_proposal_approval_and_queueing(tmp_path: Path):
    store, manifests = _setup_wave(tmp_path)
    reconciled, _ = _complete_mixed_wave(store, manifests)
    assert reconciled.result_snapshot_hash is not None
    analysis = WaveDecisionV2(
        decision_id="analysis-1",
        wave_id="wave-v2-test",
        decision_type=WaveDecisionType.ANALYSIS,
        created_at=NOW + timedelta(seconds=8),
        actor="analyzer-v2",
        reason_codes=["PANEL_ANALYSIS_COMPLETE"],
        input_hash=reconciled.result_snapshot_hash,
        payload={"eligible_candidates": 0},
    )
    assert store.apply_decision(analysis).status == WaveLifecycleStatus.ANALYZED
    proposal = WaveDecisionV2(
        decision_id="proposal-1",
        wave_id="wave-v2-test",
        decision_type=WaveDecisionType.PROPOSAL,
        created_at=NOW + timedelta(seconds=9),
        actor="planner-v2",
        reason_codes=["CONTROL_ONLY_CANARY"],
        input_hash=analysis.decision_hash,
        payload={
            "child_wave_id": "wave-v2-child",
            "planning_allowed": True,
            "plan_hash": "p" * 64,
            "materialization_hash": "m" * 64,
        },
    )
    assert store.apply_decision(proposal).status == WaveLifecycleStatus.PROPOSED

    wrong = WaveDecisionV2(
        decision_id="approval-wrong-input",
        wave_id="wave-v2-test",
        decision_type=WaveDecisionType.APPROVAL,
        created_at=NOW + timedelta(seconds=10),
        actor="operator",
        reason_codes=["MANUAL_APPROVAL"],
        input_hash="0" * 64,
    )
    with pytest.raises(WaveStateError, match="input_hash differs"):
        store.apply_decision(wrong)

    approval = wrong.model_copy(
        update={
            "decision_id": "approval-1",
            "input_hash": proposal.decision_hash,
            "payload": {
                "approved_plan_hash": "p" * 64,
                "approved_materialization_hash": "m" * 64,
            },
        }
    )
    assert store.apply_decision(approval).status == WaveLifecycleStatus.APPROVED
    with pytest.raises(WaveStateError, match="direct queue marking is disabled"):
        store.mark_queued("wave-v2-test", queued_at=NOW + timedelta(seconds=11))


def test_decision_rows_and_experiment_specs_are_database_immutable(tmp_path: Path):
    store, manifests = _setup_wave(tmp_path)
    _, abort = _complete_mixed_wave(store, manifests)
    connection = sqlite3.connect(store.path)
    try:
        with pytest.raises(sqlite3.IntegrityError, match="decisions are immutable"):
            connection.execute(
                "UPDATE wave_decisions SET input_hash = ? WHERE decision_id = ?",
                ("0" * 64, abort.decision_id),
            )
        with pytest.raises(sqlite3.IntegrityError, match="specs are immutable"):
            connection.execute(
                "UPDATE experiment_specs SET arm_type = 'EXPLORE' WHERE experiment_id = ?",
                ("control-v2-test",),
            )
        with pytest.raises(sqlite3.IntegrityError, match="wave specs are immutable"):
            connection.execute(
                "UPDATE waves SET policy_version = 'tampered' WHERE wave_id = ?",
                ("wave-v2-test",),
            )
    finally:
        connection.close()


def test_blocked_proposal_cannot_be_approved(tmp_path: Path):
    store, manifests = _setup_wave(tmp_path)
    reconciled, _ = _complete_mixed_wave(store, manifests)
    analysis = WaveDecisionV2(
        decision_id="analysis-blocked-proposal",
        wave_id="wave-v2-test",
        decision_type=WaveDecisionType.ANALYSIS,
        created_at=NOW + timedelta(seconds=8),
        actor="analyzer-v2",
        reason_codes=["ANALYSIS_COMPLETE"],
        input_hash=reconciled.result_snapshot_hash,
    )
    store.apply_decision(analysis)
    proposal = WaveDecisionV2(
        decision_id="proposal-blocked",
        wave_id="wave-v2-test",
        decision_type=WaveDecisionType.PROPOSAL,
        created_at=NOW + timedelta(seconds=9),
        actor="planner-v2",
        reason_codes=["SELECTION_BLOCKED"],
        input_hash=analysis.decision_hash,
        payload={"planning_allowed": False},
    )
    store.apply_decision(proposal)
    approval = WaveDecisionV2(
        decision_id="unsafe-approval",
        wave_id="wave-v2-test",
        decision_type=WaveDecisionType.APPROVAL,
        created_at=NOW + timedelta(seconds=10),
        actor="operator",
        reason_codes=["MANUAL_APPROVAL"],
        input_hash=proposal.decision_hash,
    )

    with pytest.raises(WaveStateError, match="cannot be approved"):
        store.apply_decision(approval)


def test_concurrent_reconciliation_is_idempotent(tmp_path: Path):
    store, manifests = _setup_wave(tmp_path)
    _begin(store)
    _finish_with_verified_failure(store, manifests[0], at_second=5)
    _finish_without_result(store, manifests[1], at_second=6)

    def reconcile():
        local = WaveStateStoreV2(store.path)
        return local.reconcile_wave(
            "wave-v2-test", reconciled_at=NOW + timedelta(seconds=7)
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        states = list(pool.map(lambda _: reconcile(), range(2)))

    assert states[0] == states[1]
    assert states[0].status == WaveLifecycleStatus.RECONCILED
    assert [event.reason for event in store.wave_events("wave-v2-test")].count(
        "COMPLETE_EVIDENCE_RECONCILED"
    ) == 1
