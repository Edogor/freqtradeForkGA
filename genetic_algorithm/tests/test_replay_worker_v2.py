"""Immutable worker-spec and real Executor-to-Replay-Worker process tests."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
import yaml

from freqtrade.misc import pair_to_filename
from genetic_algorithm.config.schema import load_config
from genetic_algorithm.orchestration.artifact_store_v2 import (
    V2ArtifactStore,
    canonical_config_hash,
)
from genetic_algorithm.orchestration.attempt_executor_v2 import (
    AttemptCommandV2,
    AttemptExecutorV2,
    ProcessProbeStatus,
    ProcessProbeV2,
    ReconciliationAction,
)
from genetic_algorithm.orchestration.attempt_state_v2 import (
    AttemptClaimConflict,
    AttemptLifecycleStatus,
    AttemptStateStoreV2,
    WorkerKind,
)
from genetic_algorithm.orchestration.manifest_builder_v2 import build_attempt_manifest_bundle
from genetic_algorithm.orchestration.promotion_policy_v2 import (
    ScenarioRequirementV2,
    ShadowGatePolicyV2,
)
from genetic_algorithm.orchestration.replay_runner_v2 import freeze_candidate
from genetic_algorithm.orchestration.replay_worker_v2 import (
    LoadedReplayWorkerV2,
    PreparedReplayWorkerV2,
    ReplayWorkerError,
    load_replay_worker,
    prepare_replay_worker,
    replay_worker_argv,
    run_replay_worker,
)
from genetic_algorithm.orchestration.result_contract import (
    AttemptManifestV2,
    AttemptStatus,
    EvaluationStatus,
)
from genetic_algorithm.orchestration.shadow_scheduler_v2 import (
    ShadowSchedulerConfigV2,
    ShadowSchedulerV2,
)


@dataclass(frozen=True)
class _WorkerContext:
    prepared: PreparedReplayWorkerV2
    config: dict
    repo_root: Path
    ledger_path: Path


class _MissingProcessInspector:
    def identity_token(self, pid: int) -> str | None:
        return f"missing:{pid}"

    def probe(self, pid: int, expected_start_token: str) -> ProcessProbeV2:
        return ProcessProbeV2(status=ProcessProbeStatus.NOT_FOUND)


def _rewrite_with_hash_consistent_invalid_config(
    prepared: PreparedReplayWorkerV2,
) -> str:
    root = Path(prepared.spec.artifact_root)
    config_path = root / V2ArtifactStore.CONFIG_NAME
    config = yaml.safe_load(config_path.read_text())
    config["genetic_algorithm"]["mutaton_rate"] = 0.2
    config_path.write_text(yaml.safe_dump(config, sort_keys=True))

    manifest_path = root / V2ArtifactStore.MANIFEST_NAME
    manifest = AttemptManifestV2.model_validate_json(manifest_path.read_bytes())
    manifest = manifest.model_copy(update={"config_hash": canonical_config_hash(config)})
    manifest_path.write_text(manifest.model_dump_json(indent=2))

    input_hashes = dict(prepared.spec.input_hashes)
    for relative in (V2ArtifactStore.CONFIG_NAME, V2ArtifactStore.MANIFEST_NAME):
        input_hashes[relative] = hashlib.sha256((root / relative).read_bytes()).hexdigest()
    spec = prepared.spec.model_copy(update={"input_hashes": input_hashes})
    prepared.spec_path.write_text(spec.model_dump_json(indent=2))
    return hashlib.sha256(prepared.spec_path.read_bytes()).hexdigest()


@pytest.fixture
def worker_context(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[2]
    suffix = hashlib.sha256(str(tmp_path).encode()).hexdigest()[:10].upper()
    pair = f"UNITTEST_WORKER_V2_{suffix}/USDT"
    data_root = repo_root / "tests" / "testdata"
    data_root.mkdir(parents=True, exist_ok=True)
    data_path = data_root / f"{pair_to_filename(pair)}-1h.feather"
    if data_path.exists():
        pytest.fail(f"worker test data path unexpectedly exists: {data_path}")
    candles = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=48, freq="1h", tz="UTC"),
            "open": [100.0] * 48,
            "high": [101.0] * 48,
            "low": [99.0] * 48,
            "close": [100.5] * 48,
            "volume": [10.0] * 48,
        }
    )
    candles.to_feather(data_path)
    try:
        policy = ShadowGatePolicyV2(
            policy_version="worker-policy-v2-test",
            required_scenarios=[
                ScenarioRequirementV2(
                    scenario_id="final-worker-cell",
                    pair=pair,
                    timeframe="1h",
                    role="FINAL_TEST",
                    period_start=date(2024, 1, 1),
                    period_end=date(2024, 1, 2),
                    cost_multiplier=1.0,
                )
            ],
            min_effective_sample_size=1,
            min_active_months=1,
        )
        config = load_config("genetic_algorithm/config/presets/safe_v2.yaml")
        config["backtesting"].update(
            {
                "exchange": "binance",
                "pairs": ["BTC/USDT"],
                "timerange": "20230101-20240101",
                "timeframe": "1h",
                "dataformat_ohlcv": "feather",
                "fee": 0.001,
                "slippage_pct": 0.0005,
                "fee_noise_std": 0.0,
                "dynamic_slippage": False,
                "spread_pct": 0.0,
                "funding_rate": 0.0,
            }
        )
        config["promotion_v2"] = {
            "enabled": True,
            **policy.model_dump(mode="json"),
        }
        created_at = datetime.now(UTC) - timedelta(seconds=10)
        artifact_root = tmp_path / "attempts" / "attempt-worker-v2"
        bundle = build_attempt_manifest_bundle(
            attempt_id="attempt-worker-v2",
            wave_id="wave-worker-v2",
            experiment_id="experiment-worker-v2",
            created_at=created_at,
            resolved_config=config,
            policy=policy,
            fitness_policy_version="fitness-worker-v2",
            seeds=[101],
            worker_count=1,
            artifact_root=artifact_root,
            repo_root=repo_root,
            data_root=data_root,
        )
        candidate = freeze_candidate(
            candidate_id="candidate-worker-v2",
            strategy_name="WorkerStrategyV2",
            strategy_code='class WorkerStrategyV2:\n    timeframe = "1h"\n',
            timeframe="1h",
            max_open_trades=1,
        )
        ledger_path = tmp_path / "ledgers" / "final-test.sqlite3"
        prepared = prepare_replay_worker(
            bundle=bundle,
            resolved_config=config,
            policy=policy,
            candidates=[candidate],
            repo_root=repo_root,
            final_test_ledger_path=ledger_path,
            created_at=created_at + timedelta(seconds=1),
        )
        yield _WorkerContext(
            prepared=prepared,
            config=config,
            repo_root=repo_root,
            ledger_path=ledger_path,
        )
    finally:
        data_path.unlink(missing_ok=True)


def _queue_worker_attempt(
    state_store: AttemptStateStoreV2,
    manifest: AttemptManifestV2,
) -> None:
    state_store.register(manifest)
    state_store.transition(
        manifest.attempt_id,
        AttemptLifecycleStatus.VALIDATED,
        occurred_at=manifest.created_at + timedelta(seconds=1),
        actor="validator",
        reason="WORKER_INPUTS_VALID",
    )
    state_store.transition(
        manifest.attempt_id,
        AttemptLifecycleStatus.QUEUED,
        occurred_at=manifest.created_at + timedelta(seconds=2),
        actor="controller",
        reason="QUEUED_FOR_V2_WORKER",
    )


def _scheduler_config(*, max_concurrent: int = 1) -> ShadowSchedulerConfigV2:
    return ShadowSchedulerConfigV2(
        max_concurrent=max_concurrent,
        lease_seconds=5,
        heartbeat_interval_seconds=0.1,
        poll_interval_seconds=0.02,
    )


def _queue_through_scheduler(
    scheduler: ShadowSchedulerV2,
    context: _WorkerContext,
    manifest: AttemptManifestV2,
) -> None:
    scheduler.queue_replay(
        manifest=manifest,
        prepared=context.prepared,
        bound_at=manifest.created_at + timedelta(seconds=1),
        validated_at=manifest.created_at + timedelta(seconds=2),
        queued_at=manifest.created_at + timedelta(seconds=3),
        working_directory=context.repo_root,
    )


def test_prepared_worker_roundtrips_all_hash_bound_inputs(worker_context: _WorkerContext):
    prepared = worker_context.prepared

    loaded = load_replay_worker(
        prepared.spec_path,
        expected_spec_sha256=prepared.spec_file_sha256,
    )

    assert isinstance(loaded, LoadedReplayWorkerV2)
    assert loaded.manifest.attempt_id == prepared.spec.attempt_id
    assert loaded.resolved_config == worker_context.config
    assert loaded.candidates[0].candidate_id == "candidate-worker-v2"
    assert loaded.spec.output_layout.artifact_root == loaded.spec.artifact_root
    assert not worker_context.ledger_path.exists()


def test_replay_worker_revalidates_hash_consistent_config(worker_context: _WorkerContext):
    spec_sha256 = _rewrite_with_hash_consistent_invalid_config(worker_context.prepared)

    with pytest.raises(ReplayWorkerError, match="violates the V2 contract") as exc_info:
        load_replay_worker(
            worker_context.prepared.spec_path,
            expected_spec_sha256=spec_sha256,
        )

    assert "genetic_algorithm.mutaton_rate" in str(exc_info.value.__cause__)


def test_tampered_worker_spec_commits_verified_invalid_result(worker_context: _WorkerContext):
    prepared = worker_context.prepared
    prepared.spec_path.write_bytes(prepared.spec_path.read_bytes() + b"\n")

    result = run_replay_worker(
        prepared.spec_path,
        expected_spec_sha256=prepared.spec_file_sha256,
    )

    assert result.status == AttemptStatus.INVALID_RESULT
    assert result.error_code == "WORKER_INPUT_INVALID"
    assert "SHA-256" in str(result.error_detail)
    assert V2ArtifactStore(prepared.spec.artifact_root).read_verified_result() == result


def test_executor_rejects_spec_changed_after_command_was_built(
    worker_context: _WorkerContext,
    tmp_path: Path,
):
    prepared = worker_context.prepared
    manifest = load_replay_worker(
        prepared.spec_path,
        expected_spec_sha256=prepared.spec_file_sha256,
    ).manifest
    command = AttemptCommandV2(
        argv=replay_worker_argv(prepared),
        working_directory=str(worker_context.repo_root),
    )
    prepared.spec_path.write_bytes(prepared.spec_path.read_bytes() + b"\n")
    state_store = AttemptStateStoreV2(tmp_path / "attempt-state.sqlite3")
    _queue_worker_attempt(state_store, manifest)
    executor = AttemptExecutorV2(
        state_store,
        worker_id="worker-tamper-test",
        lease_seconds=5,
        heartbeat_interval_seconds=0.1,
    )

    terminal = executor.claim_and_execute_next(lambda _: command)

    assert terminal is not None
    assert terminal.status == AttemptLifecycleStatus.INVALID_RESULT
    assert terminal.result_status == AttemptStatus.INVALID_RESULT
    assert terminal.result_sha256 is not None
    result = V2ArtifactStore(manifest.artifact_root).read_verified_result()
    assert result.error_code == "WORKER_INPUT_INVALID"


def test_executor_process_runs_worker_and_verifies_terminal_result(
    worker_context: _WorkerContext,
    tmp_path: Path,
):
    prepared = worker_context.prepared
    manifest = load_replay_worker(
        prepared.spec_path,
        expected_spec_sha256=prepared.spec_file_sha256,
    ).manifest
    state_store = AttemptStateStoreV2(tmp_path / "attempt-state.sqlite3")
    _queue_worker_attempt(state_store, manifest)
    command = AttemptCommandV2(
        argv=replay_worker_argv(prepared),
        working_directory=str(worker_context.repo_root),
    )
    executor = AttemptExecutorV2(
        state_store,
        worker_id="worker-process-e2e",
        lease_seconds=5,
        heartbeat_interval_seconds=0.1,
    )

    terminal = executor.claim_and_execute_next(lambda _: command)

    assert terminal is not None
    assert terminal.status == AttemptLifecycleStatus.SUCCEEDED
    assert terminal.result_status == AttemptStatus.SUCCEEDED
    assert terminal.result_sha256 is not None
    assert terminal.log_sha256 is not None
    result = V2ArtifactStore(manifest.artifact_root).read_verified_result()
    assert result.status == AttemptStatus.SUCCEEDED
    assert result.candidate_evaluations[0].status == EvaluationStatus.FAIL
    assert (Path(manifest.artifact_root) / V2ArtifactStore.FINAL_TEST_USAGE_NAME).is_file()
    strategy_path = Path(prepared.spec.output_layout.replay_strategy_dir) / "WorkerStrategyV2.py"
    assert strategy_path.is_file()
    assert "replay/generated_strategies/WorkerStrategyV2.py" in result.artifact_hashes


def test_shadow_scheduler_runs_bound_replay_until_idle(
    worker_context: _WorkerContext,
    tmp_path: Path,
):
    prepared = worker_context.prepared
    manifest = load_replay_worker(
        prepared.spec_path,
        expected_spec_sha256=prepared.spec_file_sha256,
    ).manifest
    state_store = AttemptStateStoreV2(tmp_path / "scheduler-state.sqlite3")
    with ShadowSchedulerV2(
        state_store,
        scheduler_id="shadow-scheduler-e2e",
        config=_scheduler_config(),
    ) as scheduler:
        _queue_through_scheduler(scheduler, worker_context, manifest)

        ticks = scheduler.run_until_idle(timeout_seconds=20)

    terminal = state_store.get(manifest.attempt_id)
    assert terminal.status == AttemptLifecycleStatus.SUCCEEDED
    assert terminal.worker_binding is not None
    assert terminal.worker_binding.worker_kind == WorkerKind.SHADOW_REPLAY
    assert terminal.command_hash == terminal.worker_binding.command_hash
    assert any(manifest.attempt_id in tick.launched_attempt_ids for tick in ticks)
    assert any(manifest.attempt_id in tick.completed_attempt_ids for tick in ticks)
    assert not [error for tick in ticks for error in tick.future_errors]


def test_shadow_scheduler_counts_external_active_lease_as_occupied_slot(
    worker_context: _WorkerContext,
    tmp_path: Path,
):
    prepared = worker_context.prepared
    manifest = load_replay_worker(
        prepared.spec_path,
        expected_spec_sha256=prepared.spec_file_sha256,
    ).manifest
    state_store = AttemptStateStoreV2(tmp_path / "scheduler-state.sqlite3")
    now = datetime.now(UTC)
    with ShadowSchedulerV2(
        state_store,
        scheduler_id="shadow-scheduler-slot-test",
        config=_scheduler_config(),
    ) as scheduler:
        _queue_through_scheduler(scheduler, worker_context, manifest)
        claimed = state_store.claim_next(
            worker_id="external-worker",
            claimed_at=now,
            lease_seconds=5,
            worker_kind=WorkerKind.SHADOW_REPLAY,
        )
        assert claimed is not None

        tick = scheduler.run_once(observed_at=now + timedelta(milliseconds=100))

    assert tick.active_count == 1
    assert tick.available_slots == 0
    assert tick.launched_attempt_ids == []


def test_queued_worker_binding_rejects_command_substitution(
    worker_context: _WorkerContext,
    tmp_path: Path,
):
    prepared = worker_context.prepared
    manifest = load_replay_worker(
        prepared.spec_path,
        expected_spec_sha256=prepared.spec_file_sha256,
    ).manifest
    state_store = AttemptStateStoreV2(tmp_path / "scheduler-state.sqlite3")
    with ShadowSchedulerV2(
        state_store,
        scheduler_id="shadow-scheduler-command-fence",
        config=_scheduler_config(),
    ) as scheduler:
        _queue_through_scheduler(scheduler, worker_context, manifest)
    claimed = state_store.claim_next(
        worker_id="substitution-worker",
        claimed_at=datetime.now(UTC),
        lease_seconds=5,
        worker_kind=WorkerKind.SHADOW_REPLAY,
    )
    assert claimed is not None
    binding = claimed.worker_binding
    assert binding is not None
    substituted = AttemptCommandV2(
        argv=[*binding.argv, "--unexpected-argument"],
        working_directory=binding.working_directory,
    )
    executor = AttemptExecutorV2(
        state_store,
        worker_id="substitution-worker",
        lease_seconds=5,
        heartbeat_interval_seconds=0.1,
    )

    with pytest.raises(AttemptClaimConflict, match="queued worker binding"):
        executor.execute_attempt(claimed, substituted)

    assert state_store.get(manifest.attempt_id).status == AttemptLifecycleStatus.CLAIMED


def test_shadow_scheduler_reconciles_dead_expired_process_before_claiming(
    worker_context: _WorkerContext,
    tmp_path: Path,
):
    prepared = worker_context.prepared
    manifest = load_replay_worker(
        prepared.spec_path,
        expected_spec_sha256=prepared.spec_file_sha256,
    ).manifest
    state_store = AttemptStateStoreV2(tmp_path / "scheduler-state.sqlite3")
    now = datetime.now(UTC)
    with ShadowSchedulerV2(
        state_store,
        scheduler_id="shadow-scheduler-recovery-test",
        config=_scheduler_config(),
        process_inspector=_MissingProcessInspector(),
    ) as scheduler:
        _queue_through_scheduler(scheduler, worker_context, manifest)
        claimed = state_store.claim_next(
            worker_id="dead-worker",
            claimed_at=now,
            lease_seconds=5,
            worker_kind=WorkerKind.SHADOW_REPLAY,
        )
        assert claimed is not None
        binding = claimed.worker_binding
        assert binding is not None
        command = AttemptCommandV2(
            argv=binding.argv,
            working_directory=binding.working_directory,
        )
        log_path = Path(manifest.artifact_root) / "runtime" / "attempt.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_bytes(b"dead process log\n")
        state_store.prepare_execution(
            manifest.attempt_id,
            claim_token=str(claimed.claim_token),
            command_hash=command.command_hash,
            working_directory=binding.working_directory,
            log_path=log_path,
            prepared_at=now + timedelta(milliseconds=100),
        )
        state_store.mark_running(
            manifest.attempt_id,
            claim_token=str(claimed.claim_token),
            pid=424242,
            process_start_token="missing:424242",
            started_at=now + timedelta(milliseconds=200),
            lease_seconds=1,
        )

        tick = scheduler.run_once(observed_at=now + timedelta(seconds=2))

    assert tick.reconciliation[0].action == ReconciliationAction.INTERRUPTED_PROCESS_GONE
    assert state_store.get(manifest.attempt_id).status == AttemptLifecycleStatus.INTERRUPTED
