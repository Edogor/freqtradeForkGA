"""End-to-end tests for the frozen-candidate shadow replay executor."""

from __future__ import annotations

import copy
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from genetic_algorithm.evaluation.direct_backtester import BacktestResult
from genetic_algorithm.evaluation.equity_metrics_v2 import (
    build_realized_close_equity,
    calculate_equity_risk_metrics,
)
from genetic_algorithm.orchestration.artifact_store_v2 import (
    ArtifactIntegrityError,
    V2ArtifactStore,
    canonical_config_hash,
)
from genetic_algorithm.orchestration.attempt_state_v2 import (
    AttemptLifecycleStatus,
    AttemptStateStoreV2,
)
from genetic_algorithm.orchestration.code_manifest_v2 import CodeManifestV2
from genetic_algorithm.orchestration.data_manifest_v2 import (
    DataManifestV2,
    OHLCVFileV2,
    ScenarioDataCoverageV2,
)
from genetic_algorithm.orchestration.final_test_ledger_v2 import (
    FinalTestReuseError,
    FinalTestUsageLedgerV2,
)
from genetic_algorithm.orchestration.promotion_policy_v2 import (
    ScenarioRequirementV2,
    ShadowGatePolicyV2,
    shadow_gate_policy_from_config,
)
from genetic_algorithm.orchestration.replay_runner_v2 import (
    FrozenCandidateV2,
    ShadowReplayRunnerV2,
    freeze_candidate,
)
from genetic_algorithm.orchestration.result_contract import AttemptManifestV2
from genetic_algorithm.orchestration.split_contract_v2 import build_evaluation_split_plan
from genetic_algorithm.tests.v2_metric_fixtures import trade_evidence


def _policy() -> ShadowGatePolicyV2:
    return ShadowGatePolicyV2(
        policy_version="promotion-v2-replay-test",
        required_scenarios=[
            ScenarioRequirementV2(
                scenario_id="inner-1x",
                pair="BTC/USDT",
                timeframe="1h",
                role="TEMPORAL_VALIDATION",
                period_start=date(2025, 1, 1),
                period_end=date(2025, 6, 30),
                cost_multiplier=1.0,
            ),
            ScenarioRequirementV2(
                scenario_id="final-2x",
                pair="ETH/USDT",
                timeframe="1h",
                role="FINAL_TEST",
                period_start=date(2025, 1, 1),
                period_end=date(2025, 6, 30),
                cost_multiplier=2.0,
            ),
        ],
    )


def _config(policy: ShadowGatePolicyV2) -> dict:
    return {
        "config_schema_version": 2,
        "safety_profile": {
            "name": "safe_v2",
            "shadow_mode": True,
            "automation_eligible": False,
        },
        "evaluation_v2": {
            "enabled": True,
            "mark_to_market": True,
            "bootstrap_samples": 200,
            "bootstrap_block_days": 10,
            "trade_bootstrap_block": 5,
            "bootstrap_confidence": 0.95,
        },
        "promotion_v2": {
            "enabled": True,
            **policy.model_dump(mode="python"),
        },
        "backtesting": {
            "pairs": ["BTC/USDT", "ETH/USDT"],
            "timerange": "20240101-20250101",
            "fee": 0.001,
            "slippage_pct": 0.0005,
            "fee_noise_std": 0.0,
        },
        "short_selling": {"enabled": False},
    }


def _data_manifest(policy: ShadowGatePolicyV2) -> DataManifestV2:
    files = []
    coverage = []
    for index, requirement in enumerate(policy.required_scenarios):
        start = datetime.combine(requirement.period_start, datetime.min.time(), tzinfo=UTC)
        end = datetime.combine(requirement.period_end, datetime.min.time(), tzinfo=UTC)
        candle_count = ((requirement.period_end - requirement.period_start).days + 1) * 24
        digest = f"{index + 1:064x}"
        files.append(
            OHLCVFileV2(
                pair=requirement.pair,
                timeframe=requirement.timeframe,
                relative_path=f"pair-{index}.feather",
                sha256=digest,
                size_bytes=100,
                candle_count=candle_count,
                first_candle=start,
                last_candle=end + timedelta(hours=23),
            )
        )
        coverage.append(
            ScenarioDataCoverageV2(
                scenario_id=requirement.scenario_id,
                pair=requirement.pair,
                timeframe=requirement.timeframe,
                period_start=requirement.period_start,
                period_end=requirement.period_end,
                expected_candles=candle_count,
                actual_candles=candle_count,
                first_candle=start,
                last_candle=end + timedelta(hours=23),
                segment_sha256=digest,
            )
        )
    return DataManifestV2(
        exchange="binance",
        files=files,
        scenario_coverage=coverage,
    )


def _code_manifest() -> CodeManifestV2:
    tracked_hash = "a" * 64
    dirty_hash = canonical_config_hash(
        {
            "tracked_state_sha256": tracked_hash,
            "untracked_source_files": [],
        }
    )
    return CodeManifestV2(
        code_version="commit-replay-001",
        tracked_state_sha256=tracked_hash,
        dirty=True,
        dirty_patch_hash=dirty_hash,
    )


def _manifest(
    root: Path,
    config: dict,
    data_manifest: DataManifestV2,
    code_manifest: CodeManifestV2,
) -> AttemptManifestV2:
    split_manifest = build_evaluation_split_plan(
        config,
        shadow_gate_policy_from_config(config),
    )
    return AttemptManifestV2(
        attempt_id="replay-attempt-001",
        wave_id="wave-001",
        experiment_id="experiment-001",
        created_at=datetime(2026, 7, 21, 13, 0, tzinfo=UTC),
        config_hash=canonical_config_hash(config),
        code_version=code_manifest.code_version,
        dirty_patch_hash=code_manifest.dirty_patch_hash,
        data_manifest_hash=data_manifest.manifest_hash,
        split_manifest_hash=split_manifest.split_hash,
        fitness_policy_version="fitness-v2-replay-test",
        seeds=[42],
        worker_count=1,
        resolved_config_path=str(root / "resolved_config.yaml"),
        artifact_root=str(root),
    )


def _successful_result(*, end: date = date(2025, 6, 30)) -> BacktestResult:
    start = date(2025, 1, 1)
    days = (end - start).days + 1
    daily_pnl = [
        (start + timedelta(days=index), -0.15 if index % 13 == 0 else 0.12) for index in range(days)
    ]
    series = build_realized_close_equity(
        period_start=start,
        period_end=end,
        starting_balance=100.0,
        daily_profit_abs=daily_pnl,
    )
    risk = calculate_equity_risk_metrics(series)
    trade_returns = [0.010, 0.008, -0.004, 0.006, 0.005, -0.003, 0.007] * 5
    total_profit = series.final_balance - series.starting_balance
    committed_capital = total_profit / (sum(trade_returns) * 1.001)
    trades = trade_evidence(
        trade_returns,
        start=start,
        committed_capital=committed_capital,
    )
    return BacktestResult(
        success=True,
        strategy_name="FrozenStrategy",
        total_profit=total_profit,
        profit_percent=(series.final_balance / series.starting_balance - 1.0) * 100,
        total_trades=len(trades),
        wins=sum(value > 0 for value in trade_returns),
        losses=sum(value < 0 for value in trade_returns),
        win_rate=sum(value > 0 for value in trade_returns) / len(trade_returns),
        profit_factor=2.0,
        avg_profit=sum(trade_returns) / len(trade_returns),
        trades=trades,
        starting_balance=series.starting_balance,
        final_balance=series.final_balance,
        backtest_start=str(start),
        backtest_end=str(end),
        daily_profit_abs=[
            [day.isoformat(), pnl] for day, pnl in zip(series.dates, series.daily_profit_abs)
        ],
        daily_net_returns=list(series.daily_net_returns),
        equity_curve=list(series.equity_curve),
        equity_method="MARK_TO_MARKET",
        annualized_net_return=risk.annualized_net_return,
        daily_sharpe_ratio=risk.sharpe_ratio,
        daily_sortino_ratio=risk.sortino_ratio,
        daily_expected_shortfall_5=risk.daily_expected_shortfall_5,
        calmar_ratio=risk.calmar_ratio,
        ulcer_index=risk.ulcer_index,
        time_under_water_ratio=risk.time_under_water_ratio,
        daily_max_drawdown=risk.max_drawdown,
        max_consecutive_losses=1,
        max_drawdown_duration_days=risk.max_drawdown_duration_days,
        trade_profit_ratios=trade_returns,
    )


class _FakeBacktester:
    def __init__(self, config: dict, calls: list, result: BacktestResult):
        self.config = config
        self.calls = calls
        self.result = result

    def backtest_strategy(self, code, name, **kwargs):
        self.calls.append({"config": self.config, "code": code, "name": name, **kwargs})
        return copy.deepcopy(self.result)


def _candidate() -> FrozenCandidateV2:
    return freeze_candidate(
        candidate_id="candidate-001",
        strategy_name="FrozenStrategy",
        strategy_code="class FrozenStrategy:\n    timeframe = '1h'\n",
        timeframe="1h",
        max_open_trades=3,
        execution_parameters={"trading_mode": "spot", "position_adjustment": False},
    )


def test_replay_executes_exact_panel_costs_and_commits_verified_result(tmp_path: Path):
    policy = _policy()
    config = _config(policy)
    data_manifest = _data_manifest(policy)
    code_manifest = _code_manifest()
    root = tmp_path / "replay-attempt-001"
    ledger = FinalTestUsageLedgerV2(tmp_path / "final_test.sqlite3")
    manifest = _manifest(root, config, data_manifest, code_manifest)
    calls = []
    factory = lambda scenario_config: _FakeBacktester(  # noqa: E731
        scenario_config, calls, _successful_result()
    )
    runner = ShadowReplayRunnerV2(
        manifest=manifest,
        resolved_config=config,
        policy=policy,
        data_manifest=data_manifest,
        code_manifest=code_manifest,
        final_test_ledger=ledger,
        repo_root=str(tmp_path),
        backtester_factory=factory,
        data_manifest_builder=lambda config, policy: data_manifest,
        code_manifest_builder=lambda root: code_manifest,
        clock=lambda: datetime(2026, 7, 21, 13, 1, tzinfo=UTC),
    )

    result = runner.execute(
        [_candidate()],
        finished_at=datetime(2026, 7, 21, 13, 5, tzinfo=UTC),
    )

    assert result.status.value == "SUCCEEDED"
    assert result.candidate_evaluations[0].status.value == "VALID"
    assert len(calls) == 2
    assert {call["timerange_override"] for call in calls} == {"20250101-20250701"}
    assert {tuple(call["pairs_override"]) for call in calls} == {
        ("BTC/USDT",),
        ("ETH/USDT",),
    }
    cost_configs = {
        call["config"]["backtesting"]["fee"]: call["config"]["backtesting"]["slippage_pct"]
        for call in calls
    }
    assert cost_configs == {0.001: 0.0005, 0.002: 0.001}
    persisted_candidate = FrozenCandidateV2.model_validate_json(
        (root / "candidates" / "candidate-001" / "frozen_candidate.json").read_bytes()
    )
    assert persisted_candidate == _candidate()
    assert (root / "data_manifest.json").is_file()
    assert (root / "code_manifest.json").is_file()
    assert (root / "split_manifest.json").is_file()
    assert ledger.get(manifest.attempt_id).status == "EXPOSED"
    assert (root / "final_test_usage.json").is_file()
    assert V2ArtifactStore(root).read_verified_result() == result

    state_store = AttemptStateStoreV2(tmp_path / "attempt_state.sqlite3")
    state_store.register(manifest)
    state_store.transition(
        manifest.attempt_id,
        AttemptLifecycleStatus.VALIDATED,
        occurred_at=datetime(2026, 7, 21, 13, 0, 10, tzinfo=UTC),
        actor="validator",
        reason="V2_INPUTS_VALID",
    )
    state_store.transition(
        manifest.attempt_id,
        AttemptLifecycleStatus.QUEUED,
        occurred_at=datetime(2026, 7, 21, 13, 0, 20, tzinfo=UTC),
        actor="controller",
        reason="QUEUED",
    )
    claimed = state_store.claim_next(
        worker_id="worker-replay-test",
        claimed_at=datetime(2026, 7, 21, 13, 0, 30, tzinfo=UTC),
        lease_seconds=600,
    )
    assert claimed is not None
    runtime_log = root / "runtime" / "attempt.log"
    runtime_log.parent.mkdir(parents=True, exist_ok=True)
    runtime_log.write_bytes(b"replay state integration test\n")
    state_store.prepare_execution(
        manifest.attempt_id,
        claim_token=str(claimed.claim_token),
        command_hash="e" * 64,
        working_directory=tmp_path,
        log_path=runtime_log,
        prepared_at=datetime(2026, 7, 21, 13, 0, 35, tzinfo=UTC),
    )
    state_store.mark_running(
        manifest.attempt_id,
        claim_token=str(claimed.claim_token),
        pid=4321,
        process_start_token="proc-start-replay-test",
        started_at=datetime(2026, 7, 21, 13, 0, 40, tzinfo=UTC),
        lease_seconds=600,
    )
    terminal = state_store.finalize_from_artifacts(
        manifest.attempt_id,
        claim_token=str(claimed.claim_token),
        process_start_token="proc-start-replay-test",
        finalized_at=datetime(2026, 7, 21, 13, 5, 10, tzinfo=UTC),
    )
    assert terminal.status == AttemptLifecycleStatus.SUCCEEDED
    assert terminal.result_sha256 is not None


def test_period_mismatch_is_persisted_invalid_and_blocks_candidate(tmp_path: Path):
    policy = _policy()
    config = _config(policy)
    data_manifest = _data_manifest(policy)
    code_manifest = _code_manifest()
    root = tmp_path / "period-mismatch"
    ledger = FinalTestUsageLedgerV2(tmp_path / "final_test.sqlite3")
    manifest = _manifest(root, config, data_manifest, code_manifest).model_copy(
        update={
            "attempt_id": "period-mismatch",
            "artifact_root": str(root),
            "resolved_config_path": str(root / "resolved_config.yaml"),
        }
    )
    factory = lambda scenario_config: _FakeBacktester(  # noqa: E731
        scenario_config, [], _successful_result(end=date(2025, 6, 29))
    )
    runner = ShadowReplayRunnerV2(
        manifest=manifest,
        resolved_config=config,
        policy=policy,
        data_manifest=data_manifest,
        code_manifest=code_manifest,
        final_test_ledger=ledger,
        repo_root=str(tmp_path),
        backtester_factory=factory,
        data_manifest_builder=lambda config, policy: data_manifest,
        code_manifest_builder=lambda root: code_manifest,
        clock=lambda: datetime(2026, 7, 21, 13, 1, tzinfo=UTC),
    )

    result = runner.execute(
        [_candidate()],
        finished_at=datetime(2026, 7, 21, 13, 5, tzinfo=UTC),
    )

    candidate = result.candidate_evaluations[0]
    assert candidate.status.value == "INVALID"
    assert all(
        scenario.metrics.error_code == "PERIOD_COVERAGE_MISMATCH"
        for scenario in candidate.scenarios
    )


def test_replay_preflight_rejects_untracked_dynamic_cost_model(tmp_path: Path):
    policy = _policy()
    config = _config(policy)
    data_manifest = _data_manifest(policy)
    code_manifest = _code_manifest()
    config["backtesting"]["dynamic_slippage"] = True
    root = tmp_path / "bad-costs"
    ledger = FinalTestUsageLedgerV2(tmp_path / "final_test.sqlite3")

    with pytest.raises(ArtifactIntegrityError, match="dynamic slippage"):
        ShadowReplayRunnerV2(
            manifest=_manifest(root, config, data_manifest, code_manifest),
            resolved_config=config,
            policy=policy,
            data_manifest=data_manifest,
            code_manifest=code_manifest,
            final_test_ledger=ledger,
            repo_root=str(tmp_path),
            data_manifest_builder=lambda config, policy: data_manifest,
            code_manifest_builder=lambda root: code_manifest,
        )


def test_second_attempt_cannot_reuse_exposed_final_test_cell(tmp_path: Path):
    policy = _policy()
    config = _config(policy)
    data_manifest = _data_manifest(policy)
    code_manifest = _code_manifest()
    ledger = FinalTestUsageLedgerV2(tmp_path / "final_test.sqlite3")
    first_root = tmp_path / "first-attempt"
    first_manifest = _manifest(first_root, config, data_manifest, code_manifest).model_copy(
        update={
            "attempt_id": "first-attempt",
            "artifact_root": str(first_root),
            "resolved_config_path": str(first_root / "resolved_config.yaml"),
        }
    )
    factory = lambda scenario_config: _FakeBacktester(  # noqa: E731
        scenario_config, [], _successful_result()
    )
    first_runner = ShadowReplayRunnerV2(
        manifest=first_manifest,
        resolved_config=config,
        policy=policy,
        data_manifest=data_manifest,
        code_manifest=code_manifest,
        final_test_ledger=ledger,
        repo_root=str(tmp_path),
        backtester_factory=factory,
        data_manifest_builder=lambda config, policy: data_manifest,
        code_manifest_builder=lambda root: code_manifest,
        clock=lambda: datetime(2026, 7, 21, 13, 1, tzinfo=UTC),
    )
    first_runner.execute(
        [_candidate()],
        finished_at=datetime(2026, 7, 21, 13, 5, tzinfo=UTC),
    )

    second_root = tmp_path / "second-attempt"
    second_manifest = first_manifest.model_copy(
        update={
            "attempt_id": "second-attempt",
            "wave_id": "wave-002",
            "artifact_root": str(second_root),
            "resolved_config_path": str(second_root / "resolved_config.yaml"),
        }
    )
    second_runner = ShadowReplayRunnerV2(
        manifest=second_manifest,
        resolved_config=config,
        policy=policy,
        data_manifest=data_manifest,
        code_manifest=code_manifest,
        final_test_ledger=ledger,
        repo_root=str(tmp_path),
        backtester_factory=factory,
        data_manifest_builder=lambda config, policy: data_manifest,
        code_manifest_builder=lambda root: code_manifest,
        clock=lambda: datetime(2026, 7, 21, 14, 1, tzinfo=UTC),
    )

    with pytest.raises(FinalTestReuseError, match="first-attempt"):
        second_runner.execute(
            [_candidate()],
            finished_at=datetime(2026, 7, 21, 14, 5, tzinfo=UTC),
        )
    assert not second_root.exists()


def test_final_panel_is_burned_before_final_backtester_can_fail(tmp_path: Path):
    policy = _policy()
    config = _config(policy)
    data_manifest = _data_manifest(policy)
    code_manifest = _code_manifest()
    ledger = FinalTestUsageLedgerV2(tmp_path / "final_test.sqlite3")
    root = tmp_path / "constructor-failure"
    manifest = _manifest(root, config, data_manifest, code_manifest).model_copy(
        update={
            "attempt_id": "constructor-failure",
            "artifact_root": str(root),
            "resolved_config_path": str(root / "resolved_config.yaml"),
        }
    )

    def factory(scenario_config):
        if scenario_config["backtesting"]["fee"] == 0.002:
            raise RuntimeError("final backtester construction failed")
        return _FakeBacktester(scenario_config, [], _successful_result())

    runner = ShadowReplayRunnerV2(
        manifest=manifest,
        resolved_config=config,
        policy=policy,
        data_manifest=data_manifest,
        code_manifest=code_manifest,
        final_test_ledger=ledger,
        repo_root=str(tmp_path),
        backtester_factory=factory,
        data_manifest_builder=lambda config, policy: data_manifest,
        code_manifest_builder=lambda root: code_manifest,
        clock=lambda: datetime(2026, 7, 21, 13, 1, tzinfo=UTC),
    )

    with pytest.raises(RuntimeError, match="construction failed"):
        runner.execute(
            [_candidate()],
            finished_at=datetime(2026, 7, 21, 13, 5, tzinfo=UTC),
        )
    assert ledger.get(manifest.attempt_id).status == "EXPOSED"


def test_frozen_candidate_hash_covers_code_and_external_execution_parameters():
    candidate = _candidate()
    payload = candidate.model_dump(mode="python")
    payload["max_open_trades"] = 4

    with pytest.raises(ValueError, match="phenotype_hash"):
        FrozenCandidateV2.model_validate(payload)

    changed_code = candidate.model_dump(mode="python")
    changed_code["strategy_code"] += "# executable snapshot changed\n"
    with pytest.raises(ValueError, match="phenotype_hash"):
        FrozenCandidateV2.model_validate(changed_code)

    with pytest.raises(ValueError, match="literal timeframe"):
        freeze_candidate(
            candidate_id="bad-timeframe",
            strategy_name="FrozenStrategy",
            strategy_code="class FrozenStrategy:\n    timeframe = '5m'\n",
            timeframe="1h",
            max_open_trades=3,
        )
