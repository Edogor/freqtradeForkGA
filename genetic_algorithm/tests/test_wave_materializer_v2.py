"""Tests for immutable child-wave materialization and atomic queue handoff."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from freqtrade.misc import pair_to_filename
from genetic_algorithm.config.schema import load_config
from genetic_algorithm.core.individual import Individual
from genetic_algorithm.genome.codegen import StrategyGenerator
from genetic_algorithm.orchestration.artifact_store_v2 import (
    ArtifactIntegrityError,
    V2ArtifactStore,
    canonical_config_hash,
)
from genetic_algorithm.orchestration.attempt_state_v2 import AttemptLifecycleStatus
from genetic_algorithm.orchestration.candidate_selector_v2 import (
    CandidateSelectionPolicyV2,
    select_wave_candidates,
)
from genetic_algorithm.orchestration.evolution_worker_v2 import (
    EvolutionWorkerSpecV2,
    freeze_evolution_seed,
)
from genetic_algorithm.orchestration.promotion_policy_v2 import (
    ScenarioRequirementV2,
    ShadowGatePolicyV2,
)
from genetic_algorithm.orchestration.result_contract import (
    AttemptManifestV2,
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
    CandidateAnalysisV2,
    CandidateEligibilityStatus,
    CandidateObservationRefV2,
    ExperimentAnalysisV2,
    WaveAnalysisV2,
    WaveAnalyzerPolicyV2,
)
from genetic_algorithm.orchestration.wave_materializer_v2 import (
    WaveMaterializationError,
    materialize_evolution_wave,
    materialize_replay_validation_wave,
    queue_approved_materialization,
)
from genetic_algorithm.orchestration.wave_planner_v2 import (
    ChildWavePlanV2,
    PlannerSourceMode,
    WaveArmTemplateV2,
    WavePlanMode,
    WavePlannerPolicyV2,
    plan_child_wave,
)
from genetic_algorithm.orchestration.wave_state_v2 import (
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
from genetic_algorithm.tests.v2_metric_fixtures import expectancy_metrics


NOW = datetime(2026, 7, 22, 8, 0, tzinfo=UTC)
PARENT_WAVE_ID = "wave-materializer-parent"
PARENT_EXPERIMENT_ID = "experiment-materializer-parent"
RESULT_POLICY = "materializer-result-policy-v2"


@dataclass(frozen=True)
class _Context:
    repo_root: Path
    data_path: Path
    config: dict
    parent_manifest: AttemptManifestV2
    parent_experiment: ExperimentSpecV2
    analysis: WaveAnalysisV2
    plan: ChildWavePlanV2


def _scenario_metrics(pair: str) -> ScenarioMetricsV2:
    return ScenarioMetricsV2(
        scenario_id="materializer-validation-cell",
        pair=pair,
        timeframe="1h",
        role=ScenarioRole.TRAIN,
        period_start=date(2024, 1, 1),
        period_end=date(2024, 1, 2),
        cost_multiplier=1.0,
        status=EvaluationStatus.VALID,
        success=True,
        net_return=0.08,
        annualized_net_return=0.12,
        annualized_net_return_lcb=0.05,
        max_drawdown=0.08,
        max_drawdown_ucb=0.12,
        daily_expected_shortfall_5=0.01,
        daily_expected_shortfall_5_ucb=0.02,
        net_expectancy=0.003,
        net_expectancy_lcb=0.001,
        profit_factor=1.6,
        profit_factor_censored=False,
        profit_factor_contract_version="right-censored-profit-factor-v1",
        win_rate=0.58,
        trade_count=40,
        effective_sample_size=32,
        **expectancy_metrics(
            trade_count=40,
            mean_return=0.003,
            lower_confidence_bound=0.001,
            effective_sample_size=32,
        ),
        active_months=4,
        max_consecutive_losses=3,
        max_drawdown_duration_days=12,
    )


@pytest.fixture
def context(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[2]
    suffix = hashlib.sha256(str(tmp_path).encode()).hexdigest()[:12].upper()
    pair = f"UNITTEST_MATERIALIZER_{suffix}/USDT"
    data_root = repo_root / "tests" / "testdata"
    data_root.mkdir(parents=True, exist_ok=True)
    data_path = data_root / f"{pair_to_filename(pair)}-1h.feather"
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
            policy_version=RESULT_POLICY,
            required_scenarios=[
                ScenarioRequirementV2(
                    scenario_id="materializer-validation-cell",
                    pair=pair,
                    timeframe="1h",
                    role=ScenarioRole.TRAIN,
                    period_start=date(2024, 1, 1),
                    period_end=date(2024, 1, 2),
                    cost_multiplier=1.0,
                )
            ],
            min_effective_sample_size=1,
            min_active_months=1,
            require_final_test=False,
        )
        config = load_config("genetic_algorithm/config/presets/safe_v2.yaml")
        config["backtesting"].update(
            {
                "pairs": [pair],
                "timerange": "20240101-20240103",
                "timeframe": "1h",
                "dataformat_ohlcv": "feather",
                "dynamic_slippage": False,
                "spread_pct": 0.0,
                "funding_rate": 0.0,
            }
        )
        config["promotion_v2"] = {"enabled": True, **policy.model_dump(mode="json")}
        config["parallel_evaluation"].update({"enabled": False, "num_workers": 1})
        parent_root = tmp_path / "parent" / "attempt-parent"
        parent_manifest = AttemptManifestV2(
            attempt_id="attempt-materializer-parent",
            wave_id=PARENT_WAVE_ID,
            experiment_id=PARENT_EXPERIMENT_ID,
            created_at=NOW,
            config_hash=canonical_config_hash(config),
            code_version="materializer-parent-test",
            data_manifest_hash="d" * 64,
            split_manifest_hash="s" * 64,
            fitness_policy_version=RESULT_POLICY,
            seeds=[11],
            worker_count=1,
            resolved_config_path=str(parent_root / V2ArtifactStore.CONFIG_NAME),
            artifact_root=str(parent_root),
        )
        parent_gene = StrategyGenerator(config).generate_random_strategy(
            generation=3,
            individual_id=7,
        )
        parent_individual = Individual(strategy_gene=parent_gene)
        parent_individual.set_fitness(1.0, {"profit": 8.0})
        evolution_seed = freeze_evolution_seed(
            parent_individual,
            resolved_config=config,
            candidate_id="candidate-materializer-parent",
        )
        candidate = evolution_seed.frozen_candidate
        artifacts = V2ArtifactStore(parent_root)
        artifacts.write_manifest(parent_manifest, config)
        artifacts.write_frozen_candidate(candidate)
        artifacts.write_evolution_seed(evolution_seed)
        record = BacktestRecordV2(
            attempt_id=parent_manifest.attempt_id,
            wave_id=PARENT_WAVE_ID,
            experiment_id=PARENT_EXPERIMENT_ID,
            candidate_id=candidate.candidate_id,
            config_hash=parent_manifest.config_hash,
            phenotype_hash=candidate.phenotype_hash,
            code_version=parent_manifest.code_version,
            data_manifest_hash=parent_manifest.data_manifest_hash,
            fitness_policy_version=RESULT_POLICY,
            seed=11,
            worker_count=1,
            fee_rate=0.001,
            slippage_rate=0.0005,
            spread_rate=0.0,
            funding_rate=0.0,
            equity_method="MARK_TO_MARKET",
            metrics=_scenario_metrics(pair),
            daily_net_returns=[0.01, -0.004, 0.006],
            equity_curve=[100.0, 101.0, 100.596, 101.199576],
            trades=[
                {"pair": pair, "profit_ratio": 0.01 if index % 3 else -0.004}
                for index in range(40)
            ],
        )
        evaluation = CandidateEvaluationV2(
            candidate_id=candidate.candidate_id,
            phenotype_hash=candidate.phenotype_hash,
            fitness_policy_version=RESULT_POLICY,
            status=EvaluationStatus.VALID,
            scenarios=[record],
            gates=[
                GateResultV2(
                    gate_id="MATERIALIZER_GATE",
                    passed=True,
                    status=EvaluationStatus.VALID,
                    observed=0.05,
                    threshold=0.0,
                    operator=">=",
                    reason_code="PASS",
                )
            ],
            robust_score=0.02,
        )
        artifacts.write_backtest(record)
        artifacts.write_candidate(evaluation)
        artifacts.finalize(
            {
                "attempt_id": parent_manifest.attempt_id,
                "status": AttemptStatus.SUCCEEDED,
                "started_at": NOW + timedelta(seconds=1),
                "finished_at": NOW + timedelta(seconds=2),
                "manifest": parent_manifest.model_dump(mode="json"),
                "candidate_evaluations": [evaluation.model_dump(mode="json")],
            }
        )
        result_path = parent_root / V2ArtifactStore.RESULT_NAME
        result_sha = hashlib.sha256(result_path.read_bytes()).hexdigest()
        parent_experiment = ExperimentSpecV2(
            experiment_id=PARENT_EXPERIMENT_ID,
            wave_id=PARENT_WAVE_ID,
            arm_type=ExperimentArmType.CONTROL,
            hypothesis="Parent control for materialization tests.",
            primary_metric="annualized_net_return_lcb",
            seeds=[11],
            resolved_config_hash=parent_manifest.config_hash,
            expected_attempts=[
                AttemptExpectationV2(
                    attempt_id=parent_manifest.attempt_id,
                    manifest_hash=canonical_config_hash(
                        parent_manifest.model_dump(mode="json")
                    ),
                    seed=11,
                    ordinal=0,
                )
            ],
            created_at=NOW,
        )
        candidate_analysis = CandidateAnalysisV2(
            experiment_id=PARENT_EXPERIMENT_ID,
            phenotype_hash=candidate.phenotype_hash,
            candidate_ids=[candidate.candidate_id],
            seeds=[11],
            observation_refs=[
                CandidateObservationRefV2(
                    attempt_id=parent_manifest.attempt_id,
                    candidate_id=candidate.candidate_id,
                    seed=11,
                    result_path=str(result_path.resolve()),
                    result_sha256=result_sha,
                )
            ],
            observation_count=1,
            valid_observation_count=1,
            eligibility_status=CandidateEligibilityStatus.ELIGIBLE,
            reason_codes=["CANDIDATE_ELIGIBLE"],
            comparison_panel_hash="c" * 64,
            median_robust_score=0.02,
            worst_annualized_return_lcb=0.05,
            median_annualized_return_lcb=0.05,
            worst_max_drawdown_ucb=0.12,
            worst_daily_es5_ucb=0.02,
            worst_net_expectancy_lcb=0.001,
            median_profit_factor=1.6,
            median_win_rate=0.58,
            min_effective_sample_size=32,
            min_trades_per_active_month=10,
            scenario_trade_count_sum=40,
        )
        analyzer_policy = WaveAnalyzerPolicyV2(
            analysis_policy_version="materializer-analysis-v2",
            required_result_policy_version=RESULT_POLICY,
            min_candidate_seed_count=1,
        )
        analysis = WaveAnalysisV2(
            wave_id=PARENT_WAVE_ID,
            created_at=NOW + timedelta(seconds=3),
            snapshot_hash="r" * 64,
            result_policy_version=RESULT_POLICY,
            analyzer_policy=analyzer_policy,
            analyzer_policy_hash=analyzer_policy.policy_hash,
            experiments=[
                ExperimentAnalysisV2(
                    experiment_id=PARENT_EXPERIMENT_ID,
                    arm_type=ExperimentArmType.CONTROL,
                    expected_attempt_count=1,
                    verified_result_count=1,
                    successful_attempt_count=1,
                    documented_abort_count=0,
                    candidate_observation_count=1,
                    abort_fraction=0,
                    non_success_fraction=0,
                    health_status=AnalysisHealthStatus.HEALTHY,
                    reason_codes=["EXPERIMENT_HEALTHY"],
                )
            ],
            candidates=[candidate_analysis],
            planning_allowed=True,
            has_eligible_candidates=True,
            reason_codes=["ANALYSIS_COMPLETE"],
        )
        selection = select_wave_candidates(
            analysis,
            CandidateSelectionPolicyV2(
                selection_policy_version="materializer-selection-v2",
                min_selected=1,
                max_selected=1,
                max_per_experiment=1,
                min_normalized_objective_distance=0,
            ),
        )
        planner_policy = WavePlannerPolicyV2(
            planner_policy_version="materializer-planner-v2",
            child_result_policy_version=RESULT_POLICY,
            child_search_space_version="materializer-panel-v2",
            plan_mode=WavePlanMode.REPLAY_VALIDATION,
            budget=WaveBudgetV2(
                max_attempts=1,
                max_parallel=1,
                max_wallclock_seconds=3600,
            ),
            arm_templates=[
                WaveArmTemplateV2(
                    arm_id="validation",
                    arm_type=ExperimentArmType.VALIDATION,
                    source_mode=PlannerSourceMode.SELECTED_CANDIDATES,
                    hypothesis="Replay the selected executable without evolution.",
                    primary_metric="annualized_net_return_lcb",
                    seeds=[101],
                )
            ],
        )
        plan = plan_child_wave(
            analysis,
            selection,
            [parent_experiment],
            {parent_manifest.config_hash: config},
            planner_policy,
        )
        yield _Context(
            repo_root=repo_root,
            data_path=data_path,
            config=config,
            parent_manifest=parent_manifest,
            parent_experiment=parent_experiment,
            analysis=analysis,
            plan=plan,
        )
    finally:
        data_path.unlink(missing_ok=True)


def _materialize(context: _Context, tmp_path: Path):
    return materialize_replay_validation_wave(
        context.plan,
        materialization_root=tmp_path / "materialized",
        repo_root=context.repo_root,
        final_test_ledger_path=tmp_path / "ledgers" / "final-test.sqlite3",
        materialized_at=NOW + timedelta(seconds=4),
    )


def _evolution_plan(context: _Context) -> ChildWavePlanV2:
    selection = select_wave_candidates(
        context.analysis,
        CandidateSelectionPolicyV2(
            selection_policy_version="materializer-evolution-selection-v2",
            min_selected=1,
            max_selected=1,
            max_per_experiment=1,
            min_normalized_objective_distance=0,
        ),
    )
    policy = WavePlannerPolicyV2(
        planner_policy_version="materializer-evolution-planner-v2",
        child_result_policy_version=RESULT_POLICY,
        child_search_space_version="materializer-evolution-space-v2",
        plan_mode=WavePlanMode.EVOLUTION_EXPERIMENT,
        budget=WaveBudgetV2(
            max_attempts=4,
            max_parallel=1,
            max_wallclock_seconds=3600,
        ),
        arm_templates=[
            WaveArmTemplateV2(
                arm_id="control",
                arm_type=ExperimentArmType.CONTROL,
                source_mode=PlannerSourceMode.BASELINE_CONTROL,
                hypothesis="Fresh baseline evolution.",
                primary_metric="annualized_net_return_lcb",
                seeds=[101],
            ),
            WaveArmTemplateV2(
                arm_id="replication",
                arm_type=ExperimentArmType.REPLICATION,
                source_mode=PlannerSourceMode.SELECTED_CANDIDATES,
                hypothesis="Repeat evolution from the selected genome.",
                primary_metric="annualized_net_return_lcb",
                seeds=[101],
            ),
            WaveArmTemplateV2(
                arm_id="exploit",
                arm_type=ExperimentArmType.EXPLOIT,
                source_mode=PlannerSourceMode.SELECTED_CANDIDATES,
                hypothesis="Exploit near the selected genome.",
                primary_metric="annualized_net_return_lcb",
                factor_delta={"genetic_algorithm.mutation_rate": 0.22},
                seeds=[101],
            ),
            WaveArmTemplateV2(
                arm_id="explore",
                arm_type=ExperimentArmType.EXPLORE,
                source_mode=PlannerSourceMode.SELECTED_CANDIDATES,
                hypothesis="Explore around the selected genome.",
                primary_metric="annualized_net_return_lcb",
                factor_delta={"genetic_algorithm.crossover_rate": 0.70},
                seeds=[101],
            ),
        ],
    )
    return plan_child_wave(
        context.analysis,
        selection,
        [context.parent_experiment],
        {context.parent_manifest.config_hash: context.config},
        policy,
    )


def _approved_parent_store(
    context: _Context,
    tmp_path: Path,
    prepared,
) -> WaveStateStoreV2:
    store = WaveStateStoreV2(tmp_path / "orchestration.sqlite3")
    store.register_wave(
        WaveSpecV2(
            wave_id=PARENT_WAVE_ID,
            policy_version=RESULT_POLICY,
            search_space_version="parent-search-space-v2",
            budget=WaveBudgetV2(
                max_attempts=1,
                max_parallel=1,
                max_wallclock_seconds=3600,
            ),
            created_at=NOW,
        )
    )
    store.register(context.parent_manifest)
    store.transition(
        context.parent_manifest.attempt_id,
        AttemptLifecycleStatus.VALIDATED,
        occurred_at=NOW + timedelta(milliseconds=100),
        actor="test-validator",
        reason="VALIDATED",
    )
    store.transition(
        context.parent_manifest.attempt_id,
        AttemptLifecycleStatus.QUEUED,
        occurred_at=NOW + timedelta(milliseconds=200),
        actor="test-controller",
        reason="QUEUED",
    )
    store.register_experiment(context.parent_experiment)
    store.begin_collection(
        PARENT_WAVE_ID,
        started_at=NOW + timedelta(milliseconds=300),
    )
    claimed = store.claim_next(
        worker_id="materializer-parent-worker",
        claimed_at=NOW + timedelta(milliseconds=400),
        lease_seconds=60,
    )
    assert claimed is not None and claimed.claim_token is not None
    log_path = Path(context.parent_manifest.artifact_root) / "runtime" / "attempt.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("parent completed\n")
    store.prepare_execution(
        context.parent_manifest.attempt_id,
        claim_token=claimed.claim_token,
        command_hash="e" * 64,
        working_directory=context.repo_root,
        log_path=log_path,
        prepared_at=NOW + timedelta(milliseconds=500),
    )
    process_identity = f"process-{context.parent_manifest.attempt_id}"
    store.mark_running(
        context.parent_manifest.attempt_id,
        claim_token=claimed.claim_token,
        pid=12345,
        process_start_token=process_identity,
        started_at=NOW + timedelta(milliseconds=600),
        lease_seconds=60,
    )
    store.finalize_from_artifacts(
        context.parent_manifest.attempt_id,
        claim_token=claimed.claim_token,
        process_start_token=process_identity,
        finalized_at=NOW + timedelta(seconds=2, milliseconds=500),
    )
    reconciled = store.reconcile_wave(
        PARENT_WAVE_ID,
        reconciled_at=NOW + timedelta(seconds=3),
    )
    analysis_decision = WaveDecisionV2(
        decision_id="analysis-materializer-parent",
        wave_id=PARENT_WAVE_ID,
        decision_type=WaveDecisionType.ANALYSIS,
        created_at=NOW + timedelta(seconds=3, milliseconds=100),
        actor="wave-analyzer-v2",
        reason_codes=["ANALYSIS_COMPLETE"],
        input_hash=reconciled.result_snapshot_hash,
    )
    store.apply_decision(analysis_decision)
    proposal = prepared.materialization.to_proposal_decision(
        analysis_decision_hash=analysis_decision.decision_hash
    )
    store.apply_decision(proposal)
    approval = prepared.materialization.to_approval_decision(
        proposal=proposal,
        approved_at=NOW + timedelta(seconds=5),
        actor="test-operator",
        reason_codes=["MANUAL_APPROVAL"],
    )
    store.apply_decision(approval)
    return store


def test_materialization_is_idempotent_and_binds_verified_candidate(
    context: _Context,
    tmp_path: Path,
):
    first = _materialize(context, tmp_path)
    second = _materialize(context, tmp_path)

    assert first.materialization == second.materialization
    assert first.receipt_file_sha256 == second.receipt_file_sha256
    assert first.materialization.plan_hash == context.plan.plan_hash
    source = first.materialization.experiments[0].verified_source
    assert source.parent_attempt_id == context.parent_manifest.attempt_id
    assert source.frozen_candidate.phenotype_hash == source.phenotype_hash
    assert first.materialization.attempts[0].worker_binding.worker_kind.value == "SHADOW_REPLAY"


def test_evolution_materialization_binds_all_arm_types_and_queues_atomically(
    context: _Context,
    tmp_path: Path,
):
    plan = _evolution_plan(context)
    prepared = materialize_evolution_wave(
        plan,
        materialization_root=tmp_path / "evolution-materialized",
        repo_root=context.repo_root,
        final_test_ledger_path=tmp_path / "ledgers" / "final-test.sqlite3",
        materialized_at=NOW + timedelta(seconds=4),
    )

    assert {item.experiment_spec.arm_type for item in prepared.materialization.experiments} == {
        ExperimentArmType.CONTROL,
        ExperimentArmType.REPLICATION,
        ExperimentArmType.EXPLOIT,
        ExperimentArmType.EXPLORE,
    }
    assert all(
        item.worker_binding.worker_kind == "STANDARD_EVOLUTION"
        for item in prepared.materialization.attempts
    )
    control = next(
        item
        for item in prepared.materialization.experiments
        if item.experiment_spec.arm_type == ExperimentArmType.CONTROL
    )
    assert control.verified_source is None
    selected = [
        item
        for item in prepared.materialization.experiments
        if item.experiment_spec.arm_type != ExperimentArmType.CONTROL
    ]
    assert all(item.verified_source.evolution_seed is not None for item in selected)
    arm_by_experiment = {
        item.experiment_spec.experiment_id: item.experiment_spec.arm_type
        for item in prepared.materialization.experiments
    }
    for attempt in prepared.materialization.attempts:
        spec = EvolutionWorkerSpecV2.model_validate_json(
            Path(attempt.worker_binding.spec_path).read_bytes()
        )
        assert spec.replay_input_seeds is (
            arm_by_experiment[attempt.manifest.experiment_id]
            == ExperimentArmType.REPLICATION
        )

    store = _approved_parent_store(context, tmp_path, prepared)
    queue_approved_materialization(
        store,
        prepared,
        queued_at=NOW + timedelta(seconds=6),
    )
    assert all(
        store.get(item.manifest.attempt_id).status == AttemptLifecycleStatus.QUEUED
        for item in prepared.materialization.attempts
    )


def test_materialization_rejects_tampered_parent_and_unsupported_evolution(
    context: _Context,
    tmp_path: Path,
):
    frozen_path = (
        Path(context.parent_manifest.artifact_root)
        / "candidates"
        / "candidate-materializer-parent"
        / "frozen_candidate.json"
    )
    frozen_path.write_bytes(frozen_path.read_bytes() + b"\n")
    with pytest.raises(ArtifactIntegrityError, match="artifact manifest mismatch"):
        _materialize(context, tmp_path)

    evolution_plan = context.plan.model_copy(
        update={
            "planner_policy": context.plan.planner_policy.model_copy(
                update={"plan_mode": WavePlanMode.EVOLUTION_EXPERIMENT}
            )
        }
    )
    with pytest.raises(WaveMaterializationError, match="evolution worker is not implemented"):
        materialize_replay_validation_wave(
            evolution_plan,
            materialization_root=tmp_path / "unsupported",
            repo_root=context.repo_root,
            final_test_ledger_path=tmp_path / "ledger.sqlite3",
            materialized_at=NOW + timedelta(seconds=4),
        )


def test_materialization_rejects_unknown_resolved_config_path(
    context: _Context,
    tmp_path: Path,
):
    payload = context.plan.model_dump(mode="json")
    experiment = payload["experiments"][0]
    experiment["resolved_config"]["backtesting"]["slipage_pct"] = 0.01
    changed_hash = canonical_config_hash(experiment["resolved_config"])
    experiment["resolved_config_hash"] = changed_hash
    for attempt in experiment["attempts"]:
        attempt["resolved_config_hash"] = changed_hash
    plan = ChildWavePlanV2.model_validate(payload)

    with pytest.raises(WaveMaterializationError, match=r"backtesting\.slipage_pct"):
        materialize_replay_validation_wave(
            plan,
            materialization_root=tmp_path / "unknown-config",
            repo_root=context.repo_root,
            final_test_ledger_path=tmp_path / "ledger.sqlite3",
            materialized_at=NOW + timedelta(seconds=4),
        )


def test_approval_and_all_child_queue_rows_commit_atomically(context: _Context, tmp_path: Path):
    prepared = _materialize(context, tmp_path)
    store = _approved_parent_store(context, tmp_path, prepared)

    with pytest.raises(WaveStateError, match="approval differs"):
        store.queue_materialized_child_wave(
            parent_wave_id=PARENT_WAVE_ID,
            child_wave=prepared.materialization.child_wave_spec,
            experiments=[
                item.experiment_spec for item in prepared.materialization.experiments
            ],
            attempts=[
                (item.manifest, item.worker_binding)
                for item in prepared.materialization.attempts
            ],
            plan_hash="0" * 64,
            materialization_hash=prepared.materialization.materialization_hash,
            queued_at=NOW + timedelta(seconds=6),
        )
    with pytest.raises(WaveStateError, match="unknown wave"):
        store.get_wave(prepared.materialization.child_wave_spec.wave_id)

    queue_approved_materialization(
        store,
        prepared,
        queued_at=NOW + timedelta(seconds=6),
    )
    assert store.get_wave(PARENT_WAVE_ID).status == WaveLifecycleStatus.QUEUED
    child = store.get_wave(prepared.materialization.child_wave_spec.wave_id)
    assert child.status == WaveLifecycleStatus.DRAFT
    attempt = store.get(prepared.materialization.attempts[0].manifest.attempt_id)
    assert attempt.status == AttemptLifecycleStatus.QUEUED
    assert attempt.worker_binding_hash == prepared.materialization.attempts[0].worker_binding_hash

    queue_approved_materialization(
        store,
        prepared,
        queued_at=NOW + timedelta(seconds=6),
    )
    assert store.begin_collection(
        child.wave_id,
        started_at=NOW + timedelta(seconds=7),
    ).status == WaveLifecycleStatus.COLLECTING


def test_receipt_tamper_blocks_queue_without_partial_child_state(
    context: _Context,
    tmp_path: Path,
):
    prepared = _materialize(context, tmp_path)
    store = _approved_parent_store(context, tmp_path, prepared)
    prepared.receipt_path.write_bytes(prepared.receipt_path.read_bytes() + b"\n")

    with pytest.raises(ArtifactIntegrityError, match="receipt SHA-256 differs"):
        queue_approved_materialization(
            store,
            prepared,
            queued_at=NOW + timedelta(seconds=6),
        )
    assert store.get_wave(PARENT_WAVE_ID).status == WaveLifecycleStatus.APPROVED
    with pytest.raises(WaveStateError, match="unknown wave"):
        store.get_wave(prepared.materialization.child_wave_spec.wave_id)


def test_database_conflict_rolls_back_entire_child_handoff(
    context: _Context,
    tmp_path: Path,
):
    prepared = _materialize(context, tmp_path)
    store = _approved_parent_store(context, tmp_path, prepared)
    materialized_attempt = prepared.materialization.attempts[0]
    conflicting = materialized_attempt.manifest.model_copy(
        update={
            "wave_id": "unrelated-wave",
            "experiment_id": "unrelated-experiment",
        }
    )
    store.register(conflicting)

    with pytest.raises(WaveStateError, match="conflicts with existing immutable state"):
        queue_approved_materialization(
            store,
            prepared,
            queued_at=NOW + timedelta(seconds=6),
        )

    assert store.get_wave(PARENT_WAVE_ID).status == WaveLifecycleStatus.APPROVED
    with pytest.raises(WaveStateError, match="unknown wave"):
        store.get_wave(prepared.materialization.child_wave_spec.wave_id)
    assert store.get(conflicting.attempt_id).wave_id == "unrelated-wave"
