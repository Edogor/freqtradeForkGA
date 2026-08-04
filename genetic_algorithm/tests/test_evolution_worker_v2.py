"""Canonical standard-evolution worker and scheduler contract tests."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from freqtrade.misc import pair_to_filename
from genetic_algorithm.config.schema import load_config
from genetic_algorithm.core.individual import Individual
from genetic_algorithm.evaluation.direct_backtester import BacktestResult
from genetic_algorithm.genome.codegen import StrategyGenerator
from genetic_algorithm.orchestration.artifact_store_v2 import (
    V2ArtifactStore,
    canonical_config_hash,
)
from genetic_algorithm.orchestration.attempt_scheduler_v2 import AttemptSchedulerV2
from genetic_algorithm.orchestration.attempt_state_v2 import (
    AttemptLifecycleStatus,
    AttemptStateStoreV2,
    WorkerKind,
)
from genetic_algorithm.orchestration.evolution_scheduler_v2 import (
    EvolutionSchedulerV2,
    queue_prepared_evolution_attempt,
)
from genetic_algorithm.orchestration.evolution_worker_v2 import (
    EvolutionWorkerError,
    FrozenEvolutionSeedV2,
    LoadedEvolutionWorkerV2,
    PreparedEvolutionWorkerV2,
    derive_engine_config,
    derive_search_seed,
    freeze_evolution_seed,
    load_evolution_worker,
    prepare_evolution_worker,
    run_evolution_worker,
)
from genetic_algorithm.orchestration.manifest_builder_v2 import build_attempt_manifest_bundle
from genetic_algorithm.orchestration.output_layout_v2 import AttemptOutputLayoutV2
from genetic_algorithm.orchestration.promotion_policy_v2 import (
    ScenarioRequirementV2,
    ShadowGatePolicyV2,
)
from genetic_algorithm.orchestration.result_contract import (
    AttemptManifestV2,
    AttemptStatus,
    EvaluationStatus,
)
from genetic_algorithm.orchestration.runner_v2 import (
    prepare_and_queue_standard_attempt,
)
from genetic_algorithm.orchestration.shadow_scheduler_v2 import ShadowSchedulerConfigV2


@dataclass(frozen=True)
class _Context:
    prepared: PreparedEvolutionWorkerV2
    config: dict[str, Any]
    repo_root: Path
    ledger_path: Path


@pytest.fixture
def evolution_context_factory(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[2]
    suffix = hashlib.sha256(str(tmp_path).encode()).hexdigest()[:12].upper()
    pair = f"UNITTESTEVOLUTIONV2{suffix}/BTC"
    validation_pair = f"UNITTESTEVOLUTIONVALV2{suffix}/BTC"
    data_root = repo_root / "tests" / "testdata"
    data_root.mkdir(parents=True, exist_ok=True)
    data_paths = [
        data_root / f"{pair_to_filename(item)}-1h.feather"
        for item in (pair, validation_pair)
    ]
    candle_count = 240
    candles = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=candle_count, freq="1h", tz="UTC"),
            "open": [100.0] * candle_count,
            "high": [101.0] * candle_count,
            "low": [99.0] * candle_count,
            "close": [100.5] * candle_count,
            "volume": [10.0] * candle_count,
        }
    )
    for data_path in data_paths:
        candles.to_feather(data_path)
    sequence = 0

    def factory(
        *,
        seeds: list[FrozenEvolutionSeedV2] | None = None,
        generations: int = 1,
        generic_island: bool = False,
        search_seed_salt: int = 0,
    ) -> _Context:
        nonlocal sequence
        sequence += 1
        scenarios = [
            ScenarioRequirementV2(
                scenario_id="evolution-train-cell",
                pair=pair,
                timeframe="1h",
                role="TRAIN",
                period_start=date(2024, 1, 1),
                period_end=date(2024, 1, 2),
                cost_multiplier=1.0,
            )
        ]
        if generic_island:
            scenarios.append(
                ScenarioRequirementV2(
                    scenario_id="evolution-pair-validation-cell",
                    pair=validation_pair,
                    timeframe="1h",
                    role="PAIR_VALIDATION",
                    period_start=date(2024, 1, 1),
                    period_end=date(2024, 1, 2),
                    cost_multiplier=1.0,
                )
            )
        policy = ShadowGatePolicyV2(
            policy_version="evolution-worker-policy-v2-test",
            required_scenarios=scenarios,
            min_effective_sample_size=1,
            min_active_months=1,
            min_trades_per_active_month=0.0,
            require_final_test=False,
        )
        config = load_config("genetic_algorithm/config/presets/safe_v2.yaml")
        config["backtesting"].update(
            {
                "pairs": [pair, validation_pair] if generic_island else [pair],
                "timerange": "20240101-20240111",
                "timeframe": "1h",
                "dataformat_ohlcv": "feather",
                "auto_download_data": False,
                "enable_cache": False,
                "dynamic_slippage": False,
                "spread_pct": 0.0,
                "funding_rate": 0.0,
            }
        )
        config["genetic_algorithm"].update(
            {
                "population_size": 4,
                "generations": generations,
                "elite_size": 1,
                "tournament_size": 2,
                "random_immigrants": 1,
                "search_seed_salt": search_seed_salt,
            }
        )
        config["parallel_evaluation"].update({"enabled": False, "num_workers": 1})
        config["output"]["top_n"] = 1
        if generic_island:
            config["safety_profile"].update(
                {
                    "name": "automation_island_v2",
                    "automation_eligible": False,
                }
            )
            config["genetic_algorithm"]["max_runtime_minutes"] = 10
            config["generic_island_model"].update(
                {
                    "enabled": True,
                    "num_islands": 2,
                    "population_per_island": 4,
                    "generations": generations,
                    "parallel_islands": False,
                    "specialization": {
                        "rotate_seeds": True,
                        "indicator_pools": False,
                        "indicator_overlap": 0.5,
                        "pair_rotation": False,
                        "pair_subset_size": 1,
                    },
                    "migration": {
                        "topology": "ring",
                        "interval": 1,
                        "count": 1,
                        "merge_rounds": False,
                        "merge_interval": 2,
                        "tournament_size": 2,
                    },
                    "walk_forward": {"enabled": False},
                    "external_migration": {"enabled": False, "directory": ""},
                    "islands": [
                        {
                            "name": "island-a",
                            "population_size": 4,
                            "generations": generations,
                            "seed": 9001,
                            "indicator_pool": ["RSI", "EMA"],
                            "pairs": [pair, validation_pair],
                            "walk_forward_enabled": False,
                        },
                        {
                            "name": "island-b",
                            "population_size": 4,
                            "generations": generations,
                            "seed": 9002,
                            "indicator_pool": ["MACD", "BBANDS"],
                            "pairs": [pair, validation_pair],
                            "walk_forward_enabled": False,
                        },
                    ],
                }
            )
            config["pair_validation"] = {
                "enabled": True,
                "evaluation_mode": "independent_pairs",
                "training_pairs": [pair],
                "validation_pairs": [validation_pair],
                "weight_train": 0.6,
                "weight_val": 0.4,
                "min_val_fitness": 0.0,
                "validate_top_n_only": 0,
            }
        config["promotion_v2"] = {"enabled": True, **policy.model_dump(mode="json")}
        created_at = datetime.now(UTC) - timedelta(seconds=20)
        attempt_id = f"attempt-evolution-worker-{sequence}"
        artifact_root = tmp_path / "attempts" / attempt_id
        bundle = build_attempt_manifest_bundle(
            attempt_id=attempt_id,
            wave_id="wave-evolution-worker-v2",
            experiment_id=f"experiment-evolution-worker-{sequence}",
            created_at=created_at,
            resolved_config=config,
            policy=policy,
            fitness_policy_version=policy.policy_version,
            seeds=[100 + sequence],
            worker_count=1,
            artifact_root=artifact_root,
            repo_root=repo_root,
            data_root=data_root,
        )
        ledger = tmp_path / "ledgers" / f"final-test-{sequence}.sqlite3"
        prepared = prepare_evolution_worker(
            bundle=bundle,
            resolved_config=config,
            policy=policy,
            seeds=seeds or [],
            repo_root=repo_root,
            final_test_ledger_path=ledger,
            created_at=created_at + timedelta(seconds=1),
            top_n=1,
        )
        return _Context(prepared, config, repo_root, ledger)

    try:
        yield factory
    finally:
        for data_path in data_paths:
            data_path.unlink(missing_ok=True)


def _evaluated_individual(config: dict[str, Any], *, fitness: float = 1.25) -> Individual:
    gene = StrategyGenerator(config).generate_random_strategy(generation=7, individual_id=9)
    individual = Individual(strategy_gene=gene)
    individual.set_fitness(fitness, {"profit": 999.0})
    return individual


class _FailingBacktester:
    def backtest_strategy(self, strategy_code, strategy_name, **kwargs):
        return BacktestResult(
            success=False,
            strategy_name=strategy_name,
            error_message="deliberate unit-test replay failure",
        )


def _failing_backtester_factory(config):
    return _FailingBacktester()


def _rewrite_with_hash_consistent_invalid_config(
    prepared: PreparedEvolutionWorkerV2,
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


def test_prepared_evolution_worker_roundtrips_all_bound_inputs(evolution_context_factory):
    context = evolution_context_factory()

    loaded = load_evolution_worker(
        context.prepared.spec_path,
        expected_spec_sha256=context.prepared.spec_file_sha256,
    )

    assert isinstance(loaded, LoadedEvolutionWorkerV2)
    assert loaded.resolved_config == context.config
    assert loaded.seeds == ()
    assert loaded.spec.worker_kind == "STANDARD_EVOLUTION"
    assert loaded.spec.output_layout == AttemptOutputLayoutV2.for_artifact_root(
        loaded.spec.artifact_root
    )
    assert not context.ledger_path.exists()


def test_completed_evolution_worker_can_be_inspected_after_restart(
    evolution_context_factory,
):
    context = evolution_context_factory()
    output = Path(context.prepared.spec.artifact_root) / "evolution" / "engine.log"
    output.parent.mkdir(parents=True)
    output.write_text("completed output\n")

    with pytest.raises(
        EvolutionWorkerError,
        match="pre-execution artifact set differs",
    ):
        load_evolution_worker(
            context.prepared.spec_path,
            expected_spec_sha256=context.prepared.spec_file_sha256,
        )

    loaded = load_evolution_worker(
        context.prepared.spec_path,
        expected_spec_sha256=context.prepared.spec_file_sha256,
        require_pristine_artifact_root=False,
    )

    assert loaded.manifest.attempt_id == context.prepared.spec.attempt_id


def test_generic_island_worker_is_bound_and_uses_pair_split(
    evolution_context_factory,
    tmp_path: Path,
):
    context = evolution_context_factory(generic_island=True)
    loaded = load_evolution_worker(
        context.prepared.spec_path,
        expected_spec_sha256=context.prepared.spec_file_sha256,
    )
    derived = derive_engine_config(loaded)
    state_store = AttemptStateStoreV2(tmp_path / "generic-island-state.sqlite3")
    queued = queue_prepared_evolution_attempt(
        state_store,
        manifest=loaded.manifest,
        prepared=context.prepared,
        bound_at=loaded.manifest.created_at + timedelta(seconds=2),
        validated_at=loaded.manifest.created_at + timedelta(seconds=3),
        queued_at=loaded.manifest.created_at + timedelta(seconds=4),
        working_directory=context.repo_root,
    )

    assert loaded.spec.worker_kind == "GENERIC_ISLAND_EVOLUTION"
    assert loaded.split_manifest.evolution_pairs == [
        context.config["pair_validation"]["training_pairs"][0]
    ]
    assert queued.worker_binding is not None
    assert (
        queued.worker_binding.worker_kind
        == WorkerKind.GENERIC_ISLAND_EVOLUTION
    )
    assert derived["checkpoint_provenance"]["engine_kind"] == "GENERIC_ISLAND"
    assert derived["checkpoint_provenance"]["island_names"] == [
        "island-a",
        "island-b",
    ]
    assert derived["genetic_algorithm"]["random_seed"] == loaded.manifest.seeds[0]
    assert [
        island["seed"]
        for island in derived["generic_island_model"]["islands"]
    ] == [
        loaded.manifest.seeds[0],
        loaded.manifest.seeds[0] + 1,
    ]
    assert [
        island["seed"]
        for island in context.config["generic_island_model"]["islands"]
    ] == [9001, 9002]


def test_generic_island_worker_delivers_strict_parent_seed(
    evolution_context_factory,
):
    seed_config = evolution_context_factory(generic_island=True).config
    source_seed = freeze_evolution_seed(
        _evaluated_individual(seed_config),
        resolved_config=seed_config,
        candidate_id="generic-island-parent",
    )
    context = evolution_context_factory(
        seeds=[source_seed],
        generic_island=True,
    )
    observed: list[Individual] = []

    def fake_island_evolution(config_path: Path, seeds):
        config = load_config(config_path)
        assert config["generic_island_model"]["enabled"] is True
        observed.extend(seeds)
        return [_evaluated_individual(context.config)]

    result = run_evolution_worker(
        context.prepared.spec_path,
        expected_spec_sha256=context.prepared.spec_file_sha256,
        evolution_runner=fake_island_evolution,
        backtester_factory=_failing_backtester_factory,
    )

    assert result.status == AttemptStatus.SUCCEEDED, result.error_detail
    assert len(observed) == 1
    assert observed[0].strategy_gene.to_dict() == source_seed.strategy_gene


def test_operational_runner_materializes_and_queues_without_child_registration(
    evolution_context_factory,
    tmp_path: Path,
):
    context = evolution_context_factory()
    config_path = tmp_path / "operational-safe-v2.yaml"
    config_path.write_text(yaml.safe_dump(context.config, sort_keys=True))
    state_path = tmp_path / "operational.sqlite3"

    prepared = prepare_and_queue_standard_attempt(
        config_path,
        experiment_name="operational-contract",
        state_path=state_path,
        artifact_base=tmp_path / "operational-attempts",
        created_at=datetime.now(UTC),
    )

    persisted = AttemptStateStoreV2(state_path).get(prepared.attempt_id)
    assert persisted.status == AttemptLifecycleStatus.QUEUED
    assert persisted.worker_binding is not None
    assert persisted.worker_binding.worker_kind == WorkerKind.STANDARD_EVOLUTION
    assert persisted.worker_binding.spec_sha256 == prepared.worker.spec_file_sha256
    assert "evolution_worker_v2" in persisted.worker_binding.argv[2]


def test_operational_runner_detaches_snapshot_from_mutable_source_before_queueing(
    evolution_context_factory,
    tmp_path: Path,
):
    context = evolution_context_factory()
    source_dir = tmp_path / "config" / "done"
    source_dir.mkdir(parents=True)
    config_path = source_dir / "historical-wave.yaml"
    config_path.write_text(yaml.safe_dump(context.config, sort_keys=True))
    state_path = tmp_path / "snapshot-state.sqlite3"

    prepared = prepare_and_queue_standard_attempt(
        config_path,
        experiment_name="snapshot-contract",
        state_path=state_path,
        artifact_base=tmp_path / "snapshot-attempts",
        created_at=datetime.now(UTC),
    )
    snapshot_path = (
        Path(prepared.bundle.manifest.artifact_root) / V2ArtifactStore.CONFIG_NAME
    )
    snapshot_bytes = snapshot_path.read_bytes()
    snapshot_file_hash = hashlib.sha256(snapshot_bytes).hexdigest()

    changed_source = yaml.safe_load(config_path.read_text())
    changed_source["genetic_algorithm"]["mutation_rate"] = 0.01
    config_path.write_text(yaml.safe_dump(changed_source, sort_keys=True))

    persisted = AttemptStateStoreV2(state_path).get(prepared.attempt_id)
    loaded = load_evolution_worker(
        prepared.worker.spec_path,
        expected_spec_sha256=persisted.worker_binding.spec_sha256,
    )

    assert persisted.status == AttemptLifecycleStatus.QUEUED
    assert snapshot_path.read_bytes() == snapshot_bytes
    assert loaded.resolved_config == context.config
    assert loaded.manifest.config_hash == canonical_config_hash(context.config)
    assert (
        loaded.spec.input_hashes[V2ArtifactStore.CONFIG_NAME] == snapshot_file_hash
    )
    assert canonical_config_hash(changed_source) != loaded.manifest.config_hash


def test_evolution_worker_revalidates_hash_consistent_config(evolution_context_factory):
    context = evolution_context_factory()
    spec_sha256 = _rewrite_with_hash_consistent_invalid_config(context.prepared)

    with pytest.raises(EvolutionWorkerError, match="violates the V2 contract") as exc_info:
        load_evolution_worker(
            context.prepared.spec_path,
            expected_spec_sha256=spec_sha256,
        )

    assert "genetic_algorithm.mutaton_rate" in str(exc_info.value.__cause__)


def test_evolution_search_is_not_accepted_until_strict_replay_commits(
    evolution_context_factory,
):
    context = evolution_context_factory()
    observed: dict[str, Any] = {}

    def fake_evolution(config_path: Path, seeds):
        engine_config = load_config(config_path)
        observed["config"] = engine_config
        observed["seeds"] = list(seeds)
        observed["cwd"] = Path.cwd()
        observed["ga_output_dir"] = os.environ.get("GA_OUTPUT_DIR")
        return [_evaluated_individual(context.config)]

    previous = os.environ.get("GA_OUTPUT_DIR")
    os.environ["GA_OUTPUT_DIR"] = "/tmp/must-not-control-v2-output"
    try:
        result = run_evolution_worker(
            context.prepared.spec_path,
            expected_spec_sha256=context.prepared.spec_file_sha256,
            evolution_runner=fake_evolution,
            backtester_factory=_failing_backtester_factory,
        )
    finally:
        if previous is None:
            os.environ.pop("GA_OUTPUT_DIR", None)
        else:
            os.environ["GA_OUTPUT_DIR"] = previous

    assert result.status == AttemptStatus.SUCCEEDED, result.error_detail
    assert result.candidate_evaluations[0].status == EvaluationStatus.FAIL
    assert observed["seeds"] == []
    layout = context.prepared.spec.output_layout
    assert observed["cwd"] == Path(layout.evolution_work_dir)
    assert observed["ga_output_dir"] is None
    candidate_id = result.candidate_evaluations[0].candidate_id
    root = Path(context.prepared.spec.artifact_root)
    seed_relative = f"candidates/{candidate_id}/evolution_seed.json"
    assert seed_relative in result.artifact_hashes
    assert (root / seed_relative).is_file()
    assert (root / "evolution" / "engine_config.yaml").is_file()
    engine_config = observed["config"]
    assert Path(engine_config["storage"]["checkpoint_dir"]).is_relative_to(root)
    assert Path(engine_config["storage"]["runs_dir"]).is_relative_to(root)
    assert Path(engine_config["storage"]["cache_dir"]).is_relative_to(root)
    assert Path(engine_config["storage"]["generated_strategy_dir"]).is_relative_to(root)
    layout.assert_engine_config(engine_config)
    assert V2ArtifactStore(root).read_verified_result() == result


def test_worker_spec_rejects_output_layout_outside_attempt(evolution_context_factory):
    context = evolution_context_factory()
    layout = context.prepared.spec.output_layout.model_copy(
        update={"diagnostics_dir": "/tmp/global-generation-stats"}
    )

    with pytest.raises(ValueError, match="diagnostics_dir"):
        context.prepared.spec.model_copy(
            update={"output_layout": layout},
        ).model_validate(
            {
                **context.prepared.spec.model_dump(mode="python"),
                "output_layout": layout.model_dump(mode="python"),
            }
        )


def _filesystem_snapshot(roots: list[Path]) -> dict[str, tuple[int, int]]:
    return {
        str(path): (path.stat().st_size, path.stat().st_mtime_ns)
        for root in roots
        if root.exists()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_real_mini_evolution_keeps_all_outputs_inside_attempt(
    evolution_context_factory,
):
    context = evolution_context_factory()
    legacy_roots = [
        context.repo_root / "genetic_algorithm" / "output",
        context.repo_root / "genetic_algorithm" / "data" / "runs",
        context.repo_root / "genetic_algorithm" / "data" / "checkpoints",
        context.repo_root / "genetic_algorithm" / "data" / "hall_of_fame",
        context.repo_root / "genetic_algorithm" / "data" / "cache",
        context.repo_root / "user_data" / "strategies" / "ga_generated",
        context.repo_root / "user_data" / "backtest_results",
    ]
    before = _filesystem_snapshot(legacy_roots)

    result = run_evolution_worker(
        context.prepared.spec_path,
        expected_spec_sha256=context.prepared.spec_file_sha256,
        backtester_factory=_failing_backtester_factory,
    )

    assert result.status == AttemptStatus.SUCCEEDED, result.error_detail
    assert _filesystem_snapshot(legacy_roots) == before
    root = Path(context.prepared.spec.artifact_root)
    assert (root / "evolution" / "diagnostics" / "generation_stats.csv").is_file()
    event_files = list((root / "evolution" / "runs").rglob("events.jsonl"))
    assert len(event_files) == 1
    events = [
        json.loads(line)
        for line in event_files[0].read_text().splitlines()
        if line.strip()
    ]
    assert sum(item.get("type") == "STARTED" for item in events) == 1
    assert (root / "evolution" / "hall_of_fame").is_dir()
    assert (root / "evolution" / "generated_strategies").is_dir()
    assert (root / "evolution" / "backtest_results").is_dir()
    assert (root / "evolution" / "freqtrade_user_data").is_dir()
    assert all((root / relative).is_file() for relative in result.artifact_hashes)


def test_real_mini_generic_island_evolution_uses_canonical_worker(
    evolution_context_factory,
):
    context = evolution_context_factory(generic_island=True)

    result = run_evolution_worker(
        context.prepared.spec_path,
        expected_spec_sha256=context.prepared.spec_file_sha256,
        backtester_factory=_failing_backtester_factory,
    )

    assert result.status == AttemptStatus.SUCCEEDED, result.error_detail
    root = Path(context.prepared.spec.artifact_root)
    assert list(
        (root / "evolution" / "checkpoints").glob("island_checkpoint_gen*.json")
    )
    runs_dir = root / "evolution" / "runs"
    assert runs_dir.is_dir()
    assert not list(runs_dir.rglob("events.jsonl"))
    assert result.candidate_evaluations
    assert all(
        evaluation.status == EvaluationStatus.FAIL
        for evaluation in result.candidate_evaluations
    )
    assert all((root / relative).is_file() for relative in result.artifact_hashes)


def test_strict_parent_seed_is_reproduced_and_delivered_to_engine(
    evolution_context_factory,
):
    seed_config = evolution_context_factory().config
    parent = _evaluated_individual(seed_config)
    source_seed = freeze_evolution_seed(
        parent,
        resolved_config=seed_config,
        candidate_id="parent-evolution-seed",
    )
    context = evolution_context_factory(seeds=[source_seed])
    observed: list[Individual] = []

    def fake_evolution(config_path: Path, seeds):
        assert config_path.is_file()
        observed.extend(seeds)
        return [_evaluated_individual(context.config)]

    result = run_evolution_worker(
        context.prepared.spec_path,
        expected_spec_sha256=context.prepared.spec_file_sha256,
        evolution_runner=fake_evolution,
        backtester_factory=_failing_backtester_factory,
    )

    assert result.status == AttemptStatus.SUCCEEDED
    assert len(observed) == 1
    assert observed[0].strategy_gene.to_dict() == source_seed.strategy_gene


def test_tampered_evolution_input_commits_verified_invalid_result(
    evolution_context_factory,
):
    context = evolution_context_factory()
    config_path = Path(context.prepared.spec.artifact_root) / V2ArtifactStore.CONFIG_NAME
    config_path.write_bytes(config_path.read_bytes() + b"\n")

    result = run_evolution_worker(
        context.prepared.spec_path,
        expected_spec_sha256=context.prepared.spec_file_sha256,
    )

    assert result.status == AttemptStatus.INVALID_RESULT
    assert result.error_code == "WORKER_INPUT_INVALID"
    assert "hash differs" in str(result.error_detail)


def test_engine_exception_cannot_create_a_successful_v2_result(evolution_context_factory):
    context = evolution_context_factory()

    def broken_engine(config_path: Path, seeds):
        raise RuntimeError("engine exploded")

    result = run_evolution_worker(
        context.prepared.spec_path,
        expected_spec_sha256=context.prepared.spec_file_sha256,
        evolution_runner=broken_engine,
    )

    assert result.status == AttemptStatus.INVALID_RESULT
    assert result.error_code == "WORKER_EXECUTION_FAILED"
    assert result.candidate_evaluations == []


def test_scheduler_queues_only_the_bound_standard_evolution_worker(
    evolution_context_factory,
    tmp_path: Path,
):
    context = evolution_context_factory()
    loaded = load_evolution_worker(
        context.prepared.spec_path,
        expected_spec_sha256=context.prepared.spec_file_sha256,
    )
    state_store = AttemptStateStoreV2(tmp_path / "evolution-state.sqlite3")
    config = ShadowSchedulerConfigV2(
        max_concurrent=1,
        lease_seconds=5,
        heartbeat_interval_seconds=0.1,
        poll_interval_seconds=0.02,
    )
    with EvolutionSchedulerV2(
        state_store,
        scheduler_id="evolution-scheduler-binding-test",
        config=config,
    ) as scheduler:
        queued = scheduler.queue_evolution(
            manifest=loaded.manifest,
            prepared=context.prepared,
            bound_at=loaded.manifest.created_at + timedelta(seconds=2),
            validated_at=loaded.manifest.created_at + timedelta(seconds=3),
            queued_at=loaded.manifest.created_at + timedelta(seconds=4),
            working_directory=context.repo_root,
        )

    assert queued.status == AttemptLifecycleStatus.QUEUED
    assert queued.worker_binding is not None
    assert queued.worker_binding.worker_kind == WorkerKind.STANDARD_EVOLUTION
    assert "evolution_worker_v2" in queued.worker_binding.argv[2]


def test_scheduler_executor_process_commits_tamper_as_invalid_result(
    evolution_context_factory,
    tmp_path: Path,
):
    context = evolution_context_factory()
    loaded = load_evolution_worker(
        context.prepared.spec_path,
        expected_spec_sha256=context.prepared.spec_file_sha256,
    )
    state_store = AttemptStateStoreV2(tmp_path / "evolution-process-state.sqlite3")
    config = ShadowSchedulerConfigV2(
        max_concurrent=1,
        lease_seconds=5,
        heartbeat_interval_seconds=0.1,
        poll_interval_seconds=0.02,
    )
    queue_prepared_evolution_attempt(
        state_store,
        manifest=loaded.manifest,
        prepared=context.prepared,
        bound_at=loaded.manifest.created_at + timedelta(seconds=2),
        validated_at=loaded.manifest.created_at + timedelta(seconds=3),
        queued_at=loaded.manifest.created_at + timedelta(seconds=4),
        working_directory=context.repo_root,
    )
    with AttemptSchedulerV2(
        state_store,
        scheduler_id="canonical-scheduler-process-test",
        config=config,
    ) as scheduler:
        input_path = Path(context.prepared.spec.artifact_root) / V2ArtifactStore.CONFIG_NAME
        input_path.write_bytes(input_path.read_bytes() + b"\n")

        ticks = scheduler.run_until_idle(timeout_seconds=20)

    terminal = state_store.get(loaded.manifest.attempt_id)
    assert terminal.status == AttemptLifecycleStatus.INVALID_RESULT
    assert terminal.result_status == AttemptStatus.INVALID_RESULT
    assert terminal.log_sha256 is not None
    assert any(loaded.manifest.attempt_id in tick.launched_attempt_ids for tick in ticks)
    result = V2ArtifactStore(loaded.manifest.artifact_root).read_verified_result()
    assert result.error_code == "WORKER_INPUT_INVALID"


def test_derived_engine_config_uses_manifest_seed_and_worker_count(
    evolution_context_factory,
):
    context = evolution_context_factory()
    loaded = load_evolution_worker(
        context.prepared.spec_path,
        expected_spec_sha256=context.prepared.spec_file_sha256,
    )

    derived = derive_engine_config(loaded)

    assert derived["genetic_algorithm"]["random_seed"] == loaded.manifest.seeds[0]
    assert derived["parallel_evaluation"] == context.config["parallel_evaluation"]
    assert derived["backtesting"]["timerange"] == (
        f"{int(datetime(2024, 1, 1, tzinfo=UTC).timestamp())}-"
        f"{int(datetime(2024, 1, 10, 23, tzinfo=UTC).timestamp())}"
    )
    assert derived["checkpoint_provenance"] == {
        "schema_version": "3.0",
        "engine_kind": "STANDARD",
        "config_hash": loaded.manifest.config_hash,
        "code_manifest_hash": canonical_config_hash(
            loaded.code_manifest.model_dump(mode="json")
        ),
        "data_manifest_hash": loaded.data_manifest.manifest_hash,
        "genome_schema_version": "strategy-gene-v2",
        "island_names": [],
    }
    assert context.config["genetic_algorithm"]["random_seed"] == 42


def test_nonzero_search_seed_salt_separates_search_from_paired_replay_seed(
    evolution_context_factory,
):
    context = evolution_context_factory(search_seed_salt=7)
    loaded = load_evolution_worker(
        context.prepared.spec_path,
        expected_spec_sha256=context.prepared.spec_file_sha256,
    )
    paired_seed = loaded.manifest.seeds[0]

    first = derive_engine_config(loaded)
    second = derive_engine_config(loaded)

    assert first == second
    assert first["genetic_algorithm"]["random_seed"] == derive_search_seed(
        paired_seed,
        7,
    )
    assert first["genetic_algorithm"]["random_seed"] != paired_seed
    assert loaded.manifest.seeds == [paired_seed]
