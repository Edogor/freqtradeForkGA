"""Focused contracts for the isolated hardcore dual-lane controller."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from genetic_algorithm.evaluation.raw_multipair_score import RawMultiPairPanel
from genetic_algorithm.orchestration.hardcore_campaign_v1 import (
    HARDCORE_PANEL_PAIRS,
    ArchiveNiche,
    ArchiveEntryV1,
    AttemptEvidenceV1,
    AttemptHandleV1,
    AttemptPollStatus,
    AttemptPollV1,
    CampaignLane,
    CampaignLifecycle,
    CandidateSnapshotV1,
    ControllerTickOutcome,
    ControllerTickV1,
    EvolutionStopReason,
    FailureClass,
    HardcoreCampaignControllerV1,
    HardcoreBootstrapArchiveV4,
    HardcoreQueueError,
    HardcoreCampaignPreflightV1,
    PairRole,
    RawPairMetricsV1,
    RuntimeResourcesV1,
    SearchRecipe,
    build_island_blueprints,
    build_hardcore_campaign_preflight,
    assign_candidate_niches,
    default_hardcore_campaign_policy,
    material_improvement,
    read_evolution_outcome,
    render_hardcore_systemd_user_unit,
    write_hardcore_bootstrap_archive,
)


NOW = datetime(2026, 8, 12, 8, 0, tzinfo=UTC)


def _ample_resources() -> RuntimeResourcesV1:
    return RuntimeResourcesV1(
        artifact_bytes=0,
        free_disk_bytes=100 * 1024**3,
        available_memory_bytes=16 * 1024**3,
    )


class FakeBackend:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.queued = []
        self.polls = {}
        self.stop_requests = []

    def queue(self, request):
        self.queued.append(request)
        handle = AttemptHandleV1(
            attempt_id=f"attempt-{len(self.queued)}",
            run_id=request.run_id,
            request_hash=request.request_hash,
            artifact_root=str(
                (self.root / request.run_id / f"attempt-{request.attempt_ordinal}").resolve()
            ),
        )
        self.polls[handle.attempt_id] = AttemptPollV1(
            status=AttemptPollStatus.RUNNING,
            observed_at=request.created_at,
            current_generation=1,
        )
        return handle

    def poll(self, handle):
        return self.polls[handle.attempt_id]

    def request_graceful_stop(self, handle, *, reason):
        self.stop_requests.append((handle.attempt_id, reason))

    def finish(self, request, evidence):
        handle = next(
            item
            for item in self._handles()
            if item.run_id == request.run_id and item.request_hash == request.request_hash
        )
        self.polls[handle.attempt_id] = AttemptPollV1(
            status=AttemptPollStatus.FINISHED,
            observed_at=evidence.finished_at,
            current_generation=evidence.actual_generations,
            current_score=evidence.candidates[0].score if evidence.candidates else None,
            plateau_checks=evidence.plateau_checks,
            evidence=evidence,
        )

    def _handles(self):
        for index, request in enumerate(self.queued, start=1):
            yield AttemptHandleV1(
                attempt_id=f"attempt-{index}",
                run_id=request.run_id,
                request_hash=request.request_hash,
                artifact_root=str(
                    (
                        self.root
                        / request.run_id
                        / f"attempt-{request.attempt_ordinal}"
                    ).resolve()
                ),
            )


def _policy(tmp_path: Path):
    config_15m = tmp_path / "15m.yaml"
    config_1h = tmp_path / "1h.yaml"
    config_15m.write_text(
        "config_schema_version: 2\npreset: hardcore_multipair_15m_v1\n",
        encoding="utf-8",
    )
    config_1h.write_text(
        "config_schema_version: 2\npreset: hardcore_multipair_1h_v1\n",
        encoding="utf-8",
    )
    return default_hardcore_campaign_policy(
        campaign_id="campaign-test",
        automation_root=tmp_path / "hardcore",
        config_15m=config_15m,
        config_1h=config_1h,
    )


def _pair_metrics() -> list[RawPairMetricsV1]:
    return [
        RawPairMetricsV1(
            pair=pair,
            role=(
                PairRole.DEVELOPMENT
                if pair in HARDCORE_PANEL_PAIRS[:3]
                else PairRole.VALIDATION
            ),
            net_return=0.01,
            net_expectancy=0.001,
            profit_factor=1.1,
            profit_factor_censored=False,
            trade_count=120,
            active_months=24,
            calendar_months=34.5,
            median_holding_hours=8.0,
            p90_holding_hours=36.0,
            max_drawdown=0.10,
            max_drawdown_duration_days=80.0,
            max_consecutive_losses=5,
            normalized_components={
                "R": 0.1,
                "E": 0.2,
                "P": 0.2,
                "A": 0.5,
                "Q": 0.15,
                "F": 0.075,
                "H": 1.0,
                "D": 0.1,
                "U": 0.2,
                "L": 0.2,
                "O": 0.0,
            },
        )
        for pair in HARDCORE_PANEL_PAIRS
    ]


def _candidate(
    tmp_path: Path,
    *,
    lane: CampaignLane,
    score: float,
    ordinal: int = 1,
) -> CandidateSnapshotV1:
    seed_path = (tmp_path / f"seed-{lane.value}-{ordinal}.json").resolve()
    seed_path.write_text("{}\n", encoding="utf-8")
    token = f"{lane.value}-{ordinal}".encode()
    import hashlib

    phenotype = hashlib.sha256(b"phenotype-" + token).hexdigest()
    gene_hash = hashlib.sha256(b"gene-" + token).hexdigest()
    seed_hash = hashlib.sha256(seed_path.read_bytes()).hexdigest()
    return CandidateSnapshotV1(
        candidate_id=f"candidate-{lane.value}-{ordinal}",
        phenotype_hash=phenotype,
        score=score,
        timeframe=lane,
        panel_id=RawMultiPairPanel(timeframe=lane.value).panel_id,
        policy_hash=RawMultiPairPanel(timeframe=lane.value).policy.policy_hash,
        edge_score=0.15,
        activity_score=0.5,
        productive_frequency_score=0.075,
        evolution_seed_path=str(seed_path),
        evolution_seed_sha256=seed_hash,
        gene_hash=gene_hash,
        pair_metrics=_pair_metrics(),
    )


def _evidence(
    request,
    tmp_path: Path,
    *,
    score: float | None = 1.0,
    stop_reason: EvolutionStopReason = EvolutionStopReason.COMPLETED_BUDGET,
    failure_class: FailureClass = FailureClass.NONE,
    error_code: str | None = None,
    checkpoint: bool = False,
):
    checkpoint_path = None
    checkpoint_hash = None
    if checkpoint:
        checkpoint_file = (tmp_path / f"{request.run_id}-checkpoint.json").resolve()
        checkpoint_file.write_text("{}\n", encoding="utf-8")
        import hashlib

        checkpoint_path = str(checkpoint_file)
        checkpoint_hash = hashlib.sha256(checkpoint_file.read_bytes()).hexdigest()
    candidates = (
        []
        if score is None
        else [_candidate(tmp_path, lane=request.lane, score=score)]
    )
    return AttemptEvidenceV1(
        campaign_id=request.campaign_id,
        run_id=request.run_id,
        request_hash=request.request_hash,
        lane=request.lane,
        recipe=request.recipe,
        attempt_ordinal=request.attempt_ordinal,
        started_at=request.created_at,
        finished_at=request.created_at + timedelta(minutes=5),
        stop_reason=stop_reason,
        actual_generations=0 if score is None else 12,
        score_curve=[] if score is None else [score] * 12,
        valid_evaluations=0 if score is None else 72,
        failed_evaluations=1 if score is None else 0,
        plateau_checks=0,
        candidates=candidates,
        failure_class=failure_class,
        error_code=error_code,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_hash,
    )


def _finish_active(controller, backend, tmp_path, *, score):
    request = controller.state.active_run.request
    evidence = _evidence(request, tmp_path, score=score)
    backend.finish(request, evidence)
    return controller.run_once(observed_at=evidence.finished_at)


def test_material_improvement_is_continuous_and_allows_first_negative():
    assert material_improvement(None, -5.0) is True
    assert material_improvement(-5.0, -4.75) is True
    assert material_improvement(-5.0, -4.751) is False
    assert material_improvement(1000.0, 1004.99) is False
    assert material_improvement(1000.0, 1005.0) is True


def test_blueprints_are_twelve_specialists_with_only_three_archive_islands(
    tmp_path: Path,
):
    niches = (
        [ArchiveNiche.EDGE] * 4
        + [ArchiveNiche.ACTIVITY] * 4
        + [ArchiveNiche.BALANCED] * 2
    )
    archive = [
        ArchiveEntryV1(
            candidate=_candidate(
                tmp_path, lane=CampaignLane.FIFTEEN_MINUTES, score=10 - index, ordinal=index
            ),
            niche=niches[index - 1],
            descriptor_score=float(10 - index),
            source_run_id=f"run-{index}",
            archived_at=NOW,
        )
        for index in range(1, 11)
    ]

    islands = build_island_blueprints(
        lane=CampaignLane.FIFTEEN_MINUTES,
        search_seed=100,
        archive=archive,
    )

    assert len(islands) == 12
    assert len({island.seed for island in islands}) == 12
    assert sum(island.archive_island for island in islands) == 3
    assert all(not island.seed_candidates for island in islands[:9])
    assert [len(island.seed_candidates) for island in islands[9:]] == [4, 4, 4]
    assert {island.family for island in islands} == {
        "momentum",
        "trend",
        "volatility-volume",
        "bridge",
    }


def test_partial_archive_preserves_all_three_niches_and_balanced_champion(
    tmp_path: Path,
):
    candidates = [
        _candidate(
            tmp_path,
            lane=CampaignLane.FIFTEEN_MINUTES,
            score=float(10 - index),
            ordinal=index,
        )
        for index in range(1, 8)
    ]

    assignments = assign_candidate_niches(candidates)

    assert len(assignments) == 7
    assert {niche for niche, _candidate, _descriptor in assignments} == set(
        ArchiveNiche
    )
    balanced = [candidate for niche, candidate, _ in assignments if niche is ArchiveNiche.BALANCED]
    assert max(candidates, key=lambda item: item.score) in balanced


def test_search_recipes_materially_change_explicit_indicator_pools():
    balanced = build_island_blueprints(
        lane=CampaignLane.FIFTEEN_MINUTES,
        search_seed=777,
        archive=[],
        recipe=SearchRecipe.BALANCED,
    )
    explore = build_island_blueprints(
        lane=CampaignLane.FIFTEEN_MINUTES,
        search_seed=777,
        archive=[],
        recipe=SearchRecipe.EXPLORE,
    )
    recombine = build_island_blueprints(
        lane=CampaignLane.FIFTEEN_MINUTES,
        search_seed=777,
        archive=[],
        recipe=SearchRecipe.RECOMBINE,
    )

    assert [item.indicator_pool for item in balanced] != [
        item.indicator_pool for item in explore
    ]
    assert [item.indicator_pool for item in explore] != [
        item.indicator_pool for item in recombine
    ]
    assert len(explore[0].indicator_pool) < len(balanced[0].indicator_pool)
    assert len(balanced[0].indicator_pool) < len(recombine[0].indicator_pool)


def test_controller_strictly_alternates_and_keeps_lane_archives_separate(
    tmp_path: Path,
):
    policy = _policy(tmp_path)
    backend = FakeBackend(tmp_path / "attempts")
    controller = HardcoreCampaignControllerV1(
        policy=policy,
        backend=backend,
        started_at=NOW,
        resource_probe=_ample_resources,
    )

    first = controller.run_once(observed_at=NOW)
    assert first.active_lane == CampaignLane.FIFTEEN_MINUTES
    _finish_active(controller, backend, tmp_path, score=-2.0)
    second = controller.run_once(observed_at=NOW + timedelta(minutes=6))
    assert second.active_lane == CampaignLane.ONE_HOUR
    _finish_active(controller, backend, tmp_path, score=3.0)
    third = controller.run_once(observed_at=NOW + timedelta(minutes=12))

    assert third.active_lane == CampaignLane.FIFTEEN_MINUTES
    assert [request.lane for request in backend.queued] == [
        CampaignLane.FIFTEEN_MINUTES,
        CampaignLane.ONE_HOUR,
        CampaignLane.FIFTEEN_MINUTES,
    ]
    lane_15m = controller.state.lanes[CampaignLane.FIFTEEN_MINUTES]
    lane_1h = controller.state.lanes[CampaignLane.ONE_HOUR]
    assert lane_15m.global_champion_score == -2.0
    assert lane_1h.global_champion_score == 3.0
    assert all(
        item.candidate.timeframe == CampaignLane.FIFTEEN_MINUTES
        for item in lane_15m.archive
    )
    assert all(
        item.candidate.timeframe == CampaignLane.ONE_HOUR for item in lane_1h.archive
    )


def test_recipe_rotates_without_progress_and_resets_after_improvement(tmp_path: Path):
    policy = _policy(tmp_path)
    backend = FakeBackend(tmp_path / "attempts")
    controller = HardcoreCampaignControllerV1(
        policy=policy,
        backend=backend,
        started_at=NOW,
        resource_probe=_ample_resources,
    )
    controller.run_once(observed_at=NOW)
    _finish_active(controller, backend, tmp_path, score=-2.0)
    assert controller.state.lanes[CampaignLane.FIFTEEN_MINUTES].recipe == (
        SearchRecipe.BALANCED
    )
    controller.run_once(observed_at=NOW + timedelta(minutes=6))
    _finish_active(controller, backend, tmp_path, score=1.0)
    controller.run_once(observed_at=NOW + timedelta(minutes=12))
    _finish_active(controller, backend, tmp_path, score=-1.8)
    assert controller.state.lanes[CampaignLane.FIFTEEN_MINUTES].recipe == (
        SearchRecipe.EXPLORE
    )
    assert controller.state.lanes[CampaignLane.FIFTEEN_MINUTES].global_champion_score == -2.0

    controller.run_once(observed_at=NOW + timedelta(minutes=18))
    _finish_active(controller, backend, tmp_path, score=1.1)
    controller.run_once(observed_at=NOW + timedelta(minutes=24))
    request = controller.state.active_run.request
    assert request.recipe == SearchRecipe.EXPLORE
    _finish_active(controller, backend, tmp_path, score=-1.7)
    assert controller.state.lanes[CampaignLane.FIFTEEN_MINUTES].recipe == (
        SearchRecipe.BALANCED
    )



def test_new_campaign_loads_hash_covered_niche_bootstrap_and_starts_recombine(
    tmp_path: Path,
):
    root = tmp_path / "hardcore"
    bootstrap_path = root / "bootstrap" / "bootstrap_archive_v4.json"
    lanes = {}
    for lane in CampaignLane:
        candidate = _candidate(tmp_path, lane=lane, score=3.0)
        lanes[lane] = [
            ArchiveEntryV1(
                candidate=candidate,
                niche=ArchiveNiche.BALANCED,
                descriptor_score=candidate.score,
                source_run_id=f"historical-{lane.value}",
                archived_at=NOW,
            )
        ]
    bootstrap = HardcoreBootstrapArchiveV4(
        created_at=NOW,
        source_root=str(tmp_path / "v3"),
        lanes=lanes,
    )
    digest = write_hardcore_bootstrap_archive(bootstrap_path, bootstrap)
    base = _policy(tmp_path)
    policy = base.model_copy(
        update={
            "automation_root": str(root.resolve()),
            "kill_switch_path": str((root / "STOP_HARDCORE_CAMPAIGN").resolve()),
            "bootstrap_archive_path": str(bootstrap_path.resolve()),
            "bootstrap_archive_sha256": digest,
        }
    )
    policy = type(base).model_validate(policy.model_dump(mode="python"))
    controller = HardcoreCampaignControllerV1(
        policy=policy,
        backend=FakeBackend(tmp_path / "attempts"),
        started_at=NOW,
        resource_probe=_ample_resources,
    )

    for lane in CampaignLane:
        state = controller.state.lanes[lane]
        assert state.recipe is SearchRecipe.RECOMBINE
        assert state.global_champion_score == 3.0
        assert len(state.archive) == 1
        assert state.archive[0].candidate.timeframe is lane


def test_niche_archive_can_update_without_material_global_progress(tmp_path: Path):
    backend = FakeBackend(tmp_path / "attempts")
    controller = HardcoreCampaignControllerV1(
        policy=_policy(tmp_path),
        backend=backend,
        started_at=NOW,
        resource_probe=_ample_resources,
    )
    controller.run_once(observed_at=NOW)
    _finish_active(controller, backend, tmp_path, score=1.0)
    controller.run_once(observed_at=NOW + timedelta(minutes=6))
    _finish_active(controller, backend, tmp_path, score=2.0)
    controller.run_once(observed_at=NOW + timedelta(minutes=12))
    _finish_active(controller, backend, tmp_path, score=1.1)

    lane = controller.state.lanes[CampaignLane.FIFTEEN_MINUTES]
    assert lane.global_champion_score == 1.0
    outcome = read_evolution_outcome(controller.state.outcomes[-1].outcome_path)
    assert outcome.archive_change.material_improvement is False
    assert outcome.archive_change.niche_archive_changed is True
    assert outcome.archive_change.updated_phenotype_hashes


def test_transient_error_retries_once_from_checkpoint_without_new_run(
    tmp_path: Path,
):
    policy = _policy(tmp_path)
    backend = FakeBackend(tmp_path / "attempts")
    controller = HardcoreCampaignControllerV1(
        policy=policy,
        backend=backend,
        started_at=NOW,
        resource_probe=_ample_resources,
    )
    controller.run_once(observed_at=NOW)
    initial = controller.state.active_run.request
    failure = _evidence(
        initial,
        tmp_path,
        score=None,
        stop_reason=EvolutionStopReason.TECHNICAL_INVALID,
        failure_class=FailureClass.TRANSIENT,
        error_code="WORKER_LOST",
        checkpoint=True,
    )
    backend.finish(initial, failure)

    tick = controller.run_once(observed_at=failure.finished_at)

    retry = controller.state.active_run.request
    assert tick.reason_codes == ["TRANSIENT_RETRY_QUEUED"]
    assert retry.run_id == initial.run_id
    assert retry.attempt_ordinal == 1
    assert retry.resume_checkpoint_path == failure.checkpoint_path
    assert controller.state.run_sequence == 1
    assert controller.state.lanes[CampaignLane.FIFTEEN_MINUTES].runs_started == 1


def test_retry_queue_failure_is_finalized_and_can_suspend_lane(tmp_path: Path):
    class RetryQueueFailureBackend(FakeBackend):
        def queue(self, request):
            if request.attempt_ordinal == 1:
                raise HardcoreQueueError(
                    "retry config became invalid",
                    failure_class=FailureClass.DETERMINISTIC_CONFIG,
                    error_code="QUEUE_CONFIG_INVALID",
                )
            return super().queue(request)

    policy = _policy(tmp_path)
    backend = RetryQueueFailureBackend(tmp_path / "attempts")
    controller = HardcoreCampaignControllerV1(
        policy=policy,
        backend=backend,
        started_at=NOW,
        resource_probe=_ample_resources,
    )
    controller.run_once(observed_at=NOW)
    initial = controller.state.active_run.request
    failure = _evidence(
        initial,
        tmp_path,
        score=None,
        stop_reason=EvolutionStopReason.TECHNICAL_INVALID,
        failure_class=FailureClass.TRANSIENT,
        error_code="WORKER_LOST",
        checkpoint=True,
    )
    backend.finish(initial, failure)

    tick = controller.run_once(observed_at=failure.finished_at)

    lane = controller.state.lanes[CampaignLane.FIFTEEN_MINUTES]
    assert tick.reason_codes == ["RUN_FINALIZED"]
    assert controller.state.active_run is None
    assert lane.lifecycle.value == "SUSPENDED"
    outcome = read_evolution_outcome(
        Path(policy.automation_root)
        / "runs"
        / initial.run_id
        / "evolution_outcome.json"
    )
    assert outcome.evidence.error_code == "RETRY_QUEUE_CONFIG_INVALID"


def test_deterministic_data_error_suspends_only_affected_lane(tmp_path: Path):
    policy = _policy(tmp_path)
    backend = FakeBackend(tmp_path / "attempts")
    controller = HardcoreCampaignControllerV1(
        policy=policy,
        backend=backend,
        started_at=NOW,
        resource_probe=_ample_resources,
    )
    controller.run_once(observed_at=NOW)
    request = controller.state.active_run.request
    failure = _evidence(
        request,
        tmp_path,
        score=None,
        stop_reason=EvolutionStopReason.TECHNICAL_INVALID,
        failure_class=FailureClass.DETERMINISTIC_DATA,
        error_code="DATA_MANIFEST_MISMATCH",
    )
    backend.finish(request, failure)
    controller.run_once(observed_at=failure.finished_at)
    next_tick = controller.run_once(observed_at=failure.finished_at + timedelta(seconds=1))

    assert controller.state.lanes[CampaignLane.FIFTEEN_MINUTES].lifecycle.value == (
        "SUSPENDED"
    )
    assert next_tick.active_lane == CampaignLane.ONE_HOUR
    outcome = read_evolution_outcome(
        Path(policy.automation_root) / "runs" / request.run_id / "evolution_outcome.json"
    )
    assert outcome.evidence.failure_class == FailureClass.DETERMINISTIC_DATA

    # The surviving lane may continue indefinitely; strict count balance is
    # required only while both lanes are healthy.
    _finish_active(controller, backend, tmp_path, score=1.0)
    solo_tick = controller.run_once(
        observed_at=failure.finished_at + timedelta(minutes=7)
    )
    assert solo_tick.active_lane == CampaignLane.ONE_HOUR
    assert controller.state.lanes[CampaignLane.ONE_HOUR].runs_started == 2
    assert controller.state.lanes[CampaignLane.FIFTEEN_MINUTES].runs_started == 1


def test_prequeue_config_error_suspends_lane_and_other_lane_continues(
    tmp_path: Path,
):
    policy = _policy(tmp_path)
    Path(policy.lane_configs[CampaignLane.FIFTEEN_MINUTES]).unlink()
    backend = FakeBackend(tmp_path / "attempts")
    controller = HardcoreCampaignControllerV1(
        policy=policy,
        backend=backend,
        started_at=NOW,
        resource_probe=_ample_resources,
    )

    suspended = controller.run_once(observed_at=NOW)
    next_tick = controller.run_once(observed_at=NOW + timedelta(seconds=1))

    assert suspended.reason_codes[0] == "LANE_SUSPENDED_PREQUEUE"
    assert controller.state.lanes[CampaignLane.FIFTEEN_MINUTES].lifecycle.value == (
        "SUSPENDED"
    )
    assert next_tick.active_lane == CampaignLane.ONE_HOUR
    assert [request.lane for request in backend.queued] == [CampaignLane.ONE_HOUR]


def test_canary_request_preserves_reduced_four_by_one_shape(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[2]
    policy = default_hardcore_campaign_policy(
        campaign_id="canary-test",
        automation_root=tmp_path / "canary",
        config_15m=repo_root
        / "genetic_algorithm/config/presets/hardcore_multipair_canary_15m_v1.yaml",
        config_1h=repo_root
        / "genetic_algorithm/config/presets/hardcore_multipair_canary_1h_v1.yaml",
        canary=True,
    )
    backend = FakeBackend(tmp_path / "attempts")
    controller = HardcoreCampaignControllerV1(
        policy=policy,
        backend=backend,
        started_at=NOW,
        resource_probe=_ample_resources,
    )

    controller.run_once(observed_at=NOW)
    request = controller.state.active_run.request

    assert len(request.islands) == 12
    assert {(item.population_size, item.generations) for item in request.islands} == {
        (4, 1)
    }


def test_canary_finishes_after_exactly_one_15m_and_one_1h_run(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[2]
    policy = default_hardcore_campaign_policy(
        campaign_id="dual-lane-canary",
        automation_root=tmp_path / "canary",
        config_15m=repo_root
        / "genetic_algorithm/config/presets/hardcore_multipair_canary_15m_v1.yaml",
        config_1h=repo_root
        / "genetic_algorithm/config/presets/hardcore_multipair_canary_1h_v1.yaml",
        canary=True,
    )
    backend = FakeBackend(tmp_path / "attempts")
    controller = HardcoreCampaignControllerV1(
        policy=policy,
        backend=backend,
        started_at=NOW,
        resource_probe=_ample_resources,
    )

    controller.run_once(observed_at=NOW)
    _finish_active(controller, backend, tmp_path, score=-2.0)
    second = controller.run_once(observed_at=NOW + timedelta(minutes=6))
    assert second.active_lane == CampaignLane.ONE_HOUR
    terminal = _finish_active(controller, backend, tmp_path, score=-1.0)

    assert terminal.lifecycle == CampaignLifecycle.COMPLETED
    assert terminal.reason_codes == ["CANARY_COMPLETED"]
    assert controller.state.stop_reason == "CANARY_COMPLETED"
    assert [item.lane for item in backend.queued] == [
        CampaignLane.FIFTEEN_MINUTES,
        CampaignLane.ONE_HOUR,
    ]
    terminal_refresh_time = NOW + timedelta(minutes=20)
    controller.run_once(observed_at=terminal_refresh_time)
    assert len(backend.queued) == 2
    refreshed_status = json.loads(
        (Path(policy.automation_root) / "campaign_status.json").read_text(
            encoding="utf-8"
        )
    )
    assert refreshed_status["lifecycle"] == CampaignLifecycle.COMPLETED.value
    assert refreshed_status["stop_reason"] == "CANARY_COMPLETED"
    assert refreshed_status["active_run_id"] is None
    assert refreshed_status["observed_at"] == terminal_refresh_time.isoformat().replace(
        "+00:00", "Z"
    )


@pytest.mark.parametrize("technical_lane", [
    CampaignLane.FIFTEEN_MINUTES,
    CampaignLane.ONE_HOUR,
])
def test_canary_blocks_when_either_lane_is_technical(
    tmp_path: Path, technical_lane: CampaignLane
):
    repo_root = Path(__file__).resolve().parents[2]
    policy = default_hardcore_campaign_policy(
        campaign_id=f"technical-{technical_lane.value}",
        automation_root=tmp_path / "canary",
        config_15m=repo_root
        / "genetic_algorithm/config/presets/hardcore_multipair_canary_15m_v1.yaml",
        config_1h=repo_root
        / "genetic_algorithm/config/presets/hardcore_multipair_canary_1h_v1.yaml",
        canary=True,
    )
    backend = FakeBackend(tmp_path / "attempts")
    controller = HardcoreCampaignControllerV1(
        policy=policy,
        backend=backend,
        started_at=NOW,
        resource_probe=_ample_resources,
    )

    terminal = None
    for ordinal in range(2):
        controller.run_once(observed_at=NOW + timedelta(minutes=ordinal * 10))
        request = controller.state.active_run.request
        if request.lane == technical_lane:
            evidence = _evidence(
                request,
                tmp_path,
                score=None,
                stop_reason=EvolutionStopReason.TECHNICAL_INVALID,
                failure_class=FailureClass.TRANSIENT,
                error_code="CANARY_WORKER_FAILED",
            )
        else:
            evidence = _evidence(request, tmp_path, score=-1.0)
        backend.finish(request, evidence)
        terminal = controller.run_once(observed_at=evidence.finished_at)

    assert terminal.lifecycle == CampaignLifecycle.BLOCKED
    assert terminal.outcome == ControllerTickOutcome.BLOCKED
    assert terminal.reason_codes == ["CANARY_FAILED_TECHNICAL_OUTCOME"]


def test_canary_blocks_after_two_queue_failures(tmp_path: Path):
    class QueueFailureBackend(FakeBackend):
        def queue(self, request):
            raise HardcoreQueueError(
                "synthetic deterministic queue failure",
                failure_class=FailureClass.DETERMINISTIC_CONFIG,
                error_code="QUEUE_CONFIG_INVALID",
            )

    repo_root = Path(__file__).resolve().parents[2]
    policy = default_hardcore_campaign_policy(
        campaign_id="queue-failure-canary",
        automation_root=tmp_path / "canary",
        config_15m=repo_root
        / "genetic_algorithm/config/presets/hardcore_multipair_canary_15m_v1.yaml",
        config_1h=repo_root
        / "genetic_algorithm/config/presets/hardcore_multipair_canary_1h_v1.yaml",
        canary=True,
    )
    controller = HardcoreCampaignControllerV1(
        policy=policy,
        backend=QueueFailureBackend(tmp_path / "attempts"),
        started_at=NOW,
        resource_probe=_ample_resources,
    )

    first = controller.run_once(observed_at=NOW)
    terminal = controller.run_once(observed_at=NOW + timedelta(minutes=1))

    assert first.outcome == ControllerTickOutcome.PROGRESSED
    assert terminal.outcome == ControllerTickOutcome.BLOCKED
    assert terminal.reason_codes == ["CANARY_FAILED_TECHNICAL_OUTCOME"]


def test_budget_completion_without_material_progress_is_explicit(tmp_path: Path):
    policy = _policy(tmp_path)
    backend = FakeBackend(tmp_path / "attempts")
    controller = HardcoreCampaignControllerV1(
        policy=policy,
        backend=backend,
        started_at=NOW,
        resource_probe=_ample_resources,
    )
    controller.run_once(observed_at=NOW)
    _finish_active(controller, backend, tmp_path, score=1.0)
    controller.run_once(observed_at=NOW + timedelta(minutes=6))
    _finish_active(controller, backend, tmp_path, score=2.0)
    controller.run_once(observed_at=NOW + timedelta(minutes=12))
    request = controller.state.active_run.request
    _finish_active(controller, backend, tmp_path, score=1.1)

    outcome = read_evolution_outcome(
        Path(policy.automation_root) / "runs" / request.run_id / "evolution_outcome.json"
    )
    assert outcome.evidence.stop_reason == EvolutionStopReason.NO_GLOBAL_PROGRESS
    assert outcome.archive_change.material_improvement is False


def test_active_resource_limit_requests_generation_boundary_drain(tmp_path: Path):
    policy = _policy(tmp_path)
    backend = FakeBackend(tmp_path / "attempts")
    probe = {"value": _ample_resources()}
    controller = HardcoreCampaignControllerV1(
        policy=policy,
        backend=backend,
        started_at=NOW,
        resource_probe=lambda: probe["value"],
    )
    controller.run_once(observed_at=NOW)
    request = controller.state.active_run.request
    probe["value"] = RuntimeResourcesV1(
        artifact_bytes=0,
        free_disk_bytes=10 * 1024**3,
        available_memory_bytes=16 * 1024**3,
    )

    draining = controller.run_once(observed_at=NOW + timedelta(minutes=1))

    assert draining.lifecycle == CampaignLifecycle.RESOURCE_DRAINING
    assert controller.state.stop_reason == "MIN_FREE_DISK_REACHED"
    assert backend.stop_requests == [
        (
            controller.state.active_run.handle.attempt_id,
            EvolutionStopReason.CAMPAIGN_DEADLINE,
        )
    ]
    evidence = _evidence(
        request,
        tmp_path,
        score=1.0,
        stop_reason=EvolutionStopReason.CAMPAIGN_DEADLINE,
    )
    backend.finish(request, evidence)
    terminal = controller.run_once(observed_at=evidence.finished_at)
    assert terminal.reason_codes == ["MIN_FREE_DISK_REACHED"]


def test_status_reports_other_lane_as_next_during_active_run(tmp_path: Path):
    policy = _policy(tmp_path)
    backend = FakeBackend(tmp_path / "attempts")
    controller = HardcoreCampaignControllerV1(
        policy=policy,
        backend=backend,
        started_at=NOW,
        resource_probe=_ample_resources,
    )
    controller.run_once(observed_at=NOW)

    status = controller.status(observed_at=NOW)

    assert status.active_lane == CampaignLane.FIFTEEN_MINUTES
    assert status.next_timeframe == CampaignLane.ONE_HOUR
    assert set(status.lane_champion_scores) == set(CampaignLane)
    assert set(status.lane_champion_pair_metrics) == set(CampaignLane)


def test_deadline_drains_active_attempt_then_writes_search_only_final_report(
    tmp_path: Path,
):
    policy = _policy(tmp_path)
    backend = FakeBackend(tmp_path / "attempts")
    controller = HardcoreCampaignControllerV1(
        policy=policy,
        backend=backend,
        started_at=NOW,
        resource_probe=_ample_resources,
    )
    controller.run_once(observed_at=NOW)
    request = controller.state.active_run.request

    deadline_tick = controller.run_once(observed_at=NOW + timedelta(days=7))
    assert deadline_tick.lifecycle == CampaignLifecycle.DEADLINE_DRAINING
    assert backend.stop_requests == [
        (controller.state.active_run.handle.attempt_id, EvolutionStopReason.CAMPAIGN_DEADLINE)
    ]

    evidence = _evidence(
        request,
        tmp_path,
        score=2.0,
        stop_reason=EvolutionStopReason.CAMPAIGN_DEADLINE,
    ).model_copy(
        update={"finished_at": NOW + timedelta(days=7, minutes=1)}
    )
    backend.finish(request, evidence)
    final_tick = controller.run_once(
        observed_at=NOW + timedelta(days=7, minutes=1)
    )

    assert final_tick.lifecycle == CampaignLifecycle.COMPLETED
    report = controller.final_report()
    assert report.live_ready is False
    assert report.lanes[CampaignLane.FIFTEEN_MINUTES].champion_label == (
        "SEARCH_CHAMPION"
    )
    assert not report.lanes[CampaignLane.ONE_HOUR].champion
    assert (Path(policy.automation_root) / "campaign_final_report.json.sha256").is_file()


def test_outcome_checksum_detects_tampering(tmp_path: Path):
    policy = _policy(tmp_path)
    backend = FakeBackend(tmp_path / "attempts")
    controller = HardcoreCampaignControllerV1(
        policy=policy,
        backend=backend,
        started_at=NOW,
        resource_probe=_ample_resources,
    )
    controller.run_once(observed_at=NOW)
    request = controller.state.active_run.request
    _finish_active(controller, backend, tmp_path, score=1.0)
    path = Path(policy.automation_root) / "runs" / request.run_id / "evolution_outcome.json"
    path.write_bytes(path.read_bytes() + b" ")

    with pytest.raises(Exception, match="checksum mismatch"):
        read_evolution_outcome(path)


def test_systemd_template_uses_isolated_command_and_crash_restart(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[2]
    config_15m = tmp_path / "15m.yaml"
    config_1h = tmp_path / "1h.yaml"
    config_15m.write_text("{}\n", encoding="utf-8")
    config_1h.write_text("{}\n", encoding="utf-8")

    unit = render_hardcore_systemd_user_unit(
        repo_root=repo_root,
        automation_root=tmp_path / "hardcore",
        config_15m=config_15m,
        config_1h=config_1h,
        campaign_id="production-20260812",
    )

    assert "hardcore-campaign start" in unit
    assert "Restart=on-failure" in unit
    assert "SuccessExitStatus=2" in unit
    assert "TimeoutStopSec=infinity" in unit
    assert "KillMode=mixed" in unit


def test_preflight_requires_clean_pushed_commit_and_isolated_state(
    tmp_path: Path, monkeypatch
):
    repo_root = Path(__file__).resolve().parents[2]
    policy = default_hardcore_campaign_policy(
        campaign_id="preflight-test",
        automation_root=tmp_path / "hardcore",
        config_15m=repo_root
        / "genetic_algorithm/config/presets/hardcore_multipair_15m_v1.yaml",
        config_1h=repo_root
        / "genetic_algorithm/config/presets/hardcore_multipair_1h_v1.yaml",
    )
    head = "a" * 40

    def clean_git(_repo, *args):
        command = " ".join(args)
        if command.startswith("status"):
            return 0, ""
        if command == "rev-parse HEAD":
            return 0, head
        if "--symbolic-full-name" in command:
            return 0, "origin/feature"
        if command == "rev-parse @{upstream}":
            return 0, head
        raise AssertionError(command)

    monkeypatch.setattr(
        "genetic_algorithm.orchestration.hardcore_campaign_v1._git_output",
        clean_git,
    )
    report = build_hardcore_campaign_preflight(
        policy,
        repo_root=repo_root,
        state_path=Path(policy.automation_root) / "hardcore_state.sqlite3",
        checked_at=NOW,
    )
    assert report.ready is True
    assert report.reason_codes == ["PREFLIGHT_READY"]

    def dirty_git(repo, *args):
        code, value = clean_git(repo, *args)
        return (code, " M file.py") if args[0] == "status" else (code, value)

    monkeypatch.setattr(
        "genetic_algorithm.orchestration.hardcore_campaign_v1._git_output",
        dirty_git,
    )
    blocked = build_hardcore_campaign_preflight(
        policy,
        repo_root=repo_root,
        state_path=tmp_path / "outside.sqlite3",
        checked_at=NOW,
    )
    assert blocked.ready is False
    assert "GIT_WORKTREE_NOT_CLEAN" in blocked.reason_codes
    assert "STATE_DB_NOT_ISOLATED_BELOW_AUTOMATION_ROOT" in blocked.reason_codes


def test_successful_canary_cli_terminal_state_returns_zero(
    tmp_path: Path, monkeypatch
):
    from genetic_algorithm import cli

    repo_root = Path(__file__).resolve().parents[2]
    automation_root = (tmp_path / "cli-canary").resolve()

    def ready_preflight(policy, *, repo_root, state_path):
        head = "a" * 40
        return HardcoreCampaignPreflightV1(
            checked_at=NOW,
            ready=True,
            campaign_id=policy.campaign_id,
            policy_hash=policy.policy_hash,
            repo_root=str(Path(repo_root).resolve()),
            automation_root=policy.automation_root,
            state_path=str(Path(state_path).resolve()),
            git_head=head,
            git_upstream="origin/test",
            git_upstream_head=head,
            worktree_clean=True,
            head_matches_upstream=True,
            config_hashes={
                CampaignLane.FIFTEEN_MINUTES: "b" * 64,
                CampaignLane.ONE_HOUR: "c" * 64,
            },
            reason_codes=["PREFLIGHT_READY"],
        )

    class BackendContext:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    class CompletedController:
        def __init__(self, **_kwargs):
            pass

        def run_forever(self):
            return ControllerTickV1(
                observed_at=NOW,
                outcome=ControllerTickOutcome.STOPPED,
                lifecycle=CampaignLifecycle.COMPLETED,
                reason_codes=["CANARY_COMPLETED"],
            )

    monkeypatch.setattr(
        "genetic_algorithm.orchestration.runner_v2.repository_root",
        lambda: repo_root,
    )
    monkeypatch.setattr(
        "genetic_algorithm.orchestration.hardcore_campaign_v1.require_hardcore_campaign_preflight",
        ready_preflight,
    )
    monkeypatch.setattr(
        "genetic_algorithm.orchestration.hardcore_backend_v1.V2HardcoreAttemptBackend",
        BackendContext,
    )
    monkeypatch.setattr(
        "genetic_algorithm.orchestration.hardcore_campaign_v1.HardcoreCampaignControllerV1",
        CompletedController,
    )

    exit_code = cli.main(
        [
            "hardcore-campaign",
            "canary",
            "--campaign-id",
            "cli-canary",
            "--automation-root",
            str(automation_root),
        ]
    )

    assert exit_code == 0
