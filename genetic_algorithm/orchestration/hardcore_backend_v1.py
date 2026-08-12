"""Immutable V2 worker backend for :mod:`hardcore_campaign_v1`.

The adapter intentionally reuses the canonical manifest, data/code fencing,
SQLite leases, process guard and Generic-Island worker.  It does not use V2
wave planning or promotion ranking.  A completed worker result is converted
back to the sole campaign decision input, ``raw-multipair-score-v2``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import signal
import sys
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any

from genetic_algorithm.config.schema import (
    load_config,
    validate_resolved_config_v2_or_raise,
)
from genetic_algorithm.evaluation.raw_multipair_score import (
    PairScenario,
    RawMultiPairPanel,
    RawMultiPairStatus,
    score_raw_multipair,
)
from genetic_algorithm.evaluation.period_provenance import (
    validate_exact_period_evidence,
)
from genetic_algorithm.orchestration.artifact_store_v2 import (
    ArtifactIntegrityError,
    V2ArtifactStore,
    canonical_config_hash,
)
from genetic_algorithm.orchestration.attempt_scheduler_v2 import AttemptSchedulerV2
from genetic_algorithm.orchestration.attempt_state_v2 import (
    AttemptLifecycleStatus,
    AttemptStateStoreV2,
    WorkerBindingV2,
    WorkerKind,
)
from genetic_algorithm.orchestration.data_manifest_v2 import resolve_spot_data_root
from genetic_algorithm.orchestration.evolution_worker_v2 import (
    FrozenEvolutionSeedV2,
    evolution_worker_argv,
    prepare_evolution_worker,
)
from genetic_algorithm.orchestration.hardcore_campaign_v1 import (
    HARDCORE_PANEL_PAIRS,
    RAW_MULTIPAIR_SCORE_VERSION,
    AttemptEvidenceV1,
    AttemptHandleV1,
    AttemptPollStatus,
    AttemptPollV1,
    CampaignLane,
    CandidateSnapshotV1,
    DiversityEventV1,
    EvolutionRunRequestV1,
    EvolutionStopReason,
    FailureClass,
    HardcoreCampaignError,
    HardcoreQueueError,
    PairRole,
    RawPairMetricsV1,
    material_improvement,
)
from genetic_algorithm.orchestration.manifest_builder_v2 import (
    build_attempt_manifest_bundle,
)
from genetic_algorithm.orchestration.promotion_policy_v2 import (
    shadow_gate_policy_from_config,
)
from genetic_algorithm.orchestration.result_contract import (
    AttemptResultV2,
)
from genetic_algorithm.orchestration.shadow_scheduler_v2 import ShadowSchedulerConfigV2


_TERMINAL = {
    AttemptLifecycleStatus.SUCCEEDED,
    AttemptLifecycleStatus.FAILED,
    AttemptLifecycleStatus.INTERRUPTED,
    AttemptLifecycleStatus.INVALID_RESULT,
}


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_immutable(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise HardcoreCampaignError(f"immutable backend artifact differs: {path}")
        return
    # O_EXCL is sufficient here: one controller owns this isolated queue.
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _request_from_handle(handle: AttemptHandleV1) -> EvolutionRunRequestV1:
    worker_root = Path(handle.artifact_root).resolve()
    request_path = worker_root.parent / "run_request.json"
    checksum_path = request_path.with_name(request_path.name + ".sha256")
    if not request_path.is_file() or not checksum_path.is_file():
        raise HardcoreCampaignError("attempt handle has no immutable run request")
    expected = checksum_path.read_text(encoding="ascii").strip()
    if _sha256_file(request_path) != expected:
        raise HardcoreCampaignError("run request checksum mismatch")
    request = EvolutionRunRequestV1.model_validate_json(request_path.read_bytes())
    if request.request_hash != expected or request.request_hash != handle.request_hash:
        raise HardcoreCampaignError("run request hash differs from attempt handle")
    return request


def _materialized_config(request: EvolutionRunRequestV1) -> dict[str, Any]:
    path = Path(request.config_path).resolve()
    if _sha256_file(path) != request.config_file_sha256:
        raise HardcoreCampaignError("lane config file changed after request creation")
    config = copy.deepcopy(load_config(path))
    if canonical_config_hash(config) != request.resolved_config_sha256:
        raise HardcoreCampaignError(
            "resolved lane config changed after request creation"
        )
    if config.get("safety_profile", {}).get("name") != "hardcore_multipair_v1":
        raise HardcoreCampaignError("hardcore backend requires its exact safety profile")
    if config.get("backtesting", {}).get("timeframe") != request.lane.value:
        raise HardcoreCampaignError("lane config timeframe differs from request")

    ga = config["genetic_algorithm"]
    ga["mutation_rate"] = request.recipe_parameters.mutation_rate
    ga["crossover_rate"] = request.recipe_parameters.crossover_rate
    ga["search_seed_salt"] = 0
    ga["random_seed"] = request.search_seed
    generic = config["generic_island_model"]
    generic["specialization"]["indicator_overlap"] = (
        request.recipe_parameters.indicator_overlap
    )
    generic["common_panel_replay"]["incumbent_score"] = (
        request.previous_global_score
    )
    generic["archive_seeding"]["island_names"] = [
        item.name for item in request.islands if item.archive_island
    ]
    by_name = {item["name"]: item for item in generic["islands"]}
    if set(by_name) != {item.name for item in request.islands}:
        raise HardcoreCampaignError("request islands differ from the lane preset")
    for blueprint in request.islands:
        target = by_name[blueprint.name]
        target.update(
            {
                "seed": blueprint.seed,
                "population_size": blueprint.population_size,
                "generations": blueprint.generations,
                "indicator_pool": list(blueprint.indicator_pool),
                "pairs": list(blueprint.pairs),
                "walk_forward_enabled": False,
            }
        )
    validate_resolved_config_v2_or_raise(config)
    return config


def _archive_seeds(request: EvolutionRunRequestV1) -> list[FrozenEvolutionSeedV2]:
    # Request order mirrors GenericIslandModelEvolution's round-robin archive
    # assignment. Reconstruct the original score-ordered flat archive.
    assignments = [item.seed_candidates for item in request.islands if item.archive_island]
    flattened: list[Any] = []
    for position in range(4):
        for island_entries in assignments:
            if position < len(island_entries):
                flattened.append(island_entries[position])
    seeds: list[FrozenEvolutionSeedV2] = []
    seen: set[str] = set()
    for archive_entry in flattened:
        candidate = archive_entry.candidate
        if candidate.phenotype_hash in seen:
            raise HardcoreCampaignError("run request repeats an archive phenotype")
        seen.add(candidate.phenotype_hash)
        path = Path(candidate.evolution_seed_path).resolve()
        if not path.is_file() or _sha256_file(path) != candidate.evolution_seed_sha256:
            raise HardcoreCampaignError("archived evolution seed is missing or changed")
        seed = FrozenEvolutionSeedV2.model_validate_json(path.read_bytes())
        if (
            seed.candidate_id != candidate.candidate_id
            or seed.phenotype_hash != candidate.phenotype_hash
            or seed.gene_hash != candidate.gene_hash
        ):
            raise HardcoreCampaignError("archived evolution seed differs from candidate")
        seeds.append(seed)
    return seeds


def _strict_replay_top_n(config: dict[str, Any]) -> int:
    """Resolve one fail-closed strict-replay bound from the lane preset."""

    value = config.get("output", {}).get("top_n")
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 12:
        raise HardcoreCampaignError("output.top_n must be an integer from 1 through 12")
    return value


class V2HardcoreAttemptBackend:
    """Concrete one-slot backend using canonical V2 worker infrastructure."""

    def __init__(
        self,
        *,
        state_path: str | Path,
        automation_root: str | Path,
        repo_root: str | Path,
        python_executable: str | Path = sys.executable,
        scheduler: AttemptSchedulerV2 | None = None,
    ) -> None:
        self.root = Path(automation_root).resolve()
        self.repo_root = Path(repo_root).resolve()
        self.python_executable = Path(python_executable).absolute()
        if not self.repo_root.is_dir() or not self.python_executable.is_file():
            raise HardcoreCampaignError("backend repo/python input does not exist")
        self.store = AttemptStateStoreV2(Path(state_path).resolve())
        self._owns_scheduler = scheduler is None
        self.scheduler = scheduler or AttemptSchedulerV2(
            self.store,
            scheduler_id="hardcore-campaign-v1",
            config=ShadowSchedulerConfigV2(
                max_concurrent=1,
                poll_interval_seconds=5.0,
                max_execution_seconds=10 * 60 * 60,
            ),
        )

    def close(self) -> None:
        if self._owns_scheduler:
            self.scheduler.close(wait=True)

    def __enter__(self) -> "V2HardcoreAttemptBackend":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def queue(self, request: EvolutionRunRequestV1) -> AttemptHandleV1:
        try:
            return self._queue_impl(request)
        except HardcoreQueueError:
            raise
        except Exception as exc:
            failure_class = _classify_failure(type(exc).__name__, str(exc))
            raise HardcoreQueueError(
                f"{type(exc).__name__}: {exc}",
                failure_class=failure_class,
                error_code=f"QUEUE_{type(exc).__name__.upper()}",
            ) from exc

    def _queue_impl(self, request: EvolutionRunRequestV1) -> AttemptHandleV1:
        config = _materialized_config(request)
        strict_replay_top_n = _strict_replay_top_n(config)
        seeds = _archive_seeds(request)
        policy = shadow_gate_policy_from_config(config)
        attempt_token = canonical_config_hash(
            {"request_hash": request.request_hash, "attempt": request.attempt_ordinal}
        )
        attempt_id = f"hardcore-attempt-{attempt_token[:24]}"
        worker_root = (
            self.root
            / "runs"
            / request.run_id
            / f"attempt-{request.attempt_ordinal}"
            / "worker"
        ).resolve()
        data_root = self.repo_root / resolve_spot_data_root(
            config, list(HARDCORE_PANEL_PAIRS)
        )
        bundle = build_attempt_manifest_bundle(
            attempt_id=attempt_id,
            wave_id=f"hardcore-{request.campaign_id}",
            experiment_id=request.run_id,
            created_at=request.created_at,
            resolved_config=config,
            policy=policy,
            fitness_policy_version=RAW_MULTIPAIR_SCORE_VERSION,
            seeds=[request.search_seed],
            worker_count=4,
            artifact_root=worker_root,
            repo_root=self.repo_root,
            data_root=data_root,
        )
        if request.resume_checkpoint_path is not None:
            checkpoint = Path(request.resume_checkpoint_path).resolve()
            if (
                not checkpoint.is_file()
                or _sha256_file(checkpoint) != request.resume_checkpoint_sha256
            ):
                raise HardcoreCampaignError("retry checkpoint is missing or changed")
        prepared = prepare_evolution_worker(
            bundle=bundle,
            resolved_config=config,
            policy=policy,
            seeds=seeds,
            repo_root=self.repo_root,
            final_test_ledger_path=self.root / "final_test_usage.sqlite3",
            created_at=request.created_at,
            top_n=strict_replay_top_n,
            resume_checkpoint_path=request.resume_checkpoint_path,
        )
        binding = WorkerBindingV2(
            worker_kind=WorkerKind.GENERIC_ISLAND_EVOLUTION,
            argv=evolution_worker_argv(
                prepared, python_executable=self.python_executable
            ),
            working_directory=str(self.repo_root),
            spec_path=str(prepared.spec_path.resolve()),
            spec_sha256=prepared.spec_file_sha256,
        )
        state = self.store.register(
            bundle.manifest,
            retry_number=request.attempt_ordinal,
            actor="hardcore-campaign-v1",
        )
        state = self.store.bind_worker(
            attempt_id,
            binding,
            bound_at=max(request.created_at, state.updated_at),
            actor="hardcore-campaign-v1",
        )
        if state.status == AttemptLifecycleStatus.DRAFT:
            state = self.store.transition(
                attempt_id,
                AttemptLifecycleStatus.VALIDATED,
                occurred_at=state.updated_at,
                actor="hardcore-campaign-v1",
                reason="HARDCORE_WORKER_VALID",
            )
        if state.status == AttemptLifecycleStatus.VALIDATED:
            self.store.transition(
                attempt_id,
                AttemptLifecycleStatus.QUEUED,
                occurred_at=state.updated_at,
                actor="hardcore-campaign-v1",
                reason="HARDCORE_RUN_QUEUED",
            )
        return AttemptHandleV1(
            attempt_id=attempt_id,
            run_id=request.run_id,
            request_hash=request.request_hash,
            artifact_root=str(worker_root),
        )
    def poll(self, handle: AttemptHandleV1) -> AttemptPollV1:
        request = _request_from_handle(handle)
        tick = self.scheduler.run_once()
        state = self.store.get(handle.attempt_id)
        self._deliver_pending_stop(state, handle)
        observed = tick.observed_at
        generation, current_score, plateau = _live_engine_progress(handle.artifact_root)
        current_pair_metrics = _live_pair_metrics(handle.artifact_root)
        if state.status not in _TERMINAL:
            return AttemptPollV1(
                status=(
                    AttemptPollStatus.QUEUED
                    if state.status
                    in {
                        AttemptLifecycleStatus.DRAFT,
                        AttemptLifecycleStatus.VALIDATED,
                        AttemptLifecycleStatus.QUEUED,
                        AttemptLifecycleStatus.CLAIMED,
                    }
                    else AttemptPollStatus.RUNNING
                ),
                observed_at=observed,
                current_generation=generation,
                current_score=current_score,
                current_pair_metrics=current_pair_metrics,
                plateau_checks=plateau,
            )
        evidence = self._load_or_build_evidence(request, state)
        return AttemptPollV1(
            status=AttemptPollStatus.FINISHED,
            observed_at=max(observed, evidence.finished_at),
            current_generation=evidence.actual_generations,
            current_score=(evidence.candidates[0].score if evidence.candidates else None),
            current_pair_metrics=(
                evidence.candidates[0].pair_metrics
                if evidence.candidates
                else []
            ),
            plateau_checks=evidence.plateau_checks,
            evidence=evidence,
        )

    def request_graceful_stop(
        self, handle: AttemptHandleV1, *, reason: EvolutionStopReason
    ) -> None:
        marker = Path(handle.artifact_root) / "runtime" / "graceful_stop.json"
        if marker.is_file():
            try:
                existing_reason = json.loads(
                    marker.read_text(encoding="utf-8")
                ).get("reason")
            except (json.JSONDecodeError, AttributeError) as exc:
                raise HardcoreCampaignError("graceful-stop marker is invalid") from exc
            if existing_reason != reason.value:
                raise HardcoreCampaignError("graceful-stop reason changed after request")
        else:
            _write_immutable(
                marker,
                _canonical_json_bytes(
                    {"reason": reason.value, "requested_at": datetime.now(UTC).isoformat()}
                ),
            )
        state = self.store.get(handle.attempt_id)
        self._deliver_pending_stop(state, handle)

    @staticmethod
    def _deliver_pending_stop(state, handle: AttemptHandleV1) -> None:
        runtime = Path(handle.artifact_root) / "runtime"
        marker = runtime / "graceful_stop.json"
        delivered = runtime / "graceful_stop.delivered"
        if not marker.is_file() or delivered.exists():
            return
        if state.status == AttemptLifecycleStatus.RUNNING and state.pid is not None:
            try:
                # Signal only the coordinator.  It catches SIGTERM, finishes
                # the current generation, checkpoints, and then shuts down
                # its evaluation pool cleanly. killpg would interrupt pool
                # workers mid-backtest and destroy generation integrity.
                os.kill(state.pid, signal.SIGTERM)
            except ProcessLookupError:
                return
            _write_immutable(delivered, b"SIGTERM\n")

    def _load_or_build_evidence(self, request, state) -> AttemptEvidenceV1:
        root = Path(state.artifact_root).resolve()
        path = root.parent / "attempt_evidence.json"
        checksum = path.with_name(path.name + ".sha256")
        if path.exists() or checksum.exists():
            if not path.is_file() or not checksum.is_file():
                raise HardcoreCampaignError("partial attempt evidence artifact")
            expected = checksum.read_text(encoding="ascii").strip()
            if _sha256_file(path) != expected:
                raise HardcoreCampaignError("attempt evidence checksum mismatch")
            return AttemptEvidenceV1.model_validate_json(path.read_bytes())
        evidence = _build_attempt_evidence(request, state)
        payload = _canonical_json_bytes(evidence.model_dump(mode="json"))
        _write_immutable(path, payload)
        _write_immutable(
            checksum, (hashlib.sha256(payload).hexdigest() + "\n").encode("ascii")
        )
        return evidence


def _verified_engine_outcome(root: Path) -> dict[str, Any] | None:
    path = root / "evolution" / "evolution_outcome.json"
    checksum = path.with_name(path.name + ".sha256")
    if not path.exists() and not checksum.exists():
        return None
    if not path.is_file() or not checksum.is_file():
        raise HardcoreCampaignError("partial engine outcome artifact")
    expected = checksum.read_text(encoding="ascii").strip()
    if _sha256_file(path) != expected:
        raise HardcoreCampaignError("engine outcome checksum mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise HardcoreCampaignError("engine outcome must be a mapping")
    return payload


def _live_engine_progress(root: str | Path) -> tuple[int, float | None, int]:
    engine = _verified_engine_outcome(Path(root).resolve())
    if engine is not None:
        curve = engine.get("common_panel_curve", [])
        return (
            int(engine.get("generations_completed", 0)),
            _finite_or_none(engine.get("best_score")),
            int(curve[-1].get("no_improvement_checks", 0)) if curve else 0,
        )
    trace = Path(root) / "evolution" / "generation_trace_v2.jsonl"
    if not trace.is_file():
        return 0, None, 0
    generations: list[int] = []
    scores: list[float] = []
    for raw in trace.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(row.get("generation"), int):
            generations.append(row["generation"])
        # The campaign ranks and archives the signed raw multipair score.
        # ``best_fitness`` is the post-sharing reproduction value and is not
        # comparable across islands or generations.
        score = _finite_or_none(row.get("best_raw_fitness"))
        if score is not None:
            scores.append(score)
    plateau_checks = 0
    checkpoint_path, _checkpoint_hash = _latest_checkpoint(Path(root).resolve())
    if checkpoint_path is not None:
        try:
            checkpoint = json.loads(
                Path(checkpoint_path).read_text(encoding="utf-8")
            )
            common = checkpoint.get("common_panel_replay", {})
            if isinstance(common, dict):
                plateau_checks = int(common.get("no_improvement_checks", 0))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            # Live status is observational.  A malformed checkpoint remains
            # authoritative only when resume/outcome verification reads it.
            plateau_checks = 0
    return (
        max(generations) + 1 if generations else 0,
        max(scores) if scores else None,
        plateau_checks,
    )


def _live_pair_metrics(root: str | Path) -> list[RawPairMetricsV1]:
    """Read the latest measured raw champion from a resumable checkpoint."""

    checkpoint_path, _checkpoint_hash = _latest_checkpoint(Path(root).resolve())
    if checkpoint_path is None:
        return []
    try:
        checkpoint = json.loads(Path(checkpoint_path).read_text(encoding="utf-8"))
        populations = checkpoint.get("island_populations", {})
        candidates = [
            item
            for population in populations.values()
            for item in population.get("individuals", [])
            if item.get("evaluated")
            and _finite_or_none(item.get("raw_fitness")) is not None
            and item.get("metrics", {}).get("raw_multipair_status") == "VALID"
        ]
        if not candidates:
            return []
        champion = max(
            candidates,
            key=lambda item: float(item["raw_fitness"]),
        )
        metrics = champion["metrics"]
        raw_pairs = metrics.get("raw_pair_metrics", {})
        result = metrics.get("raw_multipair_result", {})
        pair_components = result.get("pair_components", {})
        timeframe = str(result.get("timeframe") or "")
        panel = RawMultiPairPanel(timeframe=timeframe)
        rows: list[RawPairMetricsV1] = []
        for pair in HARDCORE_PANEL_PAIRS:
            raw = raw_pairs[pair]
            component = pair_components[pair]["components"]
            trade_count = int(raw["trade_count"])
            positive = trade_count > 0
            rows.append(
                RawPairMetricsV1(
                    pair=pair,
                    role=(
                        PairRole.DEVELOPMENT
                        if pair in HARDCORE_PANEL_PAIRS[:3]
                        else PairRole.VALIDATION
                    ),
                    net_return=raw.get("net_return") if positive else None,
                    net_expectancy=(
                        raw.get("net_expectancy") if positive else None
                    ),
                    profit_factor=raw.get("profit_factor") if positive else None,
                    profit_factor_censored=(
                        raw.get("profit_factor_censored") if positive else None
                    ),
                    trade_count=trade_count,
                    active_months=int(raw["active_months"]),
                    calendar_months=float(panel.calendar_months),
                    median_holding_hours=(
                        raw.get("median_holding_hours") if positive else None
                    ),
                    p90_holding_hours=(
                        raw.get("p90_holding_hours") if positive else None
                    ),
                    max_drawdown=raw.get("max_drawdown") if positive else None,
                    max_drawdown_duration_days=(
                        raw.get("max_drawdown_duration_days")
                        if positive
                        else None
                    ),
                    max_consecutive_losses=(
                        raw.get("max_consecutive_losses") if positive else None
                    ),
                    normalized_components={
                        short: float(component[long])
                        for short, long in _COMPONENT_ALIAS.items()
                    },
                )
            )
        return rows
    except (
        KeyError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ):
        return []


def _finite_or_none(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _latest_checkpoint(root: Path) -> tuple[str | None, str | None]:
    candidates = list((root / "evolution" / "checkpoints").glob("island_checkpoint_gen*.json"))
    if not candidates:
        return None, None

    def generation(path: Path) -> int:
        token = path.name.split("island_checkpoint_gen", 1)[1].split("_", 1)[0]
        try:
            return int(token)
        except ValueError:
            return -1

    latest = max(candidates, key=lambda path: (generation(path), path.name))
    return str(latest.resolve()), _sha256_file(latest)


def _classify_failure(error_code: str | None, detail: str | None) -> FailureClass:
    text = f"{error_code or ''} {detail or ''}".upper()
    if any(
        token in text
        for token in (
            "DATA_MANIFEST",
            "MISSING_DATA",
            "OHLCV",
            "PAIR_DATA",
            "MISSING_EQUITY",
            "MISSING_PERIOD",
            "PERIOD_COVERAGE",
        )
    ):
        return FailureClass.DETERMINISTIC_DATA
    if any(
        token in text
        for token in (
            "CONFIG",
            "VALUEERROR",
            "WORKER_INPUT_INVALID",
            "POLICY",
            "SPLIT_MANIFEST",
            "SCHEMA",
            "CHECKPOINT_PROVENANCE",
            "ARCHIVED EVOLUTION SEED",
            "RUN REQUEST",
            "REQUEST ISLANDS",
            "LANE PRESET",
            "RAW COMMON-PANEL REPLAY CHANGED",
        )
    ):
        return FailureClass.DETERMINISTIC_CONFIG
    return FailureClass.TRANSIENT


def _engine_stop_reason(engine: dict[str, Any] | None) -> EvolutionStopReason:
    if engine is None:
        return EvolutionStopReason.TECHNICAL_INVALID
    raw = str(engine.get("reason") or "")
    try:
        return EvolutionStopReason(raw)
    except ValueError:
        return EvolutionStopReason.TECHNICAL_INVALID


def _holding_hours(trades: list[dict[str, Any]]) -> tuple[float | None, float | None]:
    durations: list[float] = []
    for trade in trades:
        raw = trade.get("trade_duration") if isinstance(trade, dict) else None
        value = _finite_or_none(raw)
        if value is not None and value >= 0:
            durations.append(value / 60.0)
            continue
        if not isinstance(trade, dict):
            continue
        opened = trade.get("open_date") or trade.get("open_timestamp")
        closed = trade.get("close_date") or trade.get("close_timestamp")
        try:
            if isinstance(opened, (int, float)) and isinstance(closed, (int, float)):
                # Freqtrade timestamps are normally milliseconds; accept
                # seconds too without turning a missing duration into zero.
                scale = 1000.0 if max(abs(opened), abs(closed)) > 10**11 else 1.0
                duration_hours = (float(closed) - float(opened)) / scale / 3600.0
            elif isinstance(opened, str) and isinstance(closed, str):
                start = datetime.fromisoformat(opened.replace("Z", "+00:00"))
                end = datetime.fromisoformat(closed.replace("Z", "+00:00"))
                duration_hours = (end - start).total_seconds() / 3600.0
            else:
                continue
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(duration_hours) and duration_hours >= 0:
            durations.append(duration_hours)
    if not durations:
        return None, None
    ordered = sorted(durations)

    def quantile(probability: float) -> float:
        location = probability * (len(ordered) - 1)
        lower = math.floor(location)
        upper = math.ceil(location)
        if lower == upper:
            return ordered[lower]
        weight = location - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return median(ordered), quantile(0.9)


def _scenario_from_record(record, *, timeframe: CampaignLane) -> PairScenario:
    metrics = record.metrics
    panel = RawMultiPairPanel(timeframe=timeframe.value)
    declared_period = validate_exact_period_evidence(
        expected_start=panel.period_start,
        expected_end=panel.period_end,
        observed_start=getattr(metrics, "period_start", None),
        observed_end=getattr(metrics, "period_end", None),
        evidence_label=f"{metrics.pair} declared period",
    )
    measured_period = validate_exact_period_evidence(
        expected_start=panel.period_start,
        expected_end=panel.period_end,
        observed_start=getattr(metrics, "measured_period_start", None),
        observed_end=getattr(metrics, "measured_period_end", None),
        evidence_label=f"{metrics.pair} measured period",
        timeframe=panel.timeframe,
    )
    period_evidence = declared_period if not declared_period.valid else measured_period

    def failed(error_code: str, detail: str | None = None) -> PairScenario:
        return PairScenario(
            pair=metrics.pair,
            timeframe=timeframe.value,
            success=False,
            trade_count=0,
            active_months=0,
            period_start=measured_period.measured_start,
            period_end=measured_period.measured_end,
            technical_error=(
                f"{error_code}: {detail}" if detail else error_code
            ),
        )

    if not period_evidence.valid:
        return failed(
            str(period_evidence.error_code),
            period_evidence.error_detail,
        )
    if getattr(metrics, "timeframe", None) != panel.timeframe:
        return failed(
            "TIMEFRAME_MISMATCH",
            f"measured={getattr(metrics, 'timeframe', None)!r}, "
            f"expected={panel.timeframe!r}",
        )

    status = getattr(metrics.status, "value", metrics.status)
    error_code = metrics.error_code
    valid_zero_trades = (
        status == "INCONCLUSIVE"
        and error_code == "NO_TRADES"
        and metrics.success is True
        and metrics.no_trades is True
        and metrics.trade_count == 0
        and metrics.active_months == 0
    )
    if valid_zero_trades:
        return PairScenario(
            pair=metrics.pair,
            timeframe=timeframe.value,
            success=True,
            trade_count=0,
            active_months=0,
            period_start=measured_period.measured_start,
            period_end=measured_period.measured_end,
        )

    valid_point_metrics = (
        status == "INCONCLUSIVE"
        and error_code == "RAW_POINT_METRICS_ONLY"
        and metrics.success is True
        and metrics.no_trades is False
        and metrics.trade_count > 0
    )
    if not valid_point_metrics:
        return failed(
            error_code or "STRICT_RAW_RECORD_NOT_ALLOWED",
            getattr(metrics, "error_detail", None),
        )

    holding_median, holding_p90 = _holding_hours(record.trades)
    return PairScenario(
        pair=metrics.pair,
        timeframe=timeframe.value,
        success=True,
        trade_count=metrics.trade_count,
        active_months=metrics.active_months,
        period_start=measured_period.measured_start,
        period_end=measured_period.measured_end,
        net_return=metrics.net_return,
        # Point estimate only. The committed-capital field is coupled to the
        # old clustered/bootstrap evidence contract and may be absent for
        # low-trade candidates; raw score uses the unconditional per-trade
        # expectancy populated directly from the backtest result.
        net_expectancy=metrics.net_expectancy,
        profit_factor=metrics.profit_factor,
        profit_factor_censored=metrics.profit_factor_censored,
        median_holding_hours=holding_median,
        p90_holding_hours=holding_p90,
        max_drawdown=metrics.max_drawdown,
        max_drawdown_duration_days=metrics.max_drawdown_duration_days,
        max_consecutive_losses=metrics.max_consecutive_losses,
    )


_COMPONENT_ALIAS = {
    "R": "return_score",
    "E": "expectancy_score",
    "P": "profit_factor_score",
    "A": "activity_score",
    "H": "holding_score",
    "D": "drawdown_risk",
    "U": "drawdown_duration_risk",
    "L": "loss_streak_risk",
    "O": "overtrading_risk",
}


def _candidate_snapshot(
    candidate,
    *,
    request: EvolutionRunRequestV1,
    result: AttemptResultV2,
    worker_root: Path,
    strict_failure_codes: list[str] | None = None,
) -> CandidateSnapshotV1 | None:
    panel = RawMultiPairPanel(
        timeframe=request.lane.value,
    )
    records = {record.metrics.pair: record for record in candidate.scenarios}
    if set(records) != set(HARDCORE_PANEL_PAIRS):
        return None
    scenarios = [
        _scenario_from_record(records[pair], timeframe=request.lane)
        for pair in HARDCORE_PANEL_PAIRS
    ]
    score = score_raw_multipair(panel, scenarios)
    if score.status != RawMultiPairStatus.VALID or score.score is None:
        if strict_failure_codes is not None:
            technical_codes = [
                str(scenario.technical_error).split(":", 1)[0]
                for scenario in scenarios
                if scenario.technical_error
            ]
            strict_failure_codes.extend(technical_codes or [score.reason_code])
        return None
    components = score.pair_component_map()
    raw_metrics: list[RawPairMetricsV1] = []
    for pair, scenario in zip(HARDCORE_PANEL_PAIRS, scenarios, strict=True):
        pair_components = components[pair].components.to_dict()
        raw_metrics.append(
            RawPairMetricsV1(
                pair=pair,
                role=(
                    PairRole.DEVELOPMENT
                    if pair in HARDCORE_PANEL_PAIRS[:3]
                    else PairRole.VALIDATION
                ),
                net_return=scenario.net_return,
                net_expectancy=scenario.net_expectancy,
                profit_factor=scenario.profit_factor,
                profit_factor_censored=scenario.profit_factor_censored,
                trade_count=scenario.trade_count,
                active_months=scenario.active_months,
                calendar_months=float(panel.calendar_months),
                median_holding_hours=scenario.median_holding_hours,
                p90_holding_hours=scenario.p90_holding_hours,
                max_drawdown=scenario.max_drawdown,
                max_drawdown_duration_days=scenario.max_drawdown_duration_days,
                max_consecutive_losses=scenario.max_consecutive_losses,
                normalized_components={
                    short: pair_components[long]
                    for short, long in _COMPONENT_ALIAS.items()
                },
            )
        )
    seed_path = worker_root / "candidates" / candidate.candidate_id / "evolution_seed.json"
    if not seed_path.is_file():
        return None
    seed = FrozenEvolutionSeedV2.model_validate_json(seed_path.read_bytes())
    if seed.phenotype_hash != candidate.phenotype_hash:
        raise HardcoreCampaignError("evolution seed phenotype differs from replay candidate")
    return CandidateSnapshotV1(
        candidate_id=candidate.candidate_id,
        phenotype_hash=candidate.phenotype_hash,
        score=score.score,
        timeframe=request.lane,
        panel_id=score.panel_id,
        evolution_seed_path=str(seed_path.resolve()),
        evolution_seed_sha256=_sha256_file(seed_path),
        gene_hash=seed.gene_hash,
        pair_metrics=raw_metrics,
    )


def _diversity_events(engine: dict[str, Any] | None) -> list[DiversityEventV1]:
    if engine is None:
        return []
    result: list[DiversityEventV1] = []
    mapping = {
        "COLLAPSE_OBSERVED": "COLLAPSE_OBSERVED",
        "RECOVERY_APPLIED": "RECOVERY_APPLIED",
        "STOP_DIVERSITY_COLLAPSE": "RECOVERY_FAILED",
    }
    for raw in engine.get("diversity_events", []):
        event_type = mapping.get(str(raw.get("action")))
        islands = list(raw.get("collapsed_islands", []))
        if event_type is None or not islands:
            continue
        observations = raw.get("islands", {})
        duplicate = max(
            (float(observations.get(name, {}).get("duplicate_fraction", 0.0)) for name in islands),
            default=0.0,
        )
        diversity = min(
            (float(observations.get(name, {}).get("genetic_diversity", 0.0)) for name in islands),
            default=0.0,
        )
        result.append(
            DiversityEventV1(
                generation=int(raw.get("generation", 1)),
                event_type=event_type,
                affected_islands=islands,
                duplicate_fraction=duplicate,
                genetic_diversity=diversity,
                mutation_rate_after=(0.35 if event_type == "RECOVERY_APPLIED" else None),
            )
        )
    return result


def _build_attempt_evidence(request, state) -> AttemptEvidenceV1:
    worker_root = Path(state.artifact_root).resolve()
    engine = _verified_engine_outcome(worker_root)
    stop_reason = _engine_stop_reason(engine)
    result: AttemptResultV2 | None = None
    try:
        result = V2ArtifactStore(worker_root).read_verified_result()
    except ArtifactIntegrityError:
        result = None
    candidates: list[CandidateSnapshotV1] = []
    strict_failure_codes: list[str] = []
    if result is not None:
        for candidate in result.candidate_evaluations:
            snapshot = _candidate_snapshot(
                candidate,
                request=request,
                result=result,
                worker_root=worker_root,
                strict_failure_codes=strict_failure_codes,
            )
            if snapshot is not None:
                candidates.append(snapshot)
    candidates = sorted(candidates, key=lambda item: item.score, reverse=True)
    error_code = state.error_code or (result.error_code if result is not None else None)
    error_detail = state.error_detail or (result.error_detail if result is not None else None)
    engine_detail = str(engine.get("detail") or "") if engine is not None else ""
    if "raw common-panel replay changed" in engine_detail.lower():
        # The outer worker contract collapses an empty finalist set into a
        # generic IslandResultContractError.  Preserve the authoritative
        # engine cause so deterministic score corruption suspends the lane
        # instead of consuming the one transient checkpoint retry.
        error_detail = engine_detail
    graceful_marker = worker_root / "runtime" / "graceful_stop.json"
    if graceful_marker.is_file() and candidates:
        try:
            requested_reason = json.loads(
                graceful_marker.read_text(encoding="utf-8")
            ).get("reason")
            stop_reason = EvolutionStopReason(requested_reason)
        except (ValueError, json.JSONDecodeError, AttributeError):
            stop_reason = EvolutionStopReason.CAMPAIGN_DEADLINE
    if not candidates:
        stop_reason = EvolutionStopReason.TECHNICAL_INVALID
        deterministic_data_code = next(
            (
                code
                for code in strict_failure_codes
                if _classify_failure(code, None)
                == FailureClass.DETERMINISTIC_DATA
            ),
            None,
        )
        error_code = (
            deterministic_data_code
            or error_code
            or (strict_failure_codes[0] if strict_failure_codes else None)
            or "NO_VALID_RAW_MULTIPAIR_CANDIDATE"
        )
        error_detail = error_detail or "worker produced no valid six-pair raw-score candidate"
    if (
        candidates
        and stop_reason == EvolutionStopReason.COMPLETED_BUDGET
        and not material_improvement(
            request.previous_global_score, candidates[0].score
        )
    ):
        stop_reason = EvolutionStopReason.NO_GLOBAL_PROGRESS
    failure_class = (
        _classify_failure(error_code, error_detail)
        if stop_reason == EvolutionStopReason.TECHNICAL_INVALID
        else FailureClass.NONE
    )
    checkpoint_path, checkpoint_hash = _latest_checkpoint(worker_root)
    if failure_class != FailureClass.TRANSIENT:
        checkpoint_path = None
        checkpoint_hash = None
    curve = []
    plateau_checks = 0
    if engine is not None:
        for row in engine.get("common_panel_curve", []):
            value = _finite_or_none(row.get("best_fitness"))
            if value is not None:
                curve.append(value)
        if engine.get("common_panel_curve"):
            plateau_checks = int(
                engine["common_panel_curve"][-1].get("no_improvement_checks", 0)
            )
    started_at = result.started_at if result is not None else state.created_at
    finished_at = result.finished_at if result is not None else state.updated_at
    generations = int(engine.get("generations_completed", 0)) if engine else 0
    search_valid = int(engine.get("valid_evaluations", 0)) if engine else 0
    search_failed = int(engine.get("failed_evaluations", 0)) if engine else 0
    common_valid = (
        int(engine.get("common_panel_valid_evaluations", 0)) if engine else 0
    )
    common_failed = (
        int(engine.get("common_panel_failed_evaluations", 0)) if engine else 0
    )
    # One evaluation is one complete six-pair candidate in every stage. Do
    # not mix ScenarioRecord counts into the search/common candidate counts.
    strict_valid = len(candidates)
    strict_failed = (
        max(0, len(result.candidate_evaluations) - strict_valid)
        if result is not None
        else 0
    )
    total_failed = search_failed + common_failed + strict_failed
    if stop_reason == EvolutionStopReason.TECHNICAL_INVALID and total_failed == 0:
        # A coordinator/config/integrity failure may occur before any
        # backtest counter exists. Keep the top-level failure total truthful
        # while the detailed stage counters remain zero.
        total_failed = 1
    return AttemptEvidenceV1(
        campaign_id=request.campaign_id,
        run_id=request.run_id,
        request_hash=request.request_hash,
        lane=request.lane,
        recipe=request.recipe,
        attempt_ordinal=request.attempt_ordinal,
        started_at=started_at,
        finished_at=finished_at,
        stop_reason=stop_reason,
        actual_generations=generations,
        score_curve=curve[:generations],
        valid_evaluations=search_valid + common_valid + strict_valid,
        failed_evaluations=total_failed,
        search_valid_evaluations=search_valid,
        search_failed_evaluations=search_failed,
        common_panel_valid_evaluations=common_valid,
        common_panel_failed_evaluations=common_failed,
        strict_replay_valid_evaluations=strict_valid,
        strict_replay_failed_evaluations=strict_failed,
        plateau_checks=min(4, max(0, plateau_checks)),
        candidates=candidates[:12],
        diversity_events=_diversity_events(engine),
        failure_class=failure_class,
        error_code=(error_code or "TECHNICAL_WORKER_FAILURE")
        if stop_reason == EvolutionStopReason.TECHNICAL_INVALID
        else None,
        error_detail=error_detail,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_hash,
    )
