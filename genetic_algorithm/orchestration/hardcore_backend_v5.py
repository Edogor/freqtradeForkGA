"""Concrete V2-worker adapter for the three-lane V5 campaign.

V1's backend deliberately accepts only its historical two-lane request
objects.  This adapter shares the immutable V2 manifest/scheduler machinery,
but owns its V5 request and strict-replay rescore contract so that 4h never
falls through a compatibility coercion.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from genetic_algorithm.config.schema import load_config, validate_resolved_config_v2_or_raise
from genetic_algorithm.core.strategy_gene import StrategyGene
from genetic_algorithm.evaluation.panel_contract import phenotype_fingerprint
from genetic_algorithm.evaluation.raw_multipair_score import RawMultiPairStatus
from genetic_algorithm.evaluation.raw_multipair_score_v5 import (
    RAW_MULTIPAIR_SCORE_V5_VERSION,
    RawMultiPairPanelV5,
    RawMultiPairPolicyV5,
    score_raw_multipair_v5,
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
    validate_evolution_seed,
)
from genetic_algorithm.orchestration.hardcore_backend_v1 import (
    _classify_failure,
    _engine_stop_reason,
    _finite_or_none,
    _live_engine_progress,
    _live_pair_metrics,
    _scenario_from_record,
    _verified_engine_outcome,
)
from genetic_algorithm.orchestration.hardcore_campaign_v1 import (
    HARDCORE_PANEL_PAIRS,
    EvolutionStopReason,
    FailureClass,
)
from genetic_algorithm.orchestration.manifest_builder_v2 import build_attempt_manifest_bundle
from genetic_algorithm.orchestration.promotion_policy_v2 import shadow_gate_policy_from_config
from genetic_algorithm.orchestration.shadow_scheduler_v2 import ShadowSchedulerConfigV2


_TERMINAL = {
    AttemptLifecycleStatus.SUCCEEDED,
    AttemptLifecycleStatus.FAILED,
    AttemptLifecycleStatus.INTERRUPTED,
    AttemptLifecycleStatus.INVALID_RESULT,
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(payload: object) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


class V2HardcoreAttemptBackendV5:
    """Single-slot, hash-bound backend for :class:`HardcoreCampaignControllerV5`."""

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
        # ``resolve()`` would collapse a virtualenv's python symlink to the
        # base interpreter and silently drop site-packages (pandas/freqtrade).
        self.python_executable = Path(python_executable).absolute()
        if not self.repo_root.is_dir() or not self.python_executable.is_file():
            raise ValueError("V5 backend repo/python input does not exist")
        self.store = AttemptStateStoreV2(Path(state_path).resolve())
        self._owns_scheduler = scheduler is None
        self.scheduler = scheduler or AttemptSchedulerV2(
            self.store,
            scheduler_id="hardcore-campaign-v5",
            config=ShadowSchedulerConfigV2(
                max_concurrent=1,
                poll_interval_seconds=5.0,
                max_execution_seconds=10 * 60 * 60,
            ),
        )
        self._requests: dict[str, tuple[dict[str, object], dict[str, Any]]] = {}

    def close(self) -> None:
        if self._owns_scheduler:
            self.scheduler.close(wait=True)

    def __enter__(self) -> V2HardcoreAttemptBackendV5:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def queue(self, request: dict[str, object]) -> str:
        required = {"campaign_id", "run_id", "lane", "profile", "seed", "shape", "config_path"}
        if not required <= set(request):
            raise ValueError("incomplete V5 queue request")
        lane = str(request["lane"])
        if lane not in {"15m", "1h", "4h"}:
            raise ValueError("V5 queue request has an invalid lane")
        config = self._materialize(request)
        seeds = self._archive_seeds(request, config)
        persisted_request = {
            "request": request,
            "resolved_config_sha256": canonical_config_hash(config),
        }
        request_hash = hashlib.sha256(_canonical(persisted_request)).hexdigest()
        attempt_id = f"hardcore-v5-{request_hash[:24]}"
        root = self.root / "runs" / str(request["run_id"]) / "attempt-0" / "worker"
        root.parent.mkdir(parents=True, exist_ok=True)
        request_path = root.parent / "run_request_v5.json"
        request_payload = _canonical(persisted_request)
        self._quarantine_conflicting_unprepared_request(request_path, request_payload)
        policy = shadow_gate_policy_from_config(config)
        bundle = build_attempt_manifest_bundle(
            attempt_id=attempt_id,
            wave_id=f"hardcore-v5-{request['campaign_id']}",
            experiment_id=str(request["run_id"]),
            created_at=datetime.now(UTC),
            resolved_config=config,
            policy=policy,
            fitness_policy_version=RAW_MULTIPAIR_SCORE_V5_VERSION,
            seeds=[int(request["seed"])],
            worker_count=4,
            artifact_root=root,
            repo_root=self.repo_root,
            data_root=self.repo_root / resolve_spot_data_root(config, list(HARDCORE_PANEL_PAIRS)),
        )
        top_n = int(config.get("output", {}).get("top_n", 0))
        if not 1 <= top_n <= 12:
            raise ValueError("V5 output.top_n must be in 1..12")
        prepared = prepare_evolution_worker(
            bundle=bundle,
            resolved_config=config,
            policy=policy,
            seeds=seeds,
            repo_root=self.repo_root,
            final_test_ledger_path=self.root / "final_test_usage_v5.sqlite3",
            created_at=datetime.now(UTC),
            top_n=top_n,
        )
        self._persist_immutable_request(request_path, request_payload)
        binding = WorkerBindingV2(
            worker_kind=WorkerKind.GENERIC_ISLAND_EVOLUTION,
            argv=evolution_worker_argv(prepared, python_executable=self.python_executable),
            working_directory=str(self.repo_root),
            spec_path=str(prepared.spec_path.resolve()),
            spec_sha256=prepared.spec_file_sha256,
        )
        state = self.store.register(bundle.manifest, retry_number=0, actor="hardcore-v5")
        state = self.store.bind_worker(
            attempt_id, binding, bound_at=state.updated_at, actor="hardcore-v5"
        )
        if state.status == AttemptLifecycleStatus.DRAFT:
            state = self.store.transition(
                attempt_id,
                AttemptLifecycleStatus.VALIDATED,
                occurred_at=state.updated_at,
                actor="hardcore-v5",
                reason="V5_WORKER_VALID",
            )
        if state.status == AttemptLifecycleStatus.VALIDATED:
            self.store.transition(
                attempt_id,
                AttemptLifecycleStatus.QUEUED,
                occurred_at=state.updated_at,
                actor="hardcore-v5",
                reason="V5_RUN_QUEUED",
            )
        self._requests[attempt_id] = (dict(request), config)
        return attempt_id

    @staticmethod
    def _persist_immutable_request(request_path: Path, payload: bytes) -> None:
        """Write a request only after worker preparation has succeeded.

        Older versions wrote this file before validating archive assignments.
        A rejected preflight then left a request without a worker spec, which
        prevented the controller from retrying the same logical run after a
        deterministic code repair.  Such a file is not an executed immutable
        attempt, so retain it under a content-addressed quarantine name and
        create the valid request.  Any prepared worker remains immutable.
        """

        if request_path.exists():
            if request_path.read_bytes() == payload:
                return
            raise ValueError("immutable V5 request differs")
        checksum_path = request_path.with_suffix(".json.sha256")
        request_path.write_bytes(payload)
        checksum_path.write_text(hashlib.sha256(payload).hexdigest() + "\n", encoding="ascii")

    @staticmethod
    def _quarantine_conflicting_unprepared_request(request_path: Path, payload: bytes) -> None:
        """Preserve a rejected preflight request before preparing its retry."""

        if not request_path.exists() or request_path.read_bytes() == payload:
            return
        worker_spec = request_path.parent / "worker" / "worker_spec.json"
        if worker_spec.exists():
            raise ValueError("immutable V5 request differs")
        checksum_path = request_path.with_suffix(".json.sha256")
        suffix = hashlib.sha256(request_path.read_bytes()).hexdigest()[:16]
        quarantined = request_path.with_name(
            f"run_request_v5.preflight-rejected-{suffix}.json"
        )
        request_path.replace(quarantined)
        if checksum_path.exists():
            checksum_path.replace(quarantined.with_suffix(".json.sha256"))

    @staticmethod
    def _archive_seeds(
        request: dict[str, object], config: dict[str, Any]
    ) -> list[FrozenEvolutionSeedV2]:
        """Load only compatible optional parents and assign bridges by niche.

        Bad historical parents are quarantined by omission: they may not turn
        a valid lane/configuration into a deterministic failure.
        """

        supplied = request.get("archive_seeds", [])
        if not isinstance(supplied, list):
            return []
        accepted_by_id: dict[str, FrozenEvolutionSeedV2] = {}
        ids_by_niche: dict[str, list[str]] = {}
        seen: set[str] = set()
        for entry in supplied:
            if not isinstance(entry, dict):
                continue
            path = entry.get("evolution_seed_path")
            expected = entry.get("evolution_seed_sha256")
            niche = str(entry.get("niche", ""))
            if not isinstance(path, str) or not isinstance(expected, str):
                continue
            seed_path = Path(path)
            try:
                if not seed_path.is_file() or _sha256_file(seed_path) != expected:
                    continue
                seed = FrozenEvolutionSeedV2.model_validate_json(seed_path.read_bytes())
                if seed.candidate_id in seen:
                    continue
                validate_evolution_seed(seed, config)
            except (OSError, ValueError):
                continue
            seen.add(seed.candidate_id)
            accepted_by_id[seed.candidate_id] = seed
            ids_by_niche.setdefault(niche, []).append(seed.candidate_id)
        # A bridge gets up to two EDGE and two ACTIVITY parents.  If a young
        # archive does not yet contain all four, fill the remaining slots in a
        # deterministic niche order.  Crucially, only these assigned parents
        # are handed to the immutable worker.  Passing the whole 12-entry
        # archive while assigning four IDs made the worker (correctly) reject
        # the request as a tampered seed assignment after the first cycle.
        chosen: list[tuple[str, str]] = []
        seen_chosen: set[str] = set()

        def choose(niche: str, limit: int) -> None:
            for candidate_id in ids_by_niche.get(niche, []):
                if len([item for item in chosen if item[1] == niche]) >= limit:
                    break
                if candidate_id not in seen_chosen:
                    chosen.append((candidate_id, niche))
                    seen_chosen.add(candidate_id)

        choose("EDGE", 2)
        choose("ACTIVITY", 2)
        for niche in ("PRODUCTIVE", "BALANCED", "NOVELTY"):
            for candidate_id in ids_by_niche.get(niche, []):
                if len(chosen) >= 4:
                    break
                if candidate_id not in seen_chosen:
                    chosen.append((candidate_id, niche))
                    seen_chosen.add(candidate_id)
            if len(chosen) >= 4:
                break
        generic = config["generic_island_model"]
        assignments: dict[str, list[dict[str, str]]] = {}
        for island in generic.get("islands", []):
            name = str(island.get("name", ""))
            if not name.startswith("bridge-"):
                continue
            assignments[name] = [
                {"candidate_id": candidate_id, "niche": niche}
                for candidate_id, niche in chosen
            ]
        generic["archive_seeding"]["assignments"] = assignments
        return [accepted_by_id[candidate_id] for candidate_id, _ in chosen]

    def _materialize(self, request: dict[str, object]) -> dict[str, Any]:
        path = Path(str(request["config_path"])).resolve()
        config = copy.deepcopy(load_config(path))
        if config["raw_multipair_score"].get("policy_version") != RAW_MULTIPAIR_SCORE_V5_VERSION:
            raise ValueError("V5 backend requires raw-multipair-score-v5")
        if config["backtesting"]["timeframe"] != request["lane"]:
            raise ValueError("V5 lane/config timeframe differs")
        shape = request["shape"]
        if not isinstance(shape, dict):
            raise ValueError("V5 execution shape must be a mapping")
        population, generations = int(shape["population_size"]), int(shape["generations"])
        # Three evidence stages plus the deliberately reduced operational
        # canary.  Keep this allow-list fail-closed so controllers cannot
        # silently request an unreviewed resource shape.
        if (population, generations) not in {(4, 3), (10, 12), (12, 18), (12, 30)}:
            raise ValueError("unapproved V5 execution shape")
        profile = str(request["profile"])
        weights = {"edge": (0.60, 0.30), "balanced": (0.45, 0.45), "productive": (0.30, 0.60)}
        if profile not in weights:
            raise ValueError("unknown V5 fitness profile")
        raw = config["raw_multipair_score"]
        base = RawMultiPairPolicyV5.from_mapping(raw["policy"])
        policy = replace(
            base,
            score_edge_weight=weights[profile][0],
            score_productive_frequency_weight=weights[profile][1],
        )
        raw["policy"] = policy.to_dict()
        ga = config["genetic_algorithm"]
        ga["population_size"] = population
        ga["generations"] = generations
        ga["random_seed"] = int(request["seed"])
        generic = config["generic_island_model"]
        generic["population_per_island"] = population
        generic["generations"] = generations
        generic["archive_seeding"]["cross_niche_offspring_per_generation"] = int(
            shape["cross_niche_offspring"]
        )
        for island in generic["islands"]:
            island["population_size"] = population
            island["generations"] = generations
        validate_resolved_config_v2_or_raise(config)
        return config

    def poll(self, handle: str) -> dict[str, object]:
        tick = self.scheduler.run_once()
        state = self.store.get(handle)
        if handle not in self._requests:
            self._requests[handle] = self._restore_request(state)
        if state.status not in _TERMINAL:
            generation, score, plateau_checks = _live_engine_progress(state.artifact_root)
            return {
                "status": "RUNNING",
                "observed_at": tick.observed_at.isoformat(),
                "current_generation": generation,
                "current_score": score,
                "current_pair_metrics": [
                    item.model_dump(mode="json") for item in _live_pair_metrics(state.artifact_root)
                ],
                "plateau_checks": plateau_checks,
            }
        request, config = self._requests[handle]
        return {"status": "FINISHED", "evidence": self._evidence(request, config, state)}

    def _restore_request(self, state) -> tuple[dict[str, object], dict[str, Any]]:
        path = Path(state.artifact_root).resolve().parent / "run_request_v5.json"
        checksum = path.with_suffix(".json.sha256")
        if not path.is_file() or not checksum.is_file():
            raise ValueError("active V5 attempt has no immutable request")
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != checksum.read_text(encoding="ascii").strip():
            raise ValueError("active V5 request checksum differs")
        stored = json.loads(payload)
        if not isinstance(stored, dict) or not isinstance(stored.get("request"), dict):
            raise ValueError("active V5 request is malformed")
        request = stored["request"]
        config = self._materialize(request)
        self._archive_seeds(request, config)
        if stored.get("resolved_config_sha256") != canonical_config_hash(config):
            raise ValueError("active V5 resolved config changed after queue")
        return request, config

    def request_graceful_stop(self, handle: str, *, reason: str) -> None:
        if handle not in self._requests:
            state = self.store.get(handle)
            self._requests[handle] = self._restore_request(state)
        root = Path(self.store.get(handle).artifact_root) / "runtime"
        root.mkdir(parents=True, exist_ok=True)
        marker = root / "graceful_stop.json"
        payload = _canonical({"reason": reason, "requested_at": datetime.now(UTC).isoformat()})
        if marker.exists():
            try:
                existing_reason = json.loads(marker.read_text(encoding="utf-8"))["reason"]
            except (KeyError, json.JSONDecodeError, TypeError) as exc:
                raise ValueError("V5 graceful-stop marker is invalid") from exc
            if existing_reason != reason:
                raise ValueError("V5 graceful-stop reason changed")
            return
        if not marker.exists():
            marker.write_bytes(payload)

    def _evidence(
        self, request: dict[str, object], config: dict[str, Any], state
    ) -> dict[str, object]:
        root = Path(state.artifact_root).resolve()
        engine = _verified_engine_outcome(root)
        stop_reason = _engine_stop_reason(engine).value
        candidates: list[dict[str, object]] = []
        try:
            result = V2ArtifactStore(root).read_verified_result()
        except ArtifactIntegrityError:
            result = None
        if result is not None:
            for item in result.candidate_evaluations:
                candidate = self._score_candidate(
                    item,
                    lane=str(request["lane"]),
                    base_policy=RawMultiPairPolicyV5.from_mapping(
                        config["raw_multipair_score"]["policy"]
                    ),
                    worker_root=root,
                )
                if candidate is not None:
                    candidates.append(candidate)
        candidates.sort(key=lambda candidate: float(candidate["scores"]["balanced"]), reverse=True)
        valid = bool(candidates)
        failure_class = FailureClass.NONE.value
        if not valid:
            stop_reason = "TECHNICAL_INVALID"
            engine_detail = str(engine.get("detail", "")) if engine else ""
            if (
                _engine_stop_reason(engine) is EvolutionStopReason.TECHNICAL_INVALID
                and "zero valid backtest evidence" in engine_detail.lower()
            ):
                # The engine finished a whole generation and proved that no
                # candidate has valid six-pair evidence.  Retrying the same
                # config cannot repair that deterministic contract failure.
                failure_class = FailureClass.DETERMINISTIC_CONFIG.value
            else:
                failure_class = _classify_failure(
                    state.error_code, state.error_detail
                ).value
        champion = candidates[0] if candidates else None
        components = champion["scores"] if champion else {}
        pairs = champion["pairs"] if champion else []
        drawdowns = [float(pair["D"]) for pair in pairs] if pairs else [1.0]
        engine_curve = engine.get("common_panel_curve", []) if engine else []
        return {
            "valid": valid,
            "comparison_score": float(components.get("balanced", 0.0)),
            "productive_score": float(components.get("productive", 0.0)),
            "clusters": len({str(candidate["phenotype_hash"]) for candidate in candidates}),
            "worst_drawdown": max(drawdowns),
            "stop_reason": stop_reason,
            "failure_class": failure_class,
            "actual_generations": int(engine.get("generations_completed", 0)) if engine else 0,
            "score_curve": [
                float(row["best_fitness"])
                for row in engine_curve
                if _finite_or_none(row.get("best_fitness")) is not None
            ],
            "candidates": candidates,
        }

    @staticmethod
    def _score_candidate(
        candidate, *, lane: str, base_policy: RawMultiPairPolicyV5, worker_root: Path
    ) -> dict[str, object] | None:
        records = {record.metrics.pair: record for record in candidate.scenarios}
        if set(records) != set(HARDCORE_PANEL_PAIRS):
            return None
        scenarios = [
            _scenario_from_record(
                record,
                timeframe=type("Lane", (), {"value": lane})(),
                score_version=RAW_MULTIPAIR_SCORE_V5_VERSION,
            )
            for record in (records[pair] for pair in HARDCORE_PANEL_PAIRS)
        ]
        scores: dict[str, float] = {}
        policy_hashes: dict[str, str] = {}
        result = None
        for profile, weights in {
            "edge": (0.60, 0.30),
            "balanced": (0.45, 0.45),
            "productive": (0.30, 0.60),
        }.items():
            policy = replace(
                base_policy,
                score_edge_weight=weights[0],
                score_productive_frequency_weight=weights[1],
            )
            result = score_raw_multipair_v5(
                RawMultiPairPanelV5(timeframe=lane, policy=policy), scenarios
            )
            if result.status != RawMultiPairStatus.VALID or result.score is None:
                return None
            scores[profile] = float(result.score)
            policy_hashes[profile] = policy.policy_hash
        if result is None:
            return None
        seed_path = worker_root / "candidates" / candidate.candidate_id / "evolution_seed.json"
        if not seed_path.is_file():
            return None
        seed = FrozenEvolutionSeedV2.model_validate_json(seed_path.read_bytes())
        if seed.phenotype_hash != candidate.phenotype_hash:
            return None
        strategy = StrategyGene.from_dict_exact(seed.strategy_gene)
        logic_tokens = sorted(
            {
                *(f"indicator:{gene.type}" for gene in strategy.indicators),
                *(f"entry:{condition.indicator}" for condition in strategy.entry_conditions),
                *(f"exit:{condition.indicator}" for condition in strategy.exit_conditions),
            }
        )
        pair_components = result.pair_component_map()
        pairs = [
            {
                "pair": pair,
                "Q": pair_components[pair].components.edge_score,
                "A": pair_components[pair].components.activity_score,
                "F": pair_components[pair].components.productive_frequency_score,
                "R": pair_components[pair].components.return_score,
                "D": pair_components[pair].components.drawdown_risk,
                "U": pair_components[pair].components.drawdown_duration_risk,
                "trades": records[pair].metrics.trade_count,
            }
            for pair in HARDCORE_PANEL_PAIRS
        ]
        return {
            "phenotype_hash": str(getattr(candidate, "phenotype_hash", "")),
            "evolutionary_phenotype_hash": phenotype_fingerprint(strategy),
            "candidate_id": candidate.candidate_id,
            "gene_hash": seed.gene_hash,
            "evolution_seed_path": str(seed_path.resolve()),
            "evolution_seed_sha256": _sha256_file(seed_path),
            "logic_tokens": logic_tokens,
            "scores": scores,
            "policy_hashes": policy_hashes,
            "edge_value": result.aggregate_components.edge_score,
            "activity_value": result.aggregate_components.activity_score,
            "productive_value": result.aggregate_components.productive_frequency_score,
            "pairs": pairs,
        }
