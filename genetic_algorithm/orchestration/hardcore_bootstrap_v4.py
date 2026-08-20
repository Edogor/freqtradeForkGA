"""Strictly replay historical v3 seeds into a v4 niche archive.

The historical point metrics are used only to choose a bounded replay
shortlist.  Every emitted archive entry comes from a new immutable six-pair
strict replay under the current v4 config and code manifest.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from genetic_algorithm.config.schema import load_config
from genetic_algorithm.core.strategy_gene import StrategyGene
from genetic_algorithm.evaluation.panel_contract import phenotype_fingerprint
from genetic_algorithm.evaluation.raw_multipair_score import (
    PairScenario,
    RawMultiPairPanel,
    RawMultiPairPolicyV4,
    score_raw_multipair,
)
from genetic_algorithm.orchestration.data_manifest_v2 import resolve_spot_data_root
from genetic_algorithm.orchestration.evolution_worker_v2 import (
    EvolutionWorkerError,
    FrozenEvolutionSeedV2,
    prepare_evolution_worker,
    run_evolution_worker,
    validate_evolution_seed,
)
from genetic_algorithm.orchestration.hardcore_backend_v1 import (
    candidate_snapshot_from_strict_replay,
)
from genetic_algorithm.orchestration.hardcore_campaign_v1 import (
    HARDCORE_PANEL_PAIRS,
    RAW_MULTIPAIR_SCORE_VERSION,
    ArchiveEntryV1,
    BootstrapQuarantineV1,
    CampaignLane,
    CandidateSnapshotV1,
    HardcoreBootstrapArchiveV4,
    assign_candidate_niches,
    write_hardcore_bootstrap_archive,
)
from genetic_algorithm.orchestration.manifest_builder_v2 import (
    build_attempt_manifest_bundle,
)
from genetic_algorithm.orchestration.promotion_policy_v2 import (
    shadow_gate_policy_from_config,
)
from genetic_algorithm.orchestration.result_contract import AttemptStatus


@dataclass(frozen=True)
class _HistoricalCandidate:
    lane: CampaignLane
    candidate_id: str
    evolutionary_phenotype_hash: str
    provisional_score: float
    provisional_edge: float
    provisional_activity: float
    seed: FrozenEvolutionSeedV2


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scenario_from_historical(
    metric: dict[str, Any], *, panel: RawMultiPairPanel
) -> PairScenario:
    trades = int(metric.get("trade_count", 0))
    active = int(metric.get("active_months", 0))
    return PairScenario(
        pair=str(metric["pair"]),
        timeframe=panel.timeframe,
        success=True,
        trade_count=trades,
        active_months=active,
        period_start=panel.period_start,
        period_end=panel.period_end,
        net_return=metric.get("net_return"),
        net_expectancy=metric.get("net_expectancy"),
        profit_factor=metric.get("profit_factor"),
        profit_factor_censored=metric.get("profit_factor_censored"),
        median_holding_hours=metric.get("median_holding_hours"),
        p90_holding_hours=metric.get("p90_holding_hours"),
        max_drawdown=metric.get("max_drawdown"),
        max_drawdown_duration_days=metric.get("max_drawdown_duration_days"),
        max_consecutive_losses=metric.get("max_consecutive_losses"),
    )


def _historical_candidates(  # noqa: C901 - fail closed at each legacy boundary
    source_root: Path,
    *,
    lane: CampaignLane,
    config: dict[str, Any],
) -> tuple[
    list[_HistoricalCandidate], list[BootstrapQuarantineV1], list[str]
]:
    raw_policy = RawMultiPairPolicyV4.from_mapping(
        config.get("raw_multipair_score", {}).get("policy")
    )
    panel = RawMultiPairPanel(timeframe=lane.value, policy=raw_policy)
    by_phenotype: dict[str, _HistoricalCandidate] = {}
    quarantined: list[BootstrapQuarantineV1] = []
    outcome_hashes: list[str] = []
    for outcome_path in sorted(source_root.glob("runs/*/evolution_outcome.json")):
        checksum = outcome_path.with_name(outcome_path.name + ".sha256")
        if not checksum.is_file() or _sha256_file(outcome_path) != checksum.read_text(
            encoding="ascii"
        ).strip():
            continue
        outcome_hashes.append(_sha256_file(outcome_path))
        try:
            outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for raw in outcome.get("evidence", {}).get("candidates", []):
            if raw.get("timeframe") != lane.value:
                continue
            candidate_id = str(raw.get("candidate_id") or "unknown")
            try:
                metrics = raw["pair_metrics"]
                if [item["pair"] for item in metrics] != list(HARDCORE_PANEL_PAIRS):
                    raise ValueError("historical pair panel is incomplete or unordered")
                scored = score_raw_multipair(
                    panel,
                    [_scenario_from_historical(item, panel=panel) for item in metrics],
                )
                if not scored.is_valid or scored.score is None:
                    raise ValueError(
                        f"historical v4 rescore invalid: {scored.reason_code}"
                    )
                aggregate = scored.aggregate_components
                if aggregate is None:
                    raise ValueError("historical v4 rescore lacks aggregate components")
                seed_path = Path(raw["evolution_seed_path"]).resolve()
                if (
                    not seed_path.is_file()
                    or _sha256_file(seed_path) != raw["evolution_seed_sha256"]
                ):
                    raise ValueError("historical seed is missing or hash-mismatched")
                seed = FrozenEvolutionSeedV2.model_validate_json(seed_path.read_bytes())
                if (
                    seed.candidate_id != candidate_id
                    or seed.phenotype_hash != raw["phenotype_hash"]
                    or seed.gene_hash != raw["gene_hash"]
                ):
                    raise ValueError("historical seed identity differs from outcome")
                validate_evolution_seed(seed, config)
                candidate = _HistoricalCandidate(
                    lane=lane,
                    candidate_id=candidate_id,
                    evolutionary_phenotype_hash=phenotype_fingerprint(
                        StrategyGene.from_dict_exact(seed.strategy_gene)
                    ),
                    provisional_score=float(scored.score),
                    provisional_edge=float(aggregate.edge_score),
                    provisional_activity=float(aggregate.activity_score),
                    seed=seed,
                )
            except (KeyError, TypeError, ValueError, EvolutionWorkerError) as exc:
                quarantined.append(
                    BootstrapQuarantineV1(
                        lane=lane,
                        candidate_id=candidate_id,
                        reason_code="INCOMPATIBLE_HISTORICAL_SEED",
                        detail=str(exc)[:1000],
                    )
                )
                continue
            previous = by_phenotype.get(candidate.evolutionary_phenotype_hash)
            if previous is None or candidate.provisional_score > previous.provisional_score:
                by_phenotype[candidate.evolutionary_phenotype_hash] = candidate
    return list(by_phenotype.values()), quarantined, sorted(set(outcome_hashes))


def _shortlist(candidates: list[_HistoricalCandidate]) -> list[_HistoricalCandidate]:
    selected: list[_HistoricalCandidate] = []
    hashes: set[str] = set()
    rankings = (
        lambda item: (item.provisional_score, item.provisional_edge),
        lambda item: (item.provisional_edge, item.provisional_score),
        lambda item: (item.provisional_activity, item.provisional_score),
    )
    for key in rankings:
        added = 0
        for candidate in sorted(candidates, key=key, reverse=True):
            if candidate.evolutionary_phenotype_hash in hashes:
                continue
            selected.append(candidate)
            hashes.add(candidate.evolutionary_phenotype_hash)
            added += 1
            if added == 4:
                break
    return selected


def _niche_archive(
    candidates: list[CandidateSnapshotV1], *, source_run_id: str, now: datetime
) -> list[ArchiveEntryV1]:
    return [
        ArchiveEntryV1(
            candidate=candidate,
            niche=niche,
            descriptor_score=descriptor,
            source_run_id=source_run_id,
            archived_at=now,
        )
        for niche, candidate, descriptor in assign_candidate_niches(candidates)
    ]


def build_strict_v4_bootstrap_archive(
    *,
    source_root: str | Path,
    output_path: str | Path,
    repo_root: str | Path,
    config_15m: str | Path,
    config_1h: str | Path,
    created_at: datetime | None = None,
) -> HardcoreBootstrapArchiveV4:
    """Rescore, shortlist, then strictly replay both historical lanes."""

    source = Path(source_root).resolve()
    output = Path(output_path).resolve()
    repository = Path(repo_root).resolve()
    now = created_at or datetime.now(UTC)
    if now.utcoffset() is None:
        raise ValueError("bootstrap timestamp must be timezone-aware")
    if not source.is_dir() or not repository.is_dir():
        raise ValueError("bootstrap source and repository must exist")
    lane_configs = {
        CampaignLane.FIFTEEN_MINUTES: Path(config_15m).resolve(),
        CampaignLane.ONE_HOUR: Path(config_1h).resolve(),
    }
    lanes: dict[CampaignLane, list[ArchiveEntryV1]] = {}
    all_quarantined: list[BootstrapQuarantineV1] = []
    all_outcome_hashes: list[str] = []
    replay_root = output.parent / "strict_replay"
    ledger = output.parent / "bootstrap_final_test_usage.sqlite3"
    for lane, config_path in lane_configs.items():
        config = load_config(config_path)
        historical, quarantined, outcome_hashes = _historical_candidates(
            source,
            lane=lane,
            config=config,
        )
        shortlist = _shortlist(historical)
        all_quarantined.extend(quarantined)
        all_outcome_hashes.extend(outcome_hashes)
        if not shortlist:
            raise ValueError(f"no compatible historical {lane.value} seeds to replay")
        provisional = {
            item.candidate_id: item.provisional_score for item in shortlist
        }
        token = hashlib.sha256(
            (str(source) + lane.value + RAW_MULTIPAIR_SCORE_VERSION).encode("utf-8")
        ).hexdigest()[:20]
        attempt_id = f"hardcore-v4-bootstrap-{lane.value}-{token}"
        worker_root = replay_root / lane.value / "worker"
        shadow_policy = shadow_gate_policy_from_config(config)
        data_root = repository / resolve_spot_data_root(
            config, list(HARDCORE_PANEL_PAIRS)
        )
        bundle = build_attempt_manifest_bundle(
            attempt_id=attempt_id,
            wave_id="hardcore-v4-bootstrap",
            experiment_id=f"historical-v3-{lane.value}",
            created_at=now,
            resolved_config=config,
            policy=shadow_policy,
            fitness_policy_version=RAW_MULTIPAIR_SCORE_VERSION,
            seeds=[42],
            worker_count=4,
            artifact_root=worker_root,
            repo_root=repository,
            data_root=data_root,
        )
        prepared = prepare_evolution_worker(
            bundle=bundle,
            resolved_config=config,
            policy=shadow_policy,
            seeds=[item.seed for item in shortlist],
            repo_root=repository,
            final_test_ledger_path=ledger,
            created_at=now,
            top_n=len(shortlist),
        )

        def replay_only(
            _config_path: Path,
            individuals,
            provisional_scores=provisional,
        ):
            for individual in individuals:
                candidate_id = individual.metrics["archive_candidate_id"]
                individual.set_fitness(
                    provisional_scores[candidate_id],
                    {
                        **individual.metrics,
                        "bootstrap_replay_only": True,
                    },
                )
            return individuals

        result = run_evolution_worker(
            prepared.spec_path,
            expected_spec_sha256=prepared.spec_file_sha256,
            evolution_runner=replay_only,
        )
        if result.status is not AttemptStatus.SUCCEEDED:
            raise RuntimeError(
                f"strict bootstrap replay failed for {lane.value}: "
                f"{result.error_code}: {result.error_detail}"
            )
        raw_policy = RawMultiPairPolicyV4.from_mapping(
            config.get("raw_multipair_score", {}).get("policy")
        )
        strict_failures: list[str] = []
        snapshots = [
            snapshot
            for candidate in result.candidate_evaluations
            if (
                snapshot := candidate_snapshot_from_strict_replay(
                    candidate,
                    lane=lane,
                    worker_root=worker_root,
                    policy=raw_policy,
                    strict_failure_codes=strict_failures,
                )
            )
            is not None
        ]
        for index, failure in enumerate(strict_failures):
            all_quarantined.append(
                BootstrapQuarantineV1(
                    lane=lane,
                    candidate_id=f"strict-replay-failure-{index + 1}",
                    reason_code="STRICT_REPLAY_INVALID",
                    detail=failure,
                )
            )
        if not snapshots:
            raise RuntimeError(f"strict bootstrap replay produced no valid {lane.value} seed")
        lanes[lane] = _niche_archive(
            snapshots,
            source_run_id=f"historical-v3-strict-replay-{lane.value}",
            now=now,
        )
    archive = HardcoreBootstrapArchiveV4(
        created_at=now,
        source_root=str(source),
        source_outcome_sha256=sorted(set(all_outcome_hashes)),
        lanes=lanes,
        quarantined=all_quarantined,
    )
    write_hardcore_bootstrap_archive(output, archive)
    return archive
