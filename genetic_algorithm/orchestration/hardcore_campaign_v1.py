"""Seven-day, search-only dual-lane multi-pair campaign orchestration.

This module deliberately does not extend :mod:`automation_controller_v2`.
The old wave lineage, promotion gates and artifacts are immutable historical
evidence.  Hardcore campaigns use a separate root, a separate state file and
one evolution attempt per run.  An execution backend supplies hash-bound
attempt evidence; this controller owns lane alternation, retry/suspension,
archives, recipes and the final ``evolution_outcome.json`` decision artifact.

The controller is intentionally unaware of LCB/UCB, ESS, Sharpe, Sortino and
legacy promotion gates.  Its only quality input is ``raw-multipair-score-v1``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import Field, model_validator

from genetic_algorithm.config.schema import load_config
from genetic_algorithm.evaluation.raw_multipair_score import RawMultiPairPanel
from genetic_algorithm.orchestration.artifact_store_v2 import canonical_config_hash
from genetic_algorithm.orchestration.result_contract import StrictV2Model


RAW_MULTIPAIR_SCORE_VERSION = "raw-multipair-score-v1"
HARDCORE_CAMPAIGN_VERSION = "hardcore-multipair-campaign-v1"
HARDCORE_PANEL_PAIRS = (
    "BTC/USDT",
    "SOL/USDT",
    "XRP/USDT",
    "BNB/USDT",
    "ETH/USDT",
    "PEPE/USDT",
)
HARDCORE_DEV_PAIRS = ("BTC/USDT", "SOL/USDT", "XRP/USDT")
HARDCORE_VALIDATION_PAIRS = ("BNB/USDT", "ETH/USDT", "PEPE/USDT")
_ZERO_HASH = "0" * 64
_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]+$")


class HardcoreCampaignError(RuntimeError):
    """Raised when the controller cannot prove a safe state transition."""


class HardcoreQueueError(HardcoreCampaignError):
    """Classified failure before an immutable worker could enter the queue."""

    def __init__(
        self,
        message: str,
        *,
        failure_class: "FailureClass",
        error_code: str,
    ) -> None:
        super().__init__(message)
        self.failure_class = failure_class
        self.error_code = error_code


class CampaignLane(StrEnum):
    FIFTEEN_MINUTES = "15m"
    ONE_HOUR = "1h"

    @property
    def other(self) -> "CampaignLane":
        return (
            CampaignLane.ONE_HOUR
            if self == CampaignLane.FIFTEEN_MINUTES
            else CampaignLane.FIFTEEN_MINUTES
        )


class SearchRecipe(StrEnum):
    BALANCED = "balanced"
    EXPLORE = "explore"
    RECOMBINE = "recombine"

    @property
    def next_after_no_progress(self) -> "SearchRecipe":
        return {
            SearchRecipe.BALANCED: SearchRecipe.EXPLORE,
            SearchRecipe.EXPLORE: SearchRecipe.RECOMBINE,
            SearchRecipe.RECOMBINE: SearchRecipe.BALANCED,
        }[self]


class EvolutionStopReason(StrEnum):
    COMPLETED_BUDGET = "COMPLETED_BUDGET"
    PLATEAU = "PLATEAU"
    TECHNICAL_INVALID = "TECHNICAL_INVALID"
    DIVERSITY_COLLAPSE = "DIVERSITY_COLLAPSE"
    CAMPAIGN_DEADLINE = "CAMPAIGN_DEADLINE"
    NO_GLOBAL_PROGRESS = "NO_GLOBAL_PROGRESS"


class FailureClass(StrEnum):
    NONE = "NONE"
    TRANSIENT = "TRANSIENT"
    DETERMINISTIC_CONFIG = "DETERMINISTIC_CONFIG"
    DETERMINISTIC_DATA = "DETERMINISTIC_DATA"

    @property
    def suspends_lane(self) -> bool:
        return self in {
            FailureClass.DETERMINISTIC_CONFIG,
            FailureClass.DETERMINISTIC_DATA,
        }


class AttemptPollStatus(StrEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"


class LaneLifecycle(StrEnum):
    ACTIVE = "ACTIVE"
    SUSPENDED = "SUSPENDED"


class CampaignLifecycle(StrEnum):
    RUNNING = "RUNNING"
    DEADLINE_DRAINING = "DEADLINE_DRAINING"
    RESOURCE_DRAINING = "RESOURCE_DRAINING"
    KILL_DRAINING = "KILL_DRAINING"
    COMPLETED = "COMPLETED"
    BLOCKED = "BLOCKED"
    KILLED = "KILLED"


class ControllerTickOutcome(StrEnum):
    WAITING = "WAITING"
    PROGRESSED = "PROGRESSED"
    STOPPED = "STOPPED"
    BLOCKED = "BLOCKED"


class PairRole(StrEnum):
    DEVELOPMENT = "DEVELOPMENT"
    VALIDATION = "VALIDATION"


def _canonical_bytes(model: StrictV2Model) -> bytes:
    return (
        json.dumps(
            model.model_dump(mode="json", exclude_none=False),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_aware(value: datetime, name: str) -> None:
    if value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")


def material_improvement(previous: float | None, candidate: float) -> bool:
    """Return the campaign's single, continuous progress decision."""

    if not math.isfinite(candidate):
        raise ValueError("candidate score must be finite")
    if previous is None:
        return True
    if not math.isfinite(previous):
        raise ValueError("previous score must be finite")
    required = max(0.25, 0.005 * max(1.0, abs(previous)))
    return candidate - previous >= required


def _atomic_replace(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_immutable(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise HardcoreCampaignError(f"immutable artifact differs: {path}")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise HardcoreCampaignError(f"concurrent artifact differs: {path}")
    finally:
        temporary.unlink(missing_ok=True)


class RawPairMetricsV1(StrictV2Model):
    """Report-facing raw evidence for one independent pair replay."""

    pair: str = Field(min_length=3)
    role: PairRole
    net_return: float | None = None
    net_expectancy: float | None = None
    profit_factor: float | None = Field(default=None, ge=0)
    profit_factor_censored: bool | None = None
    trade_count: int = Field(ge=0)
    active_months: int = Field(ge=0)
    calendar_months: float = Field(gt=0)
    median_holding_hours: float | None = Field(default=None, ge=0)
    p90_holding_hours: float | None = Field(default=None, ge=0)
    max_drawdown: float | None = Field(default=None, ge=0)
    max_drawdown_duration_days: float | None = Field(default=None, ge=0)
    max_consecutive_losses: int | None = Field(default=None, ge=0)
    normalized_components: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _canonical_pair(self) -> "RawPairMetricsV1":
        if self.pair not in HARDCORE_PANEL_PAIRS:
            raise ValueError(f"pair is not in the fixed hardcore panel: {self.pair}")
        expected_role = (
            PairRole.DEVELOPMENT
            if self.pair in HARDCORE_DEV_PAIRS
            else PairRole.VALIDATION
        )
        if self.role != expected_role:
            raise ValueError("pair role differs from the fixed development/validation split")
        if self.trade_count == 0 and any(
            value is not None
            for value in (
                self.median_holding_hours,
                self.p90_holding_hours,
                self.net_return,
                self.net_expectancy,
                self.profit_factor,
                self.max_drawdown,
                self.max_drawdown_duration_days,
                self.max_consecutive_losses,
            )
        ):
            raise ValueError("zero-trade evidence must keep trade metrics absent")
        if self.trade_count == 0:
            if self.active_months != 0 or self.profit_factor_censored not in (None, False):
                raise ValueError("zero-trade evidence has invalid activity/censor state")
        elif any(
            value is None
            for value in (
                self.net_return,
                self.net_expectancy,
                self.profit_factor,
                self.profit_factor_censored,
                self.median_holding_hours,
                self.p90_holding_hours,
                self.max_drawdown,
                self.max_drawdown_duration_days,
                self.max_consecutive_losses,
            )
        ):
            raise ValueError("positive-trade evidence is missing raw metrics")
        required_components = {"R", "E", "P", "A", "H", "D", "U", "L", "O"}
        if set(self.normalized_components) != required_components:
            raise ValueError("normalized pair components must be exactly R,E,P,A,H,D,U,L,O")
        return self


class CandidateSnapshotV1(StrictV2Model):
    """A scored executable phenotype and the seed needed for warm-starting."""

    candidate_id: str = Field(min_length=1)
    phenotype_hash: str = Field(min_length=64, max_length=64)
    score_version: Literal["raw-multipair-score-v1"] = RAW_MULTIPAIR_SCORE_VERSION
    score: float
    timeframe: CampaignLane
    panel_id: str = Field(min_length=1)
    evolution_seed_path: str = Field(min_length=1)
    evolution_seed_sha256: str = Field(min_length=64, max_length=64)
    gene_hash: str = Field(min_length=64, max_length=64)
    pair_metrics: list[RawPairMetricsV1] = Field(min_length=6, max_length=6)

    @model_validator(mode="after")
    def _complete_panel(self) -> "CandidateSnapshotV1":
        if not _SAFE_ID.fullmatch(self.candidate_id):
            raise ValueError("candidate_id contains unsafe characters")
        if not Path(self.evolution_seed_path).is_absolute():
            raise ValueError("evolution_seed_path must be absolute")
        pairs = [item.pair for item in self.pair_metrics]
        if len(set(pairs)) != len(pairs) or set(pairs) != set(HARDCORE_PANEL_PAIRS):
            raise ValueError("candidate must contain exactly one metric row for all six pairs")
        if pairs != list(HARDCORE_PANEL_PAIRS):
            raise ValueError("candidate pair metrics must use canonical panel order")
        expected_panel = RawMultiPairPanel(timeframe=self.timeframe.value).panel_id
        if self.panel_id != expected_panel:
            raise ValueError("candidate panel_id differs from the raw-score timeframe panel")
        return self


class DiversityEventV1(StrictV2Model):
    generation: int = Field(ge=1, le=30)
    event_type: Literal["COLLAPSE_OBSERVED", "RECOVERY_APPLIED", "RECOVERY_FAILED"]
    affected_islands: list[str] = Field(min_length=1)
    duplicate_fraction: float = Field(ge=0, le=1)
    genetic_diversity: float = Field(ge=0, le=1)
    mutation_rate_after: float | None = Field(default=None, ge=0, le=1)


class AttemptEvidenceV1(StrictV2Model):
    """Hash-bound worker output consumed by the campaign controller."""

    schema_version: Literal["1.0"] = "1.0"
    evidence_version: Literal["hardcore-attempt-evidence-v1"] = (
        "hardcore-attempt-evidence-v1"
    )
    campaign_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    request_hash: str = Field(min_length=64, max_length=64)
    lane: CampaignLane
    recipe: SearchRecipe
    attempt_ordinal: int = Field(ge=0, le=1)
    started_at: datetime
    finished_at: datetime
    stop_reason: EvolutionStopReason
    actual_generations: int = Field(ge=0, le=30)
    score_curve: list[float] = Field(default_factory=list)
    valid_evaluations: int = Field(ge=0)
    failed_evaluations: int = Field(ge=0)
    search_valid_evaluations: int = Field(default=0, ge=0)
    search_failed_evaluations: int = Field(default=0, ge=0)
    common_panel_valid_evaluations: int = Field(default=0, ge=0)
    common_panel_failed_evaluations: int = Field(default=0, ge=0)
    strict_replay_valid_evaluations: int = Field(default=0, ge=0)
    strict_replay_failed_evaluations: int = Field(default=0, ge=0)
    plateau_checks: int = Field(ge=0, le=4)
    candidates: list[CandidateSnapshotV1] = Field(default_factory=list, max_length=12)
    diversity_events: list[DiversityEventV1] = Field(default_factory=list)
    failure_class: FailureClass = FailureClass.NONE
    error_code: str | None = None
    error_detail: str | None = None
    checkpoint_path: str | None = None
    checkpoint_sha256: str | None = Field(default=None, min_length=64, max_length=64)

    @model_validator(mode="after")
    def _coherent_attempt(self) -> "AttemptEvidenceV1":
        _require_aware(self.started_at, "started_at")
        _require_aware(self.finished_at, "finished_at")
        if self.finished_at < self.started_at:
            raise ValueError("attempt finished before it started")
        if (self.checkpoint_path is None) != (self.checkpoint_sha256 is None):
            raise ValueError("checkpoint path and SHA-256 must be set together")
        if self.checkpoint_path is not None and not Path(self.checkpoint_path).is_absolute():
            raise ValueError("checkpoint_path must be absolute")
        technical = self.stop_reason == EvolutionStopReason.TECHNICAL_INVALID
        if technical:
            if self.failure_class == FailureClass.NONE or not self.error_code:
                raise ValueError("technical invalid evidence requires classified error data")
        elif self.failure_class != FailureClass.NONE or self.error_code is not None:
            raise ValueError("economic/normal stop cannot carry a technical failure class")
        if not technical and (not self.candidates or self.valid_evaluations == 0):
            raise ValueError("non-technical evolution requires valid six-pair evidence")
        for candidate in self.candidates:
            if candidate.timeframe != self.lane:
                raise ValueError("candidate timeframe differs from attempt lane")
        scores = [item.score for item in self.candidates]
        if scores != sorted(scores, reverse=True):
            raise ValueError("attempt candidates must be ordered by descending raw score")
        phenotypes = [item.phenotype_hash for item in self.candidates]
        if len(phenotypes) != len(set(phenotypes)):
            raise ValueError("attempt candidate phenotypes must be unique")
        if len(self.score_curve) > self.actual_generations:
            raise ValueError("score curve cannot exceed completed generations")
        detailed_valid = (
            self.search_valid_evaluations
            + self.common_panel_valid_evaluations
            + self.strict_replay_valid_evaluations
        )
        detailed_failed = (
            self.search_failed_evaluations
            + self.common_panel_failed_evaluations
            + self.strict_replay_failed_evaluations
        )
        if detailed_valid > self.valid_evaluations:
            raise ValueError("detailed valid-evaluation counts exceed the total")
        if detailed_failed > self.failed_evaluations:
            raise ValueError("detailed failed-evaluation counts exceed the total")
        return self

    @property
    def evidence_hash(self) -> str:
        return _sha256_bytes(_canonical_bytes(self))


class ArchiveChangeV1(StrictV2Model):
    previous_global_score: float | None = None
    resulting_global_score: float | None = None
    material_improvement: bool
    added_phenotype_hashes: list[str] = Field(default_factory=list)
    evicted_phenotype_hashes: list[str] = Field(default_factory=list)


class EvolutionOutcomeV1(StrictV2Model):
    """Final controller-owned, SHA-256-covered result for one evolution run."""

    schema_version: Literal["1.0"] = "1.0"
    outcome_version: Literal["hardcore-evolution-outcome-v1"] = (
        "hardcore-evolution-outcome-v1"
    )
    evidence: AttemptEvidenceV1
    evidence_hash: str = Field(min_length=64, max_length=64)
    previous_outcome_sha256: str = Field(min_length=64, max_length=64)
    archive_change: ArchiveChangeV1
    finalized_at: datetime

    @model_validator(mode="after")
    def _hash_bound(self) -> "EvolutionOutcomeV1":
        _require_aware(self.finalized_at, "finalized_at")
        if self.evidence.evidence_hash != self.evidence_hash:
            raise ValueError("attempt evidence hash differs from content")
        if self.finalized_at < self.evidence.finished_at:
            raise ValueError("outcome finalization precedes attempt completion")
        if not self.archive_change.material_improvement:
            if self.archive_change.added_phenotype_hashes:
                raise ValueError("archive cannot add candidates without material progress")
            if self.archive_change.resulting_global_score != (
                self.archive_change.previous_global_score
            ):
                raise ValueError("non-improving outcome cannot change global score")
        return self


class OutcomeArtifactReceiptV1(StrictV2Model):
    path: str
    sha256_path: str
    outcome_sha256: str = Field(min_length=64, max_length=64)


def write_evolution_outcome(
    run_root: str | Path, outcome: EvolutionOutcomeV1
) -> OutcomeArtifactReceiptV1:
    """Atomically persist an immutable outcome plus detached SHA-256."""

    root = Path(run_root).resolve()
    payload = _canonical_bytes(outcome)
    digest = _sha256_bytes(payload)
    path = root / "evolution_outcome.json"
    digest_path = root / "evolution_outcome.json.sha256"
    _write_immutable(path, payload)
    _write_immutable(digest_path, (digest + "\n").encode("ascii"))
    return OutcomeArtifactReceiptV1(
        path=str(path), sha256_path=str(digest_path), outcome_sha256=digest
    )


def read_evolution_outcome(path: str | Path) -> EvolutionOutcomeV1:
    artifact = Path(path).resolve()
    checksum = artifact.with_name(artifact.name + ".sha256")
    if not artifact.is_file() or not checksum.is_file():
        raise HardcoreCampaignError("evolution outcome or checksum is missing")
    expected = checksum.read_text(encoding="ascii").strip()
    if len(expected) != 64 or _sha256_file(artifact) != expected:
        raise HardcoreCampaignError("evolution outcome checksum mismatch")
    try:
        return EvolutionOutcomeV1.model_validate_json(artifact.read_bytes())
    except ValueError as exc:
        raise HardcoreCampaignError("evolution outcome violates its schema") from exc


class ArchiveEntryV1(StrictV2Model):
    candidate: CandidateSnapshotV1
    source_run_id: str = Field(min_length=1)
    archived_at: datetime

    @model_validator(mode="after")
    def _aware(self) -> "ArchiveEntryV1":
        _require_aware(self.archived_at, "archived_at")
        return self


class IslandBlueprintV1(StrictV2Model):
    name: str = Field(min_length=1)
    family: Literal["momentum", "trend", "volatility-volume", "mixed"]
    variant: int = Field(ge=1, le=3)
    seed: int = Field(ge=0, lt=2**32)
    archive_island: bool
    population_size: int = Field(ge=3, le=12)
    generations: int = Field(ge=1, le=30)
    indicator_pool: list[str] = Field(min_length=1)
    pairs: list[str] = Field(min_length=6, max_length=6)
    seed_candidates: list[ArchiveEntryV1] = Field(default_factory=list, max_length=4)

    @model_validator(mode="after")
    def _fixed_panel(self) -> "IslandBlueprintV1":
        if self.pairs != list(HARDCORE_PANEL_PAIRS):
            raise ValueError("every specialist island must see the fixed six-pair panel")
        if self.seed_candidates and not self.archive_island:
            raise ValueError("fresh islands cannot receive archive candidates")
        return self


class RecipeParametersV1(StrictV2Model):
    mutation_rate: float
    crossover_rate: float
    indicator_overlap: float


_RECIPE_PARAMETERS: dict[SearchRecipe, RecipeParametersV1] = {
    SearchRecipe.BALANCED: RecipeParametersV1(
        mutation_rate=0.20, crossover_rate=0.75, indicator_overlap=0.35
    ),
    SearchRecipe.EXPLORE: RecipeParametersV1(
        mutation_rate=0.30, crossover_rate=0.70, indicator_overlap=0.20
    ),
    SearchRecipe.RECOMBINE: RecipeParametersV1(
        mutation_rate=0.25, crossover_rate=0.85, indicator_overlap=0.50
    ),
}


class EvolutionRunRequestV1(StrictV2Model):
    schema_version: Literal["1.0"] = "1.0"
    request_version: Literal["hardcore-evolution-request-v1"] = (
        "hardcore-evolution-request-v1"
    )
    campaign_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    run_sequence: int = Field(ge=1)
    lane: CampaignLane
    recipe: SearchRecipe
    recipe_parameters: RecipeParametersV1
    attempt_ordinal: int = Field(ge=0, le=1)
    created_at: datetime
    campaign_deadline: datetime
    max_attempt_runtime_seconds: Literal[36000] = 36000
    config_path: str = Field(min_length=1)
    config_file_sha256: str = Field(min_length=64, max_length=64)
    resolved_config_sha256: str = Field(min_length=64, max_length=64)
    search_seed: int = Field(ge=0, lt=2**32)
    panel_id: str = Field(min_length=1)
    score_version: Literal["raw-multipair-score-v1"] = RAW_MULTIPAIR_SCORE_VERSION
    previous_global_score: float | None = None
    islands: list[IslandBlueprintV1] = Field(min_length=12, max_length=12)
    resume_checkpoint_path: str | None = None
    resume_checkpoint_sha256: str | None = Field(
        default=None, min_length=64, max_length=64
    )

    @model_validator(mode="after")
    def _fixed_production_shape(self) -> "EvolutionRunRequestV1":
        _require_aware(self.created_at, "created_at")
        _require_aware(self.campaign_deadline, "campaign_deadline")
        if self.created_at >= self.campaign_deadline:
            raise ValueError("run request cannot be created at/after campaign deadline")
        if not Path(self.config_path).is_absolute():
            raise ValueError("config_path must be absolute")
        if self.recipe_parameters != _RECIPE_PARAMETERS[self.recipe]:
            raise ValueError("recipe parameters differ from immutable recipe definition")
        if self.panel_id != RawMultiPairPanel(timeframe=self.lane.value).panel_id:
            raise ValueError("request panel_id differs from the raw-score timeframe panel")
        if (self.resume_checkpoint_path is None) != (
            self.resume_checkpoint_sha256 is None
        ):
            raise ValueError("resume checkpoint path/hash must be set together")
        if self.attempt_ordinal == 0 and self.resume_checkpoint_path is not None:
            raise ValueError("initial attempt cannot declare a retry checkpoint")
        if self.attempt_ordinal == 1 and self.resume_checkpoint_path is None:
            raise ValueError("retry must be bound to a verified checkpoint")
        names = [item.name for item in self.islands]
        if len(names) != len(set(names)):
            raise ValueError("island names must be unique")
        if sum(item.archive_island for item in self.islands) != 3:
            raise ValueError("exactly three islands must be archive-enabled")
        families = [(item.family, item.variant) for item in self.islands]
        expected = {
            (family, variant)
            for family in ("momentum", "trend", "volatility-volume", "mixed")
            for variant in (1, 2, 3)
        }
        if set(families) != expected:
            raise ValueError("islands must be four specialist families x three variants")
        shapes = {(item.population_size, item.generations) for item in self.islands}
        if len(shapes) != 1:
            raise ValueError("all twelve islands must share one population/generation shape")
        seeded = [
            entry.candidate.phenotype_hash
            for island in self.islands
            for entry in island.seed_candidates
        ]
        if len(seeded) != len(set(seeded)):
            raise ValueError("an archived phenotype can seed only one island")
        if any(
            entry.candidate.timeframe != self.lane
            for island in self.islands
            for entry in island.seed_candidates
        ):
            raise ValueError("cross-timeframe archive seeding is forbidden")
        return self

    @property
    def request_hash(self) -> str:
        return _sha256_bytes(_canonical_bytes(self))


class AttemptHandleV1(StrictV2Model):
    attempt_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    request_hash: str = Field(min_length=64, max_length=64)
    artifact_root: str = Field(min_length=1)

    @model_validator(mode="after")
    def _absolute_root(self) -> "AttemptHandleV1":
        if not Path(self.artifact_root).is_absolute():
            raise ValueError("attempt artifact_root must be absolute")
        return self


class AttemptPollV1(StrictV2Model):
    status: AttemptPollStatus
    observed_at: datetime
    current_generation: int = Field(default=0, ge=0, le=30)
    current_score: float | None = None
    current_pair_metrics: list[RawPairMetricsV1] = Field(default_factory=list)
    plateau_checks: int = Field(default=0, ge=0, le=4)
    evidence: AttemptEvidenceV1 | None = None

    @model_validator(mode="after")
    def _terminal_evidence(self) -> "AttemptPollV1":
        _require_aware(self.observed_at, "observed_at")
        if (self.status == AttemptPollStatus.FINISHED) != (self.evidence is not None):
            raise ValueError("FINISHED poll and attempt evidence must be present together")
        return self


class HardcoreAttemptBackend(Protocol):
    """Adapter boundary for the existing immutable V2 worker queue."""

    def queue(self, request: EvolutionRunRequestV1) -> AttemptHandleV1: ...

    def poll(self, handle: AttemptHandleV1) -> AttemptPollV1: ...

    def request_graceful_stop(
        self, handle: AttemptHandleV1, *, reason: EvolutionStopReason
    ) -> None: ...


class HardcoreCampaignPolicyV1(StrictV2Model):
    schema_version: Literal["1.0"] = "1.0"
    policy_version: Literal["hardcore-multipair-policy-v1"] = (
        "hardcore-multipair-policy-v1"
    )
    campaign_id: str = Field(min_length=1)
    campaign_mode: Literal["production", "canary"] = "production"
    max_completed_runs: Literal[2] | None = None
    automation_root: str = Field(min_length=1)
    lane_configs: dict[CampaignLane, str]
    runtime_seconds: Literal[604800] = 7 * 24 * 60 * 60
    max_attempt_runtime_seconds: Literal[36000] = 10 * 60 * 60
    max_artifact_bytes: Literal[26843545600] = 25 * 1024**3
    min_free_disk_bytes: Literal[21474836480] = 20 * 1024**3
    min_available_memory_bytes: Literal[2147483648] = 2 * 1024**3
    max_concurrent: Literal[1] = 1
    archive_size: Literal[12] = 12
    archive_islands: Literal[3] = 3
    fresh_islands: Literal[9] = 9
    max_transient_retries: Literal[1] = 1
    poll_interval_seconds: float = Field(default=5.0, gt=0)
    kill_switch_path: str = Field(min_length=1)

    @model_validator(mode="after")
    def _isolated_paths(self) -> "HardcoreCampaignPolicyV1":
        if not _SAFE_ID.fullmatch(self.campaign_id):
            raise ValueError("campaign_id contains unsafe characters")
        root = Path(self.automation_root)
        kill = Path(self.kill_switch_path)
        if not root.is_absolute() or not kill.is_absolute():
            raise ValueError("automation and kill-switch paths must be absolute")
        if root not in kill.parents:
            raise ValueError("kill switch must be isolated below automation_root")
        if set(self.lane_configs) != set(CampaignLane):
            raise ValueError("policy must declare exactly one config per timeframe lane")
        if any(not Path(path).is_absolute() for path in self.lane_configs.values()):
            raise ValueError("lane config paths must be absolute")
        if (self.campaign_mode == "canary") != (self.max_completed_runs == 2):
            raise ValueError("canary mode must stop after exactly two finalized runs")
        return self

    @property
    def policy_hash(self) -> str:
        return canonical_config_hash(self.model_dump(mode="json"))


def default_hardcore_campaign_policy(
    *,
    campaign_id: str,
    automation_root: str | Path,
    config_15m: str | Path,
    config_1h: str | Path,
    canary: bool = False,
) -> HardcoreCampaignPolicyV1:
    root = Path(automation_root).resolve()
    return HardcoreCampaignPolicyV1(
        campaign_id=campaign_id,
        campaign_mode="canary" if canary else "production",
        max_completed_runs=2 if canary else None,
        automation_root=str(root),
        lane_configs={
            CampaignLane.FIFTEEN_MINUTES: str(Path(config_15m).resolve()),
            CampaignLane.ONE_HOUR: str(Path(config_1h).resolve()),
        },
        kill_switch_path=str(root / "STOP_HARDCORE_CAMPAIGN"),
    )


class HardcoreCampaignPreflightV1(StrictV2Model):
    schema_version: Literal["1.0"] = "1.0"
    checked_at: datetime
    ready: bool
    campaign_id: str
    policy_hash: str = Field(min_length=64, max_length=64)
    repo_root: str
    automation_root: str
    state_path: str
    git_head: str | None = None
    git_upstream: str | None = None
    git_upstream_head: str | None = None
    worktree_clean: bool
    head_matches_upstream: bool
    config_hashes: dict[CampaignLane, str]
    reason_codes: list[str] = Field(min_length=1)


def _git_output(repo_root: Path, *arguments: str) -> tuple[int, str]:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return completed.returncode, completed.stdout.strip()


def _hardcore_warmup_coverage_errors(
    config: dict[str, Any],
    *,
    repository: Path,
    lane: CampaignLane,
) -> list[str]:
    """Verify every fixed pair has the exact panel cells plus warmup candles."""

    try:
        import pandas as pd
        from freqtrade.configuration import TimeRange
        from freqtrade.data.history import load_pair_history

        backtesting = config["backtesting"]
        constraints = config["strategy_constraints"]
        timerange = TimeRange.parse_timerange(str(backtesting["timerange"]))
        expected_start = datetime.fromtimestamp(timerange.startts, tz=UTC)
        expected_stop = datetime.fromtimestamp(timerange.stopts, tz=UTC)
        startup = int(constraints["startup_candle_cap"])
        configured_datadir = backtesting.get("datadir") or "user_data/data/binance"
        datadir = Path(configured_datadir)
        if not datadir.is_absolute():
            datadir = repository / datadir
        data_format = backtesting.get("dataformat_ohlcv") or "feather"
    except Exception:
        return [f"{lane.value.upper()}_WARMUP_PREFLIGHT_INVALID"]

    errors: list[str] = []
    for pair in backtesting.get("pairs", []):
        token = str(pair).replace("/", "_").replace(":", "_").upper()
        try:
            frame = load_pair_history(
                pair=pair,
                timeframe=backtesting["timeframe"],
                datadir=datadir,
                timerange=timerange,
                startup_candles=startup,
                data_format=data_format,
            )
            dates = pd.to_datetime(frame["date"], utc=True)
            if int((dates < expected_start).sum()) < startup:
                errors.append(
                    f"{lane.value.upper()}_WARMUP_INSUFFICIENT_{token}"
                )
            if not bool((dates == expected_start).any()) or not bool(
                (dates == expected_stop).any()
            ):
                errors.append(
                    f"{lane.value.upper()}_PANEL_COVERAGE_INCOMPLETE_{token}"
                )
        except Exception:
            errors.append(f"{lane.value.upper()}_DATA_UNREADABLE_{token}")
    return errors


def build_hardcore_campaign_preflight(
    policy: HardcoreCampaignPolicyV1,
    *,
    repo_root: str | Path,
    state_path: str | Path,
    checked_at: datetime | None = None,
) -> HardcoreCampaignPreflightV1:
    """Prove the exact pushed commit and isolated production inputs."""

    now = checked_at or datetime.now(UTC)
    _require_aware(now, "checked_at")
    repository = Path(repo_root).resolve()
    root = Path(policy.automation_root).resolve()
    state = Path(state_path).resolve()
    reasons: list[str] = []
    config_hashes: dict[CampaignLane, str] = {}
    if not repository.is_dir() or not (repository / ".git").exists():
        reasons.append("REPO_ROOT_NOT_GIT_WORKTREE")
    if state.parent != root:
        reasons.append("STATE_DB_NOT_ISOLATED_BELOW_AUTOMATION_ROOT")
    legacy_root = (repository / "genetic_algorithm/data/v2/automation").resolve()
    if root == legacy_root:
        reasons.append("LEGACY_AUTOMATION_ROOT_REUSE_FORBIDDEN")
    if any((root / name).exists() for name in ("bootstrap_intent.json", "bootstrap_receipt.json")):
        reasons.append("LEGACY_AUTOMATION_ARTIFACTS_PRESENT")
    for lane, raw_path in policy.lane_configs.items():
        path = Path(raw_path).resolve()
        try:
            config = load_config(path)
        except (OSError, ValueError) as exc:
            reasons.append(f"{lane.value.upper()}_CONFIG_INVALID_{type(exc).__name__.upper()}")
            continue
        if (
            config.get("safety_profile", {}).get("name")
            != "hardcore_multipair_v1"
            or config.get("backtesting", {}).get("timeframe") != lane.value
        ):
            reasons.append(f"{lane.value.upper()}_CONFIG_CONTRACT_MISMATCH")
        config_is_canary = bool(config.get("safety_profile", {}).get("canary", False))
        if config_is_canary != (policy.campaign_mode == "canary"):
            reasons.append(f"{lane.value.upper()}_CONFIG_MODE_MISMATCH")
        config_hashes[lane] = canonical_config_hash(config)
        reasons.extend(
            _hardcore_warmup_coverage_errors(
                config,
                repository=repository,
                lane=lane,
            )
        )

    git_head = None
    git_upstream = None
    git_upstream_head = None
    worktree_clean = False
    head_matches = False
    if repository.is_dir():
        status_code, status = _git_output(
            repository, "status", "--porcelain", "--untracked-files=all"
        )
        worktree_clean = status_code == 0 and not status
        if not worktree_clean:
            reasons.append("GIT_WORKTREE_NOT_CLEAN")
        head_code, git_head_value = _git_output(repository, "rev-parse", "HEAD")
        if head_code == 0 and len(git_head_value) == 40:
            git_head = git_head_value
        else:
            reasons.append("GIT_HEAD_UNRESOLVED")
        upstream_code, upstream_value = _git_output(
            repository,
            "rev-parse",
            "--abbrev-ref",
            "--symbolic-full-name",
            "@{upstream}",
        )
        if upstream_code == 0 and upstream_value:
            git_upstream = upstream_value
            upstream_head_code, upstream_head_value = _git_output(
                repository, "rev-parse", "@{upstream}"
            )
            if upstream_head_code == 0 and len(upstream_head_value) == 40:
                git_upstream_head = upstream_head_value
        if git_upstream is None or git_upstream_head is None:
            reasons.append("GIT_UPSTREAM_UNRESOLVED")
        head_matches = bool(git_head and git_head == git_upstream_head)
        if not head_matches:
            reasons.append("GIT_HEAD_NOT_PUSHED_TO_UPSTREAM")
    if not reasons:
        reasons.append("PREFLIGHT_READY")
    return HardcoreCampaignPreflightV1(
        checked_at=now,
        ready=reasons == ["PREFLIGHT_READY"],
        campaign_id=policy.campaign_id,
        policy_hash=policy.policy_hash,
        repo_root=str(repository),
        automation_root=str(root),
        state_path=str(state),
        git_head=git_head,
        git_upstream=git_upstream,
        git_upstream_head=git_upstream_head,
        worktree_clean=worktree_clean,
        head_matches_upstream=head_matches,
        config_hashes=config_hashes,
        reason_codes=sorted(set(reasons)),
    )


def require_hardcore_campaign_preflight(
    policy: HardcoreCampaignPolicyV1,
    *,
    repo_root: str | Path,
    state_path: str | Path,
) -> HardcoreCampaignPreflightV1:
    report = build_hardcore_campaign_preflight(
        policy, repo_root=repo_root, state_path=state_path
    )
    if not report.ready:
        raise HardcoreCampaignError(
            "hardcore campaign preflight failed: " + ", ".join(report.reason_codes)
        )
    return report


class OutcomeRecordV1(StrictV2Model):
    run_id: str
    lane: CampaignLane
    stop_reason: EvolutionStopReason
    outcome_path: str
    outcome_sha256: str = Field(min_length=64, max_length=64)
    completed_at: datetime
    material_improvement: bool


class LaneStateV1(StrictV2Model):
    lane: CampaignLane
    lifecycle: LaneLifecycle = LaneLifecycle.ACTIVE
    suspension_reason: str | None = None
    recipe: SearchRecipe = SearchRecipe.BALANCED
    runs_started: int = Field(default=0, ge=0)
    runs_completed: int = Field(default=0, ge=0)
    global_champion_score: float | None = None
    archive: list[ArchiveEntryV1] = Field(default_factory=list, max_length=12)
    champion_history: list[ArchiveEntryV1] = Field(default_factory=list)

    @model_validator(mode="after")
    def _archive_contract(self) -> "LaneStateV1":
        if self.lifecycle == LaneLifecycle.SUSPENDED and not self.suspension_reason:
            raise ValueError("suspended lane requires a reason")
        phenotypes = [item.candidate.phenotype_hash for item in self.archive]
        if len(phenotypes) != len(set(phenotypes)):
            raise ValueError("lane archive phenotype hashes must be unique")
        scores = [item.candidate.score for item in self.archive]
        if scores != sorted(scores, reverse=True):
            raise ValueError("lane archive must be score ordered")
        if any(item.candidate.timeframe != self.lane for item in self.archive):
            raise ValueError("lane archive contains a cross-timeframe candidate")
        panel_ids = {item.candidate.panel_id for item in self.archive}
        if len(panel_ids) > 1:
            raise ValueError("lane archive mixes raw-score panels")
        if self.archive:
            if self.global_champion_score != self.archive[0].candidate.score:
                raise ValueError("global champion score differs from archive head")
        elif self.global_champion_score is not None:
            raise ValueError("empty archive cannot declare a global champion")
        return self


class ActiveRunV1(StrictV2Model):
    request: EvolutionRunRequestV1
    handle: AttemptHandleV1

    @model_validator(mode="after")
    def _consistent_handle(self) -> "ActiveRunV1":
        if self.handle.run_id != self.request.run_id:
            raise ValueError("attempt handle belongs to another run")
        if self.handle.request_hash != self.request.request_hash:
            raise ValueError("attempt handle request hash differs")
        return self


class CampaignStateV1(StrictV2Model):
    schema_version: Literal["1.0"] = "1.0"
    state_version: Literal["hardcore-campaign-state-v1"] = (
        "hardcore-campaign-state-v1"
    )
    campaign_id: str
    policy_hash: str = Field(min_length=64, max_length=64)
    started_at: datetime
    deadline: datetime
    updated_at: datetime
    lifecycle: CampaignLifecycle
    stop_reason: str | None = None
    next_lane: CampaignLane
    run_sequence: int = Field(default=0, ge=0)
    lanes: dict[CampaignLane, LaneStateV1]
    active_run: ActiveRunV1 | None = None
    previous_outcome_sha256: str = Field(
        default=_ZERO_HASH, min_length=64, max_length=64
    )
    outcomes: list[OutcomeRecordV1] = Field(default_factory=list)

    @model_validator(mode="after")
    def _coherent_campaign(self) -> "CampaignStateV1":
        for field_name in ("started_at", "deadline", "updated_at"):
            _require_aware(getattr(self, field_name), field_name)
        if self.deadline != self.started_at + timedelta(days=7):
            raise ValueError("campaign deadline must be exactly seven days after start")
        if self.updated_at < self.started_at:
            raise ValueError("campaign update precedes start")
        if set(self.lanes) != set(CampaignLane):
            raise ValueError("campaign state must contain both timeframe lanes")
        if any(key != lane.lane for key, lane in self.lanes.items()):
            raise ValueError("lane state key differs from lane identity")
        counts = [lane.runs_started for lane in self.lanes.values()]
        both_active = all(
            lane.lifecycle == LaneLifecycle.ACTIVE for lane in self.lanes.values()
        )
        if both_active and abs(counts[0] - counts[1]) > 1:
            raise ValueError("strict alternation permits at most one-run lane imbalance")
        if self.run_sequence != sum(counts):
            raise ValueError("run sequence differs from total started runs")
        if self.active_run is not None:
            active_lane = self.active_run.request.lane
            if self.lanes[active_lane].lifecycle != LaneLifecycle.ACTIVE:
                raise ValueError("active run belongs to a suspended lane")
        if self.lifecycle in {
            CampaignLifecycle.COMPLETED,
            CampaignLifecycle.BLOCKED,
            CampaignLifecycle.KILLED,
        } and self.active_run is not None:
            raise ValueError("terminal campaign cannot retain an active run")
        return self


class ControllerTickV1(StrictV2Model):
    observed_at: datetime
    outcome: ControllerTickOutcome
    lifecycle: CampaignLifecycle
    reason_codes: list[str] = Field(min_length=1)
    active_run_id: str | None = None
    active_lane: CampaignLane | None = None


class RuntimeResourcesV1(StrictV2Model):
    artifact_bytes: int = Field(ge=0)
    free_disk_bytes: int = Field(ge=0)
    available_memory_bytes: int = Field(ge=0)


class CampaignStatusV1(StrictV2Model):
    schema_version: Literal["1.0"] = "1.0"
    campaign_id: str
    observed_at: datetime
    lifecycle: CampaignLifecycle
    stop_reason: str | None = None
    active_lane: CampaignLane | None = None
    active_run_id: str | None = None
    active_generation: int | None = None
    current_score: float | None = None
    current_pair_metrics: list[RawPairMetricsV1] = Field(default_factory=list)
    global_champion_score: float | None = None
    global_champion_pair_metrics: list[RawPairMetricsV1] = Field(default_factory=list)
    lane_champion_scores: dict[CampaignLane, float | None]
    lane_champion_pair_metrics: dict[CampaignLane, list[RawPairMetricsV1]]
    plateau_checks: int | None = None
    next_timeframe: CampaignLane
    elapsed_seconds: float = Field(ge=0)
    deadline: datetime
    artifact_bytes: int = Field(ge=0)
    free_disk_bytes: int = Field(ge=0)
    available_memory_bytes: int = Field(ge=0)
    lane_run_counts: dict[CampaignLane, int]


class LaneFinalReportV1(StrictV2Model):
    lane: CampaignLane
    runs_started: int
    runs_completed: int
    suspended: bool
    suspension_reason: str | None
    champion_label: Literal["SEARCH_CHAMPION"] | None
    champion: CandidateSnapshotV1 | None
    champion_history: list[ArchiveEntryV1]
    outcomes: list[OutcomeRecordV1]


class CampaignFinalReportV1(StrictV2Model):
    schema_version: Literal["1.0"] = "1.0"
    report_version: Literal["hardcore-final-report-v1"] = "hardcore-final-report-v1"
    campaign_id: str
    started_at: datetime
    finished_at: datetime
    stop_reason: str
    score_version: Literal["raw-multipair-score-v1"] = RAW_MULTIPAIR_SCORE_VERSION
    live_ready: Literal[False] = False
    lanes: dict[CampaignLane, LaneFinalReportV1]


_INDICATOR_POOLS: dict[str, list[str]] = {
    "momentum": ["RSI", "MACD", "STOCH", "CCI", "MFI", "ROC", "WILLR"],
    "trend": ["EMA", "SMA", "TEMA", "KAMA", "SUPERTREND", "AROON", "PSAR"],
    "volatility-volume": [
        "BBANDS",
        "ATR",
        "DONCHIAN",
        "OBV",
        "CMF",
        "VROC",
        "VWAP",
    ],
    "mixed": ["RSI", "MACD", "EMA", "SUPERTREND", "BBANDS", "ATR", "CMF", "VWAP"],
}


def _recipe_indicator_pool(
    *,
    family: str,
    variant: int,
    recipe: SearchRecipe,
    search_seed: int,
) -> list[str]:
    """Combine a stable family core with deterministic cross-family overlap."""

    core = list(_INDICATOR_POOLS[family])
    universe = sorted(
        {
            indicator
            for values in _INDICATOR_POOLS.values()
            for indicator in values
        }
    )
    foreign = [indicator for indicator in universe if indicator not in core]
    overlap = _RECIPE_PARAMETERS[recipe].indicator_overlap
    foreign_count = min(len(foreign), max(1, round(len(core) * overlap)))
    digest = hashlib.sha256(
        f"{family}:{variant}:{recipe.value}:{search_seed}".encode("utf-8")
    ).digest()
    offset = int.from_bytes(digest[:4], "big") % len(foreign)
    rotated = foreign[offset:] + foreign[:offset]
    return [*core, *rotated[:foreign_count]]


def build_island_blueprints(
    *,
    lane: CampaignLane,
    search_seed: int,
    archive: Sequence[ArchiveEntryV1],
    recipe: SearchRecipe = SearchRecipe.BALANCED,
    population_size: int = 12,
    generations: int = 30,
) -> list[IslandBlueprintV1]:
    """Create twelve deterministic specialists with three archive islands."""

    if len(archive) > 12:
        raise ValueError("lane archive exceeds twelve entries")
    if any(item.candidate.timeframe != lane for item in archive):
        raise ValueError("cross-timeframe archive input is forbidden")
    unique = [item.candidate.phenotype_hash for item in archive]
    if len(unique) != len(set(unique)):
        raise ValueError("archive input contains duplicate phenotypes")
    assignments: list[list[ArchiveEntryV1]] = [[], [], []]
    for index, entry in enumerate(archive):
        # Match GenericIslandModelEvolution.set_archive_initial_seeds(), which
        # deterministically distributes its flat seed list round-robin.
        assignments[index % 3].append(entry)
    result: list[IslandBlueprintV1] = []
    ordinal = 0
    for family in ("momentum", "trend", "volatility-volume", "mixed"):
        for variant in (1, 2, 3):
            archive_island = ordinal < 3
            result.append(
                IslandBlueprintV1(
                    name=f"{family}-{variant}",
                    family=family,
                    variant=variant,
                    seed=(search_seed + ordinal) % (2**32),
                    archive_island=archive_island,
                    population_size=population_size,
                    generations=generations,
                    indicator_pool=_recipe_indicator_pool(
                        family=family,
                        variant=variant,
                        recipe=recipe,
                        search_seed=search_seed,
                    ),
                    pairs=list(HARDCORE_PANEL_PAIRS),
                    seed_candidates=assignments[ordinal] if archive_island else [],
                )
            )
            ordinal += 1
    return result


def _available_memory_bytes() -> int:
    try:
        with Path("/proc/meminfo").open(encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return 0
    return 0


class HardcoreCampaignControllerV1:
    """Persistently supervise one strictly alternating 15m/1h campaign."""

    STATE_NAME = "campaign_state.json"
    STATUS_NAME = "campaign_status.json"
    FINAL_REPORT_NAME = "campaign_final_report.json"

    def __init__(
        self,
        *,
        policy: HardcoreCampaignPolicyV1,
        backend: HardcoreAttemptBackend,
        started_at: datetime | None = None,
        resource_probe: Callable[[], RuntimeResourcesV1] | None = None,
    ) -> None:
        self.policy = policy
        self.backend = backend
        self._resource_probe = resource_probe
        self.root = Path(policy.automation_root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / self.STATE_NAME
        if self.state_path.exists():
            self.state = self._read_mutable_model(self.state_path, CampaignStateV1)
            if self.state.policy_hash != policy.policy_hash:
                raise HardcoreCampaignError("existing campaign state has another policy")
        else:
            now = started_at or datetime.now(UTC)
            _require_aware(now, "started_at")
            self.state = CampaignStateV1(
                campaign_id=policy.campaign_id,
                policy_hash=policy.policy_hash,
                started_at=now,
                deadline=now + timedelta(seconds=policy.runtime_seconds),
                updated_at=now,
                lifecycle=CampaignLifecycle.RUNNING,
                next_lane=CampaignLane.FIFTEEN_MINUTES,
                lanes={lane: LaneStateV1(lane=lane) for lane in CampaignLane},
            )
            self._persist_state()

    @staticmethod
    def _read_mutable_model(path: Path, model_type):
        checksum = path.with_name(path.name + ".sha256")
        if not checksum.is_file():
            raise HardcoreCampaignError(f"state checksum is missing: {checksum}")
        expected = checksum.read_text(encoding="ascii").strip()
        if _sha256_file(path) != expected:
            raise HardcoreCampaignError(f"state checksum differs: {path}")
        return model_type.model_validate_json(path.read_bytes())

    def _persist_state(self) -> None:
        payload = _canonical_bytes(self.state)
        _atomic_replace(self.state_path, payload)
        _atomic_replace(
            self.state_path.with_name(self.state_path.name + ".sha256"),
            (_sha256_bytes(payload) + "\n").encode("ascii"),
        )

    def _artifact_bytes(self) -> int:
        total = 0
        for path in self.root.rglob("*"):
            if path.is_file():
                try:
                    total += path.stat().st_size
                except OSError:
                    continue
        return total

    def _resources(self) -> RuntimeResourcesV1:
        if self._resource_probe is not None:
            return self._resource_probe()
        probe = self.root
        return RuntimeResourcesV1(
            artifact_bytes=self._artifact_bytes(),
            free_disk_bytes=shutil.disk_usage(probe).free,
            available_memory_bytes=_available_memory_bytes(),
        )

    def _search_seed(self, run_sequence: int, lane: CampaignLane) -> int:
        digest = hashlib.sha256(
            f"{self.policy.campaign_id}:{run_sequence}:{lane.value}".encode()
        ).digest()
        return int.from_bytes(digest[:4], "big")

    def _build_request(
        self,
        lane: CampaignLane,
        *,
        now: datetime,
        run_id: str | None = None,
        attempt_ordinal: int = 0,
        checkpoint_path: str | None = None,
        checkpoint_sha256: str | None = None,
    ) -> EvolutionRunRequestV1:
        lane_state = self.state.lanes[lane]
        sequence = self.state.run_sequence + (1 if attempt_ordinal == 0 else 0)
        identifier = run_id or (
            f"{self.policy.campaign_id}-run-{sequence:06d}-{lane.value}"
        )
        config_path = Path(self.policy.lane_configs[lane]).resolve()
        if not config_path.is_file():
            raise HardcoreCampaignError(f"lane config does not exist: {config_path}")
        resolved_config = load_config(config_path)
        configured_islands = resolved_config.get("generic_island_model", {}).get(
            "islands", []
        )
        if len(configured_islands) != 12:
            raise HardcoreCampaignError("lane config must resolve to twelve islands")
        island_shapes = {
            (int(item["population_size"]), int(item["generations"]))
            for item in configured_islands
        }
        if len(island_shapes) != 1:
            raise HardcoreCampaignError("lane islands do not share one execution shape")
        population_size, generations = island_shapes.pop()
        seed = self._search_seed(sequence, lane)
        return EvolutionRunRequestV1(
            campaign_id=self.policy.campaign_id,
            run_id=identifier,
            run_sequence=sequence,
            lane=lane,
            recipe=lane_state.recipe,
            recipe_parameters=_RECIPE_PARAMETERS[lane_state.recipe],
            attempt_ordinal=attempt_ordinal,
            created_at=now,
            campaign_deadline=self.state.deadline,
            config_path=str(config_path),
            config_file_sha256=_sha256_file(config_path),
            resolved_config_sha256=canonical_config_hash(resolved_config),
            search_seed=seed,
            panel_id=RawMultiPairPanel(timeframe=lane.value).panel_id,
            previous_global_score=lane_state.global_champion_score,
            islands=build_island_blueprints(
                lane=lane,
                search_seed=seed,
                archive=lane_state.archive,
                recipe=lane_state.recipe,
                population_size=population_size,
                generations=generations,
            ),
            resume_checkpoint_path=checkpoint_path,
            resume_checkpoint_sha256=checkpoint_sha256,
        )

    def _queue_request(self, request: EvolutionRunRequestV1) -> ActiveRunV1:
        request_root = (
            self.root
            / "runs"
            / request.run_id
            / f"attempt-{request.attempt_ordinal}"
        )
        _write_immutable(request_root / "run_request.json", _canonical_bytes(request))
        _write_immutable(
            request_root / "run_request.json.sha256",
            (request.request_hash + "\n").encode("ascii"),
        )
        handle = self.backend.queue(request)
        active = ActiveRunV1(request=request, handle=handle)
        self.state.active_run = active
        return active

    def _start_next_run(self, now: datetime) -> ControllerTickV1:
        canary_terminal = self._finish_canary_if_ready(now)
        if canary_terminal is not None:
            return canary_terminal
        lane = self.state.next_lane
        if self.state.lanes[lane].lifecycle == LaneLifecycle.SUSPENDED:
            lane = lane.other
        if self.state.lanes[lane].lifecycle == LaneLifecycle.SUSPENDED:
            self._finish(now, CampaignLifecycle.BLOCKED, "BOTH_LANES_SUSPENDED")
            return self._tick(now, ControllerTickOutcome.BLOCKED, ["BOTH_LANES_SUSPENDED"])
        try:
            request = self._build_request(lane, now=now)
        except (OSError, ValueError, HardcoreCampaignError) as exc:
            error_code = f"REQUEST_BUILD_{type(exc).__name__.upper()}"
            lane_state = self.state.lanes[lane]
            lane_state.suspension_reason = error_code
            lane_state.lifecycle = LaneLifecycle.SUSPENDED
            self.state.next_lane = lane.other
            self.state.updated_at = now
            if all(
                item.lifecycle == LaneLifecycle.SUSPENDED
                for item in self.state.lanes.values()
            ):
                self._finish(now, CampaignLifecycle.BLOCKED, "BOTH_LANES_SUSPENDED")
                return self._tick(
                    now, ControllerTickOutcome.BLOCKED, ["BOTH_LANES_SUSPENDED"]
                )
            self._persist_state()
            self._persist_status(now)
            return self._tick(
                now,
                ControllerTickOutcome.PROGRESSED,
                ["LANE_SUSPENDED_PREQUEUE", error_code],
            )
        self.state.lanes[lane].runs_started += 1
        # Nested lane assignment is validated first so the campaign-wide
        # run-sequence invariant is never transiently broken under Pydantic's
        # validate-assignment semantics.
        self.state.run_sequence += 1
        try:
            self._queue_request(request)
        except HardcoreQueueError as exc:
            evidence = AttemptEvidenceV1(
                campaign_id=request.campaign_id,
                run_id=request.run_id,
                request_hash=request.request_hash,
                lane=request.lane,
                recipe=request.recipe,
                attempt_ordinal=request.attempt_ordinal,
                started_at=now,
                finished_at=now,
                stop_reason=EvolutionStopReason.TECHNICAL_INVALID,
                actual_generations=0,
                valid_evaluations=0,
                failed_evaluations=1,
                plateau_checks=0,
                failure_class=exc.failure_class,
                error_code=exc.error_code,
                error_detail=str(exc)[:2000],
            )
            self._finalize_evidence(evidence, now=now)
            lane_state = self.state.lanes[lane]
            lane_state.runs_completed += 1
            lane_state.recipe = lane_state.recipe.next_after_no_progress
            if exc.failure_class.suspends_lane:
                lane_state.suspension_reason = exc.error_code
                lane_state.lifecycle = LaneLifecycle.SUSPENDED
            self.state.active_run = None
            self.state.next_lane = lane.other
            self.state.updated_at = now
            canary_terminal = self._finish_canary_if_ready(now)
            if canary_terminal is not None:
                return canary_terminal
            if all(
                item.lifecycle == LaneLifecycle.SUSPENDED
                for item in self.state.lanes.values()
            ):
                self._finish(now, CampaignLifecycle.BLOCKED, "BOTH_LANES_SUSPENDED")
                return self._tick(
                    now, ControllerTickOutcome.BLOCKED, ["BOTH_LANES_SUSPENDED"]
                )
            self._persist_state()
            self._persist_status(now)
            return self._tick(
                now,
                ControllerTickOutcome.PROGRESSED,
                ["QUEUE_FAILURE_RECORDED", exc.failure_class.value],
            )
        self.state.updated_at = now
        self._persist_state()
        self._persist_status(now)
        return self._tick(now, ControllerTickOutcome.PROGRESSED, ["RUN_QUEUED"])

    def _archive_change(
        self, lane_state: LaneStateV1, evidence: AttemptEvidenceV1, now: datetime
    ) -> ArchiveChangeV1:
        previous = lane_state.global_champion_score
        if (
            evidence.stop_reason == EvolutionStopReason.TECHNICAL_INVALID
            or not evidence.candidates
        ):
            return ArchiveChangeV1(
                previous_global_score=previous,
                resulting_global_score=previous,
                material_improvement=False,
            )
        best = evidence.candidates[0]
        improved = material_improvement(previous, best.score)
        if not improved:
            return ArchiveChangeV1(
                previous_global_score=previous,
                resulting_global_score=previous,
                material_improvement=False,
            )
        old = {item.candidate.phenotype_hash: item for item in lane_state.archive}
        merged = dict(old)
        for candidate in evidence.candidates:
            current = merged.get(candidate.phenotype_hash)
            if current is None or candidate.score > current.candidate.score:
                merged[candidate.phenotype_hash] = ArchiveEntryV1(
                    candidate=candidate,
                    source_run_id=evidence.run_id,
                    archived_at=now,
                )
        new_archive = sorted(
            merged.values(), key=lambda item: item.candidate.score, reverse=True
        )[: self.policy.archive_size]
        new_hashes = {item.candidate.phenotype_hash for item in new_archive}
        old_hashes = set(old)
        added = sorted(new_hashes - old_hashes)
        evicted = sorted(old_hashes - new_hashes)
        history = [
            *lane_state.champion_history,
            ArchiveEntryV1(
                candidate=best, source_run_id=evidence.run_id, archived_at=now
            ),
        ]
        updated_lane = lane_state.model_copy(
            update={
                "archive": new_archive,
                "global_champion_score": new_archive[0].candidate.score,
                "champion_history": history,
            }
        )
        # Revalidate the whole lane atomically before placing it in state.
        updated_lane = LaneStateV1.model_validate(updated_lane.model_dump(mode="python"))
        self.state.lanes[evidence.lane] = updated_lane
        return ArchiveChangeV1(
            previous_global_score=previous,
            resulting_global_score=updated_lane.global_champion_score,
            material_improvement=True,
            added_phenotype_hashes=added,
            evicted_phenotype_hashes=evicted,
        )

    def _finalize_evidence(
        self, evidence: AttemptEvidenceV1, *, now: datetime
    ) -> OutcomeArtifactReceiptV1:
        lane_state = self.state.lanes[evidence.lane]
        archive_change = self._archive_change(lane_state, evidence, now)
        outcome = EvolutionOutcomeV1(
            evidence=evidence,
            evidence_hash=evidence.evidence_hash,
            previous_outcome_sha256=self.state.previous_outcome_sha256,
            archive_change=archive_change,
            finalized_at=now,
        )
        run_root = self.root / "runs" / evidence.run_id
        receipt = write_evolution_outcome(run_root, outcome)
        self.state.previous_outcome_sha256 = receipt.outcome_sha256
        self.state.outcomes.append(
            OutcomeRecordV1(
                run_id=evidence.run_id,
                lane=evidence.lane,
                stop_reason=evidence.stop_reason,
                outcome_path=receipt.path,
                outcome_sha256=receipt.outcome_sha256,
                completed_at=now,
                material_improvement=archive_change.material_improvement,
            )
        )
        return receipt

    def _normalize_budget_stop_reason(
        self, evidence: AttemptEvidenceV1
    ) -> AttemptEvidenceV1:
        """Make an exhausted budget explicit when it did not advance the lane."""

        if (
            evidence.stop_reason != EvolutionStopReason.COMPLETED_BUDGET
            or not evidence.candidates
        ):
            return evidence
        previous = self.state.lanes[evidence.lane].global_champion_score
        if material_improvement(previous, evidence.candidates[0].score):
            return evidence
        payload = evidence.model_dump(mode="python")
        payload["stop_reason"] = EvolutionStopReason.NO_GLOBAL_PROGRESS
        return AttemptEvidenceV1.model_validate(payload)

    def _finish_canary_if_ready(
        self, now: datetime
    ) -> ControllerTickV1 | None:
        """Pass the two-run canary only with two verified six-pair results."""

        limit = self.policy.max_completed_runs
        if limit is None or len(self.state.outcomes) < limit:
            return None

        failure_reason = None
        if len(self.state.outcomes) != 2:
            failure_reason = "CANARY_FAILED_OUTCOME_COUNT"
        elif {item.lane for item in self.state.outcomes} != {
            CampaignLane.FIFTEEN_MINUTES,
            CampaignLane.ONE_HOUR,
        }:
            failure_reason = "CANARY_FAILED_LANE_COVERAGE"

        if failure_reason is None:
            for record in self.state.outcomes:
                if record.stop_reason == EvolutionStopReason.TECHNICAL_INVALID:
                    failure_reason = "CANARY_FAILED_TECHNICAL_OUTCOME"
                    break
                try:
                    outcome_path = Path(record.outcome_path)
                    if _sha256_file(outcome_path) != record.outcome_sha256:
                        raise HardcoreCampaignError(
                            "recorded outcome hash differs from artifact"
                        )
                    outcome = read_evolution_outcome(outcome_path)
                    evidence = outcome.evidence
                    if (
                        evidence.stop_reason == EvolutionStopReason.TECHNICAL_INVALID
                        or evidence.valid_evaluations == 0
                        or not evidence.candidates
                    ):
                        raise HardcoreCampaignError(
                            "canary outcome lacks valid raw six-pair candidates"
                        )
                    for candidate in evidence.candidates:
                        seed_path = Path(candidate.evolution_seed_path)
                        if (
                            not seed_path.is_file()
                            or _sha256_file(seed_path)
                            != candidate.evolution_seed_sha256
                        ):
                            raise HardcoreCampaignError(
                                "canary candidate seed hash verification failed"
                            )
                except (OSError, ValueError, HardcoreCampaignError):
                    failure_reason = "CANARY_FAILED_ARTIFACT_VERIFICATION"
                    break

        if failure_reason is not None:
            self._finish(now, CampaignLifecycle.BLOCKED, failure_reason)
            return self._tick(
                now,
                ControllerTickOutcome.BLOCKED,
                [failure_reason],
            )

        self._finish(now, CampaignLifecycle.COMPLETED, "CANARY_COMPLETED")
        return self._tick(
            now,
            ControllerTickOutcome.STOPPED,
            ["CANARY_COMPLETED"],
        )

    def _retry_transient(
        self, evidence: AttemptEvidenceV1, *, now: datetime
    ) -> tuple[ControllerTickV1 | None, AttemptEvidenceV1]:
        if (
            self.state.lifecycle != CampaignLifecycle.RUNNING
            or evidence.stop_reason != EvolutionStopReason.TECHNICAL_INVALID
            or evidence.failure_class != FailureClass.TRANSIENT
            or evidence.attempt_ordinal >= self.policy.max_transient_retries
        ):
            return None, evidence
        if evidence.checkpoint_path is None or evidence.checkpoint_sha256 is None:
            return None, evidence
        request = self._build_request(
            evidence.lane,
            now=now,
            run_id=evidence.run_id,
            attempt_ordinal=evidence.attempt_ordinal + 1,
            checkpoint_path=evidence.checkpoint_path,
            checkpoint_sha256=evidence.checkpoint_sha256,
        )
        # Retry keeps the original immutable run sequence and search seed.
        request = request.model_copy(update={"run_sequence": self.state.run_sequence})
        try:
            self._queue_request(request)
        except HardcoreQueueError as exc:
            # The retry opportunity has been consumed. Finalize the original
            # run with the classified queue failure so a restart cannot create
            # a new, timestamp-different retry request for the same attempt.
            payload = evidence.model_dump(mode="python")
            payload.update(
                {
                    "failure_class": exc.failure_class,
                    "error_code": f"RETRY_{exc.error_code}",
                    "error_detail": str(exc)[:2000],
                    "checkpoint_path": None,
                    "checkpoint_sha256": None,
                }
            )
            return None, AttemptEvidenceV1.model_validate(payload)
        self.state.updated_at = now
        self._persist_state()
        self._persist_status(now)
        return (
            self._tick(
                now, ControllerTickOutcome.PROGRESSED, ["TRANSIENT_RETRY_QUEUED"]
            ),
            evidence,
        )

    def _complete_active(self, evidence: AttemptEvidenceV1, *, now: datetime) -> ControllerTickV1:
        active = self.state.active_run
        if active is None:
            raise HardcoreCampaignError("attempt evidence arrived without an active run")
        request = active.request
        if (
            evidence.campaign_id != self.state.campaign_id
            or evidence.run_id != request.run_id
            or evidence.request_hash != request.request_hash
            or evidence.lane != request.lane
            or evidence.recipe != request.recipe
            or evidence.attempt_ordinal != request.attempt_ordinal
        ):
            raise HardcoreCampaignError("attempt evidence differs from active run request")
        retry_tick, evidence = self._retry_transient(evidence, now=now)
        if retry_tick is not None:
            return retry_tick

        evidence = self._normalize_budget_stop_reason(evidence)
        self._finalize_evidence(evidence, now=now)
        # Finalization may atomically replace the lane while updating archive
        # invariants; always reacquire it before recording lifecycle changes.
        lane_state = self.state.lanes[evidence.lane]
        lane_state.runs_completed += 1
        improved = self.state.outcomes[-1].material_improvement
        lane_state.recipe = (
            SearchRecipe.BALANCED
            if improved
            else lane_state.recipe.next_after_no_progress
        )
        if evidence.failure_class.suspends_lane:
            lane_state.suspension_reason = str(evidence.error_code)
            lane_state.lifecycle = LaneLifecycle.SUSPENDED
        self.state.active_run = None
        self.state.next_lane = evidence.lane.other
        self.state.updated_at = now

        if self.state.lifecycle == CampaignLifecycle.DEADLINE_DRAINING:
            self._finish(now, CampaignLifecycle.COMPLETED, "SEVEN_DAY_DEADLINE")
            return self._tick(now, ControllerTickOutcome.STOPPED, ["SEVEN_DAY_DEADLINE"])
        if self.state.lifecycle == CampaignLifecycle.KILL_DRAINING:
            self._finish(now, CampaignLifecycle.KILLED, "KILL_SWITCH_PRESENT")
            return self._tick(now, ControllerTickOutcome.STOPPED, ["KILL_SWITCH_PRESENT"])
        if self.state.lifecycle == CampaignLifecycle.RESOURCE_DRAINING:
            reason = self.state.stop_reason or "RESOURCE_LIMIT_REACHED"
            self._finish(now, CampaignLifecycle.COMPLETED, reason)
            return self._tick(now, ControllerTickOutcome.STOPPED, [reason])
        canary_terminal = self._finish_canary_if_ready(now)
        if canary_terminal is not None:
            return canary_terminal
        if all(
            item.lifecycle == LaneLifecycle.SUSPENDED
            for item in self.state.lanes.values()
        ):
            self._finish(now, CampaignLifecycle.BLOCKED, "BOTH_LANES_SUSPENDED")
            return self._tick(now, ControllerTickOutcome.BLOCKED, ["BOTH_LANES_SUSPENDED"])
        self._persist_state()
        self._persist_status(now)
        return self._tick(now, ControllerTickOutcome.PROGRESSED, ["RUN_FINALIZED"])

    def _tick(
        self,
        now: datetime,
        outcome: ControllerTickOutcome,
        reasons: list[str],
    ) -> ControllerTickV1:
        active = self.state.active_run
        return ControllerTickV1(
            observed_at=now,
            outcome=outcome,
            lifecycle=self.state.lifecycle,
            reason_codes=reasons,
            active_run_id=active.request.run_id if active else None,
            active_lane=active.request.lane if active else None,
        )

    def _finish(
        self, now: datetime, lifecycle: CampaignLifecycle, reason: str
    ) -> None:
        self.state.active_run = None
        self.state.lifecycle = lifecycle
        self.state.stop_reason = reason
        self.state.updated_at = now
        self._persist_state()
        self._persist_status(now)
        self._persist_final_report(now)

    def _enter_drain(
        self,
        now: datetime,
        lifecycle: CampaignLifecycle,
        reason: EvolutionStopReason,
        *,
        final_reason: str,
    ) -> ControllerTickV1:
        active = self.state.active_run
        if active is None:
            terminal = (
                CampaignLifecycle.KILLED
                if lifecycle == CampaignLifecycle.KILL_DRAINING
                else CampaignLifecycle.COMPLETED
            )
            self._finish(now, terminal, final_reason)
            return self._tick(now, ControllerTickOutcome.STOPPED, [final_reason])
        self.state.lifecycle = lifecycle
        self.state.stop_reason = final_reason
        self.state.updated_at = now
        self.backend.request_graceful_stop(active.handle, reason=reason)
        self._persist_state()
        self._persist_status(now)
        return self._tick(now, ControllerTickOutcome.WAITING, [f"{reason.value}_DRAINING"])

    def _active_resource_stop_reason(
        self, resources: RuntimeResourcesV1
    ) -> str | None:
        if resources.artifact_bytes >= self.policy.max_artifact_bytes:
            return "MAX_ARTIFACT_BYTES_REACHED"
        if resources.free_disk_bytes <= self.policy.min_free_disk_bytes:
            return "MIN_FREE_DISK_REACHED"
        if (
            resources.available_memory_bytes
            and resources.available_memory_bytes
            <= self.policy.min_available_memory_bytes
        ):
            return "MIN_AVAILABLE_MEMORY_REACHED"
        return None

    def run_once(self, *, observed_at: datetime | None = None) -> ControllerTickV1:
        now = observed_at or datetime.now(UTC)
        _require_aware(now, "observed_at")
        if self.state.lifecycle in {
            CampaignLifecycle.COMPLETED,
            CampaignLifecycle.BLOCKED,
            CampaignLifecycle.KILLED,
        }:
            return self._tick(now, ControllerTickOutcome.STOPPED, ["CAMPAIGN_TERMINAL"])

        if Path(self.policy.kill_switch_path).is_file() and self.state.lifecycle != (
            CampaignLifecycle.KILL_DRAINING
        ):
            return self._enter_drain(
                now,
                CampaignLifecycle.KILL_DRAINING,
                EvolutionStopReason.CAMPAIGN_DEADLINE,
                final_reason="KILL_SWITCH_PRESENT",
            )
        if now >= self.state.deadline and self.state.lifecycle == CampaignLifecycle.RUNNING:
            return self._enter_drain(
                now,
                CampaignLifecycle.DEADLINE_DRAINING,
                EvolutionStopReason.CAMPAIGN_DEADLINE,
                final_reason="SEVEN_DAY_DEADLINE",
            )

        active = self.state.active_run
        if active is not None:
            poll = self.backend.poll(active.handle)
            self._persist_status(now, poll=poll)
            if poll.status == AttemptPollStatus.FINISHED:
                if poll.evidence is None:
                    raise HardcoreCampaignError(
                        "finished attempt poll omitted terminal evidence"
                    )
                return self._complete_active(poll.evidence, now=now)
            resources = self._resources()
            resource_reason = self._active_resource_stop_reason(resources)
            if (
                resource_reason is not None
                and self.state.lifecycle == CampaignLifecycle.RUNNING
            ):
                return self._enter_drain(
                    now,
                    CampaignLifecycle.RESOURCE_DRAINING,
                    EvolutionStopReason.CAMPAIGN_DEADLINE,
                    final_reason=resource_reason,
                )
            return self._tick(now, ControllerTickOutcome.WAITING, ["ATTEMPT_PENDING"])

        resources = self._resources()
        if resources.artifact_bytes >= self.policy.max_artifact_bytes:
            self._finish(now, CampaignLifecycle.COMPLETED, "MAX_ARTIFACT_BYTES_REACHED")
            return self._tick(
                now, ControllerTickOutcome.STOPPED, ["MAX_ARTIFACT_BYTES_REACHED"]
            )
        if resources.free_disk_bytes <= self.policy.min_free_disk_bytes:
            self._finish(now, CampaignLifecycle.COMPLETED, "MIN_FREE_DISK_REACHED")
            return self._tick(now, ControllerTickOutcome.STOPPED, ["MIN_FREE_DISK_REACHED"])
        if (
            resources.available_memory_bytes
            and resources.available_memory_bytes
            <= self.policy.min_available_memory_bytes
        ):
            self._persist_status(now, resources=resources)
            return self._tick(now, ControllerTickOutcome.WAITING, ["MEMORY_RESERVE_ACTIVE"])
        return self._start_next_run(now)

    def run_forever(self) -> ControllerTickV1:
        """Advance until a terminal guard, preserving crash-resumable state."""

        while True:
            tick = self.run_once()
            if tick.outcome in {
                ControllerTickOutcome.STOPPED,
                ControllerTickOutcome.BLOCKED,
            }:
                return tick
            time.sleep(self.policy.poll_interval_seconds)

    def status(
        self,
        *,
        observed_at: datetime | None = None,
        poll: AttemptPollV1 | None = None,
        resources: RuntimeResourcesV1 | None = None,
    ) -> CampaignStatusV1:
        now = observed_at or datetime.now(UTC)
        active = self.state.active_run
        current_lane = active.request.lane if active else None
        status_lane = current_lane or self.state.next_lane
        lane_state = self.state.lanes[status_lane]
        champion = lane_state.archive[0].candidate if lane_state and lane_state.archive else None
        next_lane = current_lane.other if current_lane is not None else self.state.next_lane
        if self.state.lanes[next_lane].lifecycle == LaneLifecycle.SUSPENDED:
            next_lane = next_lane.other
        resource_state = resources or self._resources()
        return CampaignStatusV1(
            campaign_id=self.state.campaign_id,
            observed_at=now,
            lifecycle=self.state.lifecycle,
            stop_reason=self.state.stop_reason,
            active_lane=current_lane,
            active_run_id=active.request.run_id if active else None,
            active_generation=poll.current_generation if poll else None,
            current_score=poll.current_score if poll else None,
            current_pair_metrics=poll.current_pair_metrics if poll else [],
            global_champion_score=(
                lane_state.global_champion_score if lane_state else None
            ),
            global_champion_pair_metrics=(
                champion.pair_metrics if champion is not None else []
            ),
            lane_champion_scores={
                lane: state.global_champion_score
                for lane, state in self.state.lanes.items()
            },
            lane_champion_pair_metrics={
                lane: (
                    state.archive[0].candidate.pair_metrics if state.archive else []
                )
                for lane, state in self.state.lanes.items()
            },
            plateau_checks=poll.plateau_checks if poll else None,
            next_timeframe=next_lane,
            elapsed_seconds=max(0.0, (now - self.state.started_at).total_seconds()),
            deadline=self.state.deadline,
            artifact_bytes=resource_state.artifact_bytes,
            free_disk_bytes=resource_state.free_disk_bytes,
            available_memory_bytes=resource_state.available_memory_bytes,
            lane_run_counts={
                lane: state.runs_completed for lane, state in self.state.lanes.items()
            },
        )

    def _persist_status(
        self,
        now: datetime,
        *,
        poll: AttemptPollV1 | None = None,
        resources: RuntimeResourcesV1 | None = None,
    ) -> None:
        status = self.status(observed_at=now, poll=poll, resources=resources)
        payload = _canonical_bytes(status)
        path = self.root / self.STATUS_NAME
        _atomic_replace(path, payload)
        _atomic_replace(
            path.with_name(path.name + ".sha256"),
            (_sha256_bytes(payload) + "\n").encode("ascii"),
        )

    def final_report(self, *, finished_at: datetime | None = None) -> CampaignFinalReportV1:
        if self.state.lifecycle not in {
            CampaignLifecycle.COMPLETED,
            CampaignLifecycle.BLOCKED,
            CampaignLifecycle.KILLED,
        }:
            raise HardcoreCampaignError("final report requires a terminal campaign")
        now = finished_at or self.state.updated_at
        lane_reports: dict[CampaignLane, LaneFinalReportV1] = {}
        for lane, lane_state in self.state.lanes.items():
            champion = lane_state.archive[0].candidate if lane_state.archive else None
            lane_reports[lane] = LaneFinalReportV1(
                lane=lane,
                runs_started=lane_state.runs_started,
                runs_completed=lane_state.runs_completed,
                suspended=lane_state.lifecycle == LaneLifecycle.SUSPENDED,
                suspension_reason=lane_state.suspension_reason,
                champion_label="SEARCH_CHAMPION" if champion is not None else None,
                champion=champion,
                champion_history=lane_state.champion_history,
                outcomes=[item for item in self.state.outcomes if item.lane == lane],
            )
        return CampaignFinalReportV1(
            campaign_id=self.state.campaign_id,
            started_at=self.state.started_at,
            finished_at=now,
            stop_reason=self.state.stop_reason or "UNKNOWN",
            lanes=lane_reports,
        )

    def _persist_final_report(self, now: datetime) -> None:
        report = self.final_report(finished_at=now)
        payload = _canonical_bytes(report)
        path = self.root / self.FINAL_REPORT_NAME
        _write_immutable(path, payload)
        _write_immutable(
            path.with_name(path.name + ".sha256"),
            (_sha256_bytes(payload) + "\n").encode("ascii"),
        )


def render_hardcore_systemd_user_unit(
    *,
    repo_root: str | Path,
    automation_root: str | Path,
    config_15m: str | Path,
    config_1h: str | Path,
    campaign_id: str,
) -> str:
    """Render, but never install, the isolated crash-restart service."""

    repository = Path(repo_root).resolve()
    executable = repository / ".venv/bin/python"
    paths = [executable, Path(config_15m).resolve(), Path(config_1h).resolve()]
    if not repository.is_dir() or any(not path.is_file() for path in paths):
        raise HardcoreCampaignError("systemd unit inputs do not exist")
    if not _SAFE_ID.fullmatch(campaign_id):
        raise HardcoreCampaignError("unsafe campaign_id")

    def quote(value: str | Path) -> str:
        text = str(value)
        if any(character in text for character in ("\n", "\r", "\x00")):
            raise HardcoreCampaignError("systemd argument contains control characters")
        return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'

    command = " ".join(
        [
            quote(executable),
            "-m",
            "genetic_algorithm",
            "hardcore-campaign",
            "start",
            "--campaign-id",
            quote(campaign_id),
            "--config-15m",
            quote(Path(config_15m).resolve()),
            "--config-1h",
            quote(Path(config_1h).resolve()),
            "--automation-root",
            quote(Path(automation_root).resolve()),
        ]
    )
    return (
        "[Unit]\n"
        "Description=Hardcore six-pair GA search campaign\n"
        "After=default.target\n\n"
        "[Service]\n"
        "Type=simple\n"
        f"WorkingDirectory={repository}\n"
        f"ExecStart={command}\n"
        "Restart=on-failure\n"
        "SuccessExitStatus=2\n"
        "RestartPreventExitStatus=2\n"
        "RestartSec=30s\n"
        "TimeoutStopSec=90s\n"
        "KillMode=control-group\n"
        "NoNewPrivileges=true\n\n"
        "[Install]\n"
        "WantedBy=default.target\n"
    )
