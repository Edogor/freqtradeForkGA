"""V5 three-lane campaign scheduling and profile-selection contract.

This deliberately lives beside the immutable V4 controller.  It owns the
three-lane order and staged policy experiment, while worker execution remains
an adapter concern.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from statistics import median
from typing import Protocol

from genetic_algorithm.orchestration.hardcore_archive_v5 import (
    ArchiveCandidateV5,
    select_archive_v5,
)
from genetic_algorithm.orchestration.hardcore_campaign_v1 import (
    EvolutionStopReason,
    FailureClass,
    HardcoreQueueError,
    material_improvement,
)


class CampaignLaneV5(StrEnum):
    FIFTEEN_MINUTES = "15m"
    ONE_HOUR = "1h"
    FOUR_HOURS = "4h"


LANE_ORDER_V5 = (
    CampaignLaneV5.FIFTEEN_MINUTES,
    CampaignLaneV5.ONE_HOUR,
    CampaignLaneV5.FOUR_HOURS,
)


class FitnessProfileV5(StrEnum):
    EDGE = "edge"
    BALANCED = "balanced"
    PRODUCTIVE = "productive"


class CampaignStageV5(StrEnum):
    DIAGNOSTIC = "DIAGNOSTIC"
    CONFIRMATION = "CONFIRMATION"
    PRODUCTION = "PRODUCTION"
    COMPLETED = "COMPLETED"


@dataclass(frozen=True)
class RunShapeV5:
    population_size: int
    generations: int
    cross_niche_offspring: int


DIAGNOSTIC_SHAPE_V5 = RunShapeV5(10, 12, 2)
CONFIRMATION_SHAPE_V5 = RunShapeV5(12, 18, 4)
# Thirty remains the upper generation budget; the runtime watchdog may finish
# an attempt earlier at a generation boundary.
PRODUCTION_SHAPE_V5 = RunShapeV5(12, 30, 4)
# A canary validates the three-lane artifact and transition contract without
# consuming an entire diagnostic arm.
CANARY_SHAPE_V5 = RunShapeV5(4, 3, 2)


@dataclass(frozen=True)
class ProfileRunEvidenceV5:
    lane: CampaignLaneV5
    profile: FitnessProfileV5
    seed: int
    best_comparison_score: float
    best_productive_score: float
    distinct_clusters: int
    worst_pair_drawdown: float


@dataclass(frozen=True)
class ProfileDecisionV5:
    lane: CampaignLaneV5
    ordered_profiles: tuple[FitnessProfileV5, ...]
    comparison_medians: dict[FitnessProfileV5, float]


def next_lane_v5(lane: CampaignLaneV5) -> CampaignLaneV5:
    return LANE_ORDER_V5[(LANE_ORDER_V5.index(lane) + 1) % len(LANE_ORDER_V5)]


def diagnostic_arms_v5(
    *, seed_by_lane: dict[CampaignLaneV5, int]
) -> list[tuple[CampaignLaneV5, FitnessProfileV5, int, RunShapeV5]]:
    """Nine fair first-stage arms: same initial seed within each lane."""

    # Profile is the outer loop so every consecutive trio remains the strict
    # 15m -> 1h -> 4h cadence.  The diagnostic comparison is still fair:
    # each lane/profile receives the same lane-local seed.
    return [
        (lane, profile, seed_by_lane[lane], DIAGNOSTIC_SHAPE_V5)
        for profile in FitnessProfileV5
        for lane in LANE_ORDER_V5
    ]


def select_profiles_v5(
    lane: CampaignLaneV5,
    evidence: Iterable[ProfileRunEvidenceV5],
) -> ProfileDecisionV5:
    """Order policies by the fixed balanced comparison policy.

    The primary metric is the median best comparison score.  Ties within the
    campaign material-improvement epsilon (0.25) favour productive frequency,
    then behavioural/logic cluster count, then lower worst-pair drawdown.
    """

    grouped: dict[FitnessProfileV5, list[ProfileRunEvidenceV5]] = {
        profile: [] for profile in FitnessProfileV5
    }
    for item in evidence:
        if item.lane == lane:
            grouped[item.profile].append(item)
    if any(not rows for rows in grouped.values()):
        raise ValueError("profile selection requires evidence from every V5 policy")
    comparison_medians = {
        profile: float(median(item.best_comparison_score for item in rows))
        for profile, rows in grouped.items()
    }

    def secondary_key(profile: FitnessProfileV5) -> tuple[float, float, float, float, str]:
        rows = grouped[profile]
        comparison = comparison_medians[profile]
        productive = float(median(item.best_productive_score for item in rows))
        clusters = float(median(item.distinct_clusters for item in rows))
        drawdown = float(median(item.worst_pair_drawdown for item in rows))
        return (productive, clusters, -drawdown, comparison, profile.value)

    best = max(comparison_medians.values())
    close = [profile for profile in FitnessProfileV5 if comparison_medians[profile] >= best - 0.25]
    remaining = [profile for profile in FitnessProfileV5 if profile not in close]
    # A sub-material score difference deliberately defers to the productive
    # frontier.  Outside that band, the comparison policy remains primary.
    ordered = sorted(close, key=secondary_key, reverse=True) + sorted(
        remaining,
        key=lambda profile: (comparison_medians[profile], *secondary_key(profile)),
        reverse=True,
    )

    return ProfileDecisionV5(
        lane=lane,
        ordered_profiles=tuple(ordered),
        comparison_medians=comparison_medians,
    )


def confirmation_arms_v5(
    decisions: Iterable[ProfileDecisionV5],
    *,
    second_seed_by_lane: dict[CampaignLaneV5, int],
) -> list[tuple[CampaignLaneV5, FitnessProfileV5, int, RunShapeV5]]:
    """Six second-stage arms: the top two profiles for every lane."""

    by_lane = {item.lane: item for item in decisions}
    if set(by_lane) != set(LANE_ORDER_V5):
        raise ValueError("confirmation requires one decision per lane")
    arms: list[tuple[CampaignLaneV5, FitnessProfileV5, int, RunShapeV5]] = []
    for rank in range(2):
        for lane in LANE_ORDER_V5:
            arms.append(
                (
                    lane,
                    by_lane[lane].ordered_profiles[rank],
                    second_seed_by_lane[lane],
                    CONFIRMATION_SHAPE_V5,
                )
            )
    return arms


def profile_rescue_order_v5(decision: ProfileDecisionV5) -> tuple[FitnessProfileV5, ...]:
    """Winner first; only a full stale recipe cycle unlocks the next profile."""

    return decision.ordered_profiles


# The scheduler below intentionally has a narrow, JSON-only backend boundary.
# The engine adapter can evolve independently, but it may only report a
# completed attempt through this evidence envelope.  In particular, a backend
# cannot silently turn a failed six-pair replay into a zero score.


class AttemptBackendV5(Protocol):
    def queue(self, request: dict[str, object]) -> str: ...

    def poll(self, handle: str) -> dict[str, object]: ...

    def request_graceful_stop(self, handle: str, *, reason: str) -> None: ...


@dataclass(frozen=True)
class HardcoreCampaignPolicyV5:
    campaign_id: str
    automation_root: str
    lane_configs: dict[CampaignLaneV5, str]
    runtime_seconds: int = 7 * 24 * 60 * 60
    max_attempt_runtime_seconds: int = 10 * 60 * 60
    campaign_mode: str = "production"
    max_completed_runs: int | None = None
    kill_switch_path: str | None = None

    def __post_init__(self) -> None:
        if not self.campaign_id or any(
            char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
            for char in self.campaign_id
        ):
            raise ValueError("campaign_id contains unsafe characters")
        if set(self.lane_configs) != set(LANE_ORDER_V5):
            raise ValueError("V5 policy needs exactly 15m, 1h and 4h configs")
        if self.runtime_seconds != 7 * 24 * 60 * 60:
            raise ValueError("V5 campaign duration is exactly seven days")
        if self.max_attempt_runtime_seconds != 10 * 60 * 60:
            raise ValueError("V5 attempt budget is exactly ten hours")
        if self.campaign_mode not in {"production", "canary"}:
            raise ValueError("campaign_mode must be production or canary")
        if self.campaign_mode == "canary" and self.max_completed_runs != 3:
            raise ValueError("V5 canary must complete exactly one run per lane")

    @property
    def policy_hash(self) -> str:
        return _sha256(
            _canonical(
                {
                    "version": "hardcore-multipair-campaign-v5",
                    "campaign_id": self.campaign_id,
                    "automation_root": str(Path(self.automation_root).resolve()),
                    "lane_configs": {
                        lane.value: str(Path(path).resolve())
                        for lane, path in self.lane_configs.items()
                    },
                    "runtime_seconds": self.runtime_seconds,
                    "max_attempt_runtime_seconds": self.max_attempt_runtime_seconds,
                    "campaign_mode": self.campaign_mode,
                    "max_completed_runs": self.max_completed_runs,
                }
            )
        )


def default_hardcore_campaign_policy_v5(
    *,
    campaign_id: str,
    automation_root: str | Path,
    config_15m: str | Path,
    config_1h: str | Path,
    config_4h: str | Path,
    canary: bool = False,
) -> HardcoreCampaignPolicyV5:
    root = Path(automation_root).resolve()
    return HardcoreCampaignPolicyV5(
        campaign_id=campaign_id,
        automation_root=str(root),
        lane_configs={
            CampaignLaneV5.FIFTEEN_MINUTES: str(Path(config_15m).resolve()),
            CampaignLaneV5.ONE_HOUR: str(Path(config_1h).resolve()),
            CampaignLaneV5.FOUR_HOURS: str(Path(config_4h).resolve()),
        },
        campaign_mode="canary" if canary else "production",
        max_completed_runs=3 if canary else None,
        kill_switch_path=str(root / "STOP_HARDCORE_CAMPAIGN_V5"),
    )


@dataclass
class LaneStateV5:
    runs_started: int = 0
    runs_completed: int = 0
    global_balanced_score: float | None = None
    selected_profiles: list[str] = field(default_factory=list)
    rescue_index: int = 0
    no_progress_cycles: int = 0
    diagnostics: list[dict[str, object]] = field(default_factory=list)
    archive: list[dict[str, object]] = field(default_factory=list)
    suspended_reason: str | None = None


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )
        + "\n"
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class HardcoreCampaignControllerV5:
    """Persistent three-lane staged V5 campaign controller.

    It separates diagnostic, confirmation, and production evidence. A profile
    is never selected from another timeframe, and exhausting every selected
    rescue recipe ends that lane instead of looping recipe names forever.
    """

    STATE_NAME = "campaign_state_v5.json"

    def __init__(
        self,
        *,
        policy: HardcoreCampaignPolicyV5,
        backend: AttemptBackendV5,
        started_at: datetime | None = None,
    ) -> None:
        self.policy, self.backend = policy, backend
        self.root = Path(policy.automation_root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / self.STATE_NAME
        if self.state_path.exists():
            self.state = self._load()
            if self.state["policy_hash"] != policy.policy_hash:
                raise ValueError("existing V5 state belongs to another policy")
        else:
            now = started_at or datetime.now(UTC)
            if now.tzinfo is None:
                raise ValueError("started_at must be timezone-aware")
            self.state: dict[str, object] = {
                "schema_version": "hardcore-campaign-state-v5",
                "campaign_id": policy.campaign_id,
                "policy_hash": policy.policy_hash,
                "started_at": now.isoformat(),
                "deadline": (now + timedelta(seconds=policy.runtime_seconds)).isoformat(),
                "lifecycle": "RUNNING",
                "stage": CampaignStageV5.DIAGNOSTIC.value,
                "next_lane": CampaignLaneV5.FIFTEEN_MINUTES.value,
                "active": None,
                "run_sequence": 0,
                "outcomes": [],
                "lanes": {lane.value: vars(LaneStateV5()) for lane in LANE_ORDER_V5},
            }
            self._persist()

    def _load(self) -> dict[str, object]:
        checksum = self.state_path.with_suffix(self.state_path.suffix + ".sha256")
        if (
            not checksum.is_file()
            or _sha256(self.state_path.read_bytes()) != checksum.read_text(encoding="ascii").strip()
        ):
            raise ValueError("V5 campaign state checksum mismatch")
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def _persist(self) -> None:
        payload = _canonical(self.state)
        _atomic_write(self.state_path, payload)
        _atomic_write(
            self.state_path.with_suffix(self.state_path.suffix + ".sha256"),
            (_sha256(payload) + "\n").encode("ascii"),
        )

    def _lane(self, lane: CampaignLaneV5) -> dict[str, object]:
        return self.state["lanes"][lane.value]  # type: ignore[index, no-any-return]

    def _seed(self, lane: CampaignLaneV5, sequence: int) -> int:
        return int.from_bytes(
            hashlib.sha256(f"{self.policy.campaign_id}:{lane.value}:{sequence}".encode()).digest()[
                :4
            ],
            "big",
        )

    def _queued_arms(self) -> list[tuple[CampaignLaneV5, FitnessProfileV5, int, RunShapeV5]]:
        seed = {lane: self._seed(lane, 0) for lane in LANE_ORDER_V5}
        stage = CampaignStageV5(self.state["stage"])
        if stage is CampaignStageV5.DIAGNOSTIC:
            completed = {
                (row["lane"], row["profile"])
                for row in self.state["outcomes"]
                if row.get("stage") == stage.value
            }  # type: ignore[union-attr]
            arms = [
                arm
                for arm in diagnostic_arms_v5(seed_by_lane=seed)
                if (arm[0].value, arm[1].value) not in completed
            ]
            if self.policy.campaign_mode == "canary":
                # The first profile is EDGE and diagnostic ordering is
                # 15m -> 1h -> 4h, exactly once each.
                return [
                    (lane, profile, arm_seed, CANARY_SHAPE_V5)
                    for lane, profile, arm_seed, _shape in arms[:3]
                ]
            return arms
        if stage is CampaignStageV5.CONFIRMATION:
            decisions = self._decisions()
            second = {lane: self._seed(lane, 1) for lane in LANE_ORDER_V5}
            completed = {
                (row["lane"], row["profile"])
                for row in self.state["outcomes"]
                if row.get("stage") == stage.value
            }  # type: ignore[union-attr]
            return [
                arm
                for arm in confirmation_arms_v5(decisions, second_seed_by_lane=second)
                if (arm[0].value, arm[1].value) not in completed
            ]
        if stage is CampaignStageV5.PRODUCTION:
            return self._production_arms()
        return []

    def _decisions(self) -> list[ProfileDecisionV5]:
        rows = self.state["outcomes"]  # type: ignore[assignment]
        evidence = [
            ProfileRunEvidenceV5(
                CampaignLaneV5(row["lane"]),
                FitnessProfileV5(row["profile"]),
                int(row["seed"]),
                float(row["comparison_score"]),
                float(row["productive_score"]),
                int(row["clusters"]),
                float(row["worst_drawdown"]),
            )
            for row in rows
            if row.get("stage") == CampaignStageV5.DIAGNOSTIC.value and row.get("valid")
        ]  # type: ignore[union-attr]
        return [select_profiles_v5(lane, evidence) for lane in LANE_ORDER_V5]

    def _production_arms(self) -> list[tuple[CampaignLaneV5, FitnessProfileV5, int, RunShapeV5]]:
        requested = CampaignLaneV5(str(self.state["next_lane"]))
        for offset in range(len(LANE_ORDER_V5)):
            lane = LANE_ORDER_V5[(LANE_ORDER_V5.index(requested) + offset) % len(LANE_ORDER_V5)]
            state = self._lane(lane)
            selected = state["selected_profiles"]
            if state["suspended_reason"] or not selected:
                continue
            profile = FitnessProfileV5(selected[min(int(state["rescue_index"]), len(selected) - 1)])
            return [
                (
                    lane,
                    profile,
                    self._seed(lane, int(state["runs_started"]) + 2),
                    PRODUCTION_SHAPE_V5,
                )
            ]
        return []

    def _advance_stage_if_ready(self) -> None:
        stage = CampaignStageV5(self.state["stage"])
        if stage is CampaignStageV5.DIAGNOSTIC and not self._queued_arms():
            try:
                decisions = self._decisions()
            except ValueError:
                self.state["lifecycle"] = "BLOCKED"
                return
            for decision in decisions:
                self._lane(decision.lane)["selected_profiles"] = [
                    profile.value for profile in decision.ordered_profiles
                ]
            self.state["stage"] = CampaignStageV5.CONFIRMATION.value
        elif stage is CampaignStageV5.CONFIRMATION and not self._queued_arms():
            self.state["stage"] = CampaignStageV5.PRODUCTION.value

    def tick(self, *, now: datetime | None = None) -> dict[str, object]:  # noqa: C901
        now = now or datetime.now(UTC)
        if now.tzinfo is None:
            raise ValueError("now must be timezone-aware")
        if self.state["lifecycle"] != "RUNNING":
            return self.status(now)
        active = self.state.get("active")
        stop_marker = Path(
            self.policy.kill_switch_path or (self.root / "STOP_HARDCORE_CAMPAIGN_V5")
        )
        if stop_marker.is_file() and active:
            self.backend.request_graceful_stop(str(active["handle"]), reason="CAMPAIGN_DEADLINE")
        if active:
            polled = self.backend.poll(str(active["handle"]))
            if polled.get("status") != "FINISHED":
                return self.status(now, polled)
            self._finalize_active(active, polled, now)
            self._advance_stage_if_ready()
            if stop_marker.is_file():
                self.state["lifecycle"] = "COMPLETED"
        elif stop_marker.is_file():
            self.state["lifecycle"] = "COMPLETED"
        if now >= datetime.fromisoformat(str(self.state["deadline"])):
            self.state["lifecycle"] = "COMPLETED"
        elif self.policy.campaign_mode == "canary" and len(self.state["outcomes"]) >= 3:  # type: ignore[arg-type]
            valid_lanes = {row["lane"] for row in self.state["outcomes"] if row.get("valid")}  # type: ignore[union-attr]
            self.state["lifecycle"] = (
                "COMPLETED" if valid_lanes == {lane.value for lane in LANE_ORDER_V5} else "BLOCKED"
            )
        if self.state["lifecycle"] == "RUNNING":
            arms = self._queued_arms()
            if arms:
                lane, profile, seed, shape = arms[0]
                sequence = int(self.state["run_sequence"]) + 1
                run_id = (
                    f"{self.policy.campaign_id}-run-{sequence:06d}-{lane.value}-{profile.value}"
                )
                request: dict[str, object] = {
                    "version": "v5",
                    "campaign_id": self.policy.campaign_id,
                    "run_id": run_id,
                    "lane": lane.value,
                    "profile": profile.value,
                    "seed": seed,
                    "stage": self.state["stage"],
                    "shape": vars(shape),
                    "config_path": self.policy.lane_configs[lane],
                    "archive_seeds": [
                        {**entry["candidate"], "niche": entry["niche"]}
                        for entry in self._lane(lane)["archive"]
                    ],
                }
                self.state["run_sequence"] = sequence
                self._lane(lane)["runs_started"] = int(self._lane(lane)["runs_started"]) + 1
                try:
                    handle = self.backend.queue(request)
                except HardcoreQueueError as exc:
                    self._record_queue_failure(
                        request,
                        failure_class=exc.failure_class,
                        error_code=exc.error_code,
                        detail=str(exc),
                        now=now,
                    )
                except (OSError, ValueError) as exc:
                    # Queue-time validation and immutable-artifact failures
                    # cannot be repaired by restarting the systemd service.
                    # Suspend just this lane; the remaining lanes may still
                    # supply useful, independent evidence.
                    self._record_queue_failure(
                        request,
                        failure_class=FailureClass.DETERMINISTIC_CONFIG,
                        error_code=f"QUEUE_{type(exc).__name__.upper()}",
                        detail=str(exc),
                        now=now,
                    )
                else:
                    self.state["active"] = {"handle": handle, **request}
            elif self.state["stage"] == CampaignStageV5.PRODUCTION.value:
                self.state["lifecycle"] = "COMPLETED"
        self._persist()
        return self.status(now)

    def _record_queue_failure(
        self,
        request: dict[str, object],
        *,
        failure_class: FailureClass,
        error_code: str,
        detail: str,
        now: datetime,
    ) -> None:
        """Persist a pre-worker failure and let other timeframes continue."""

        lane = CampaignLaneV5(str(request["lane"]))
        record = {
            "run_id": request["run_id"],
            "lane": lane.value,
            "profile": request["profile"],
            "seed": request["seed"],
            "stage": request["stage"],
            "valid": False,
            "comparison_score": 0.0,
            "productive_score": 0.0,
            "clusters": 0,
            "worst_drawdown": 1.0,
            "stop_reason": EvolutionStopReason.TECHNICAL_INVALID.value,
            "failure_class": failure_class.value,
            "actual_generations": 0,
            "score_curve": [],
            "candidates": [],
            "error_code": error_code,
            "error_detail": detail[:2000],
            "finished_at": now.isoformat(),
        }
        outcome_path = self.root / "runs" / str(request["run_id"]) / "evolution_outcome_v5.json"
        payload = _canonical({"schema_version": "2.0", "outcome_version": "hardcore-v5", **record})
        _atomic_write(outcome_path, payload)
        _atomic_write(
            outcome_path.with_suffix(outcome_path.suffix + ".sha256"),
            (_sha256(payload) + "\n").encode("ascii"),
        )
        record["outcome_sha256"] = _sha256(payload)
        self.state["outcomes"].append(record)  # type: ignore[union-attr]
        lane_state = self._lane(lane)
        lane_state["runs_completed"] = int(lane_state["runs_completed"]) + 1
        if failure_class.suspends_lane:
            lane_state["suspended_reason"] = error_code
        self.state["active"] = None
        self.state["next_lane"] = next_lane_v5(lane).value

    def _finalize_active(
        self, active: dict[str, object], polled: dict[str, object], now: datetime
    ) -> None:
        lane = CampaignLaneV5(str(active["lane"]))
        evidence = polled.get("evidence")
        if not isinstance(evidence, dict):
            raise ValueError("finished V5 attempt lacks evidence")
        required = {
            "valid",
            "comparison_score",
            "productive_score",
            "clusters",
            "worst_drawdown",
            "stop_reason",
        }
        if not required <= set(evidence):
            raise ValueError("V5 attempt evidence is incomplete")
        valid = bool(evidence["valid"])
        record = {
            "run_id": active["run_id"],
            "lane": lane.value,
            "profile": active["profile"],
            "seed": active["seed"],
            "stage": active["stage"],
            **evidence,
            "finished_at": now.isoformat(),
        }
        # Each terminal decision is independently immutable.  The mutable
        # campaign state references these records but is not a substitute for
        # a worker/result artifact when diagnosing an interrupted campaign.
        outcome_path = self.root / "runs" / str(active["run_id"]) / "evolution_outcome_v5.json"
        payload = _canonical({"schema_version": "2.0", "outcome_version": "hardcore-v5", **record})
        _atomic_write(outcome_path, payload)
        _atomic_write(
            outcome_path.with_suffix(outcome_path.suffix + ".sha256"),
            (_sha256(payload) + "\n").encode("ascii"),
        )
        record["outcome_sha256"] = _sha256(payload)
        self.state["outcomes"].append(record)  # type: ignore[union-attr]
        lane_state = self._lane(lane)
        if valid:
            self._update_archive(lane_state, evidence)
        lane_state["runs_completed"] = int(lane_state["runs_completed"]) + 1
        if valid:
            previous = lane_state["global_balanced_score"]
            candidate = float(evidence["comparison_score"])
            if material_improvement(previous if isinstance(previous, float) else None, candidate):
                lane_state["global_balanced_score"] = candidate
                lane_state["rescue_index"] = 0
                lane_state["no_progress_cycles"] = 0
            elif self.state["stage"] == CampaignStageV5.PRODUCTION:
                lane_state["no_progress_cycles"] = int(lane_state["no_progress_cycles"]) + 1
                if int(lane_state["no_progress_cycles"]) >= 1:
                    lane_state["rescue_index"] = int(lane_state["rescue_index"]) + 1
                    lane_state["no_progress_cycles"] = 0
                    if int(lane_state["rescue_index"]) >= len(lane_state["selected_profiles"]):
                        lane_state["suspended_reason"] = (
                            EvolutionStopReason.NO_GLOBAL_PROGRESS.value
                        )
        else:
            failure = str(evidence.get("failure_class", FailureClass.DETERMINISTIC_DATA.value))
            if failure in {
                FailureClass.DETERMINISTIC_CONFIG.value,
                FailureClass.DETERMINISTIC_DATA.value,
            }:
                lane_state["suspended_reason"] = failure
        self.state["active"] = None
        self.state["next_lane"] = next_lane_v5(lane).value

    @staticmethod
    def _update_archive(lane_state: dict[str, object], evidence: dict[str, object]) -> None:
        """Merge strict-replay candidates into the five v5 archive niches."""

        raw_candidates = [
            *(
                entry["candidate"]
                for entry in lane_state["archive"]
                if isinstance(entry, dict) and isinstance(entry.get("candidate"), dict)
            ),
            *(entry for entry in evidence.get("candidates", []) if isinstance(entry, dict)),
        ]
        archive_candidates: list[ArchiveCandidateV5] = []
        by_hash: dict[str, dict[str, object]] = {}
        for candidate in raw_candidates:
            try:
                pairs = candidate["pairs"]
                behavior = tuple(
                    value
                    for pair in pairs
                    for value in (
                        float(pair["Q"]),
                        float(pair["A"]),
                        float(pair["F"]),
                        float(pair["R"]),
                        float(pair["D"]),
                        float(pair["U"]),
                    )
                )
                converted = ArchiveCandidateV5(
                    candidate_id=str(candidate["candidate_id"]),
                    evolutionary_phenotype_hash=str(candidate["evolutionary_phenotype_hash"]),
                    balanced_score=float(candidate["scores"]["balanced"]),
                    edge_score=float(candidate["edge_value"]),
                    activity_score=float(candidate["activity_value"]),
                    productive_frequency_score=float(candidate["productive_value"]),
                    behavior_vector=behavior,
                    logic_tokens=frozenset(map(str, candidate.get("logic_tokens", []))),
                )
            except (KeyError, TypeError, ValueError):
                continue
            archive_candidates.append(converted)
            by_hash[converted.evolutionary_phenotype_hash] = candidate
        assignments = select_archive_v5(archive_candidates)
        lane_state["archive"] = [
            {
                "niche": assignment.niche.value,
                "candidate": by_hash[assignment.candidate.evolutionary_phenotype_hash],
            }
            for assignment in assignments
        ]

    def status(
        self, now: datetime | None = None, live: dict[str, object] | None = None
    ) -> dict[str, object]:
        now = now or datetime.now(UTC)
        archive_leaders = {
            lane.value: {
                str(entry["niche"]): entry["candidate"] for entry in self._lane(lane)["archive"]
            }
            for lane in LANE_ORDER_V5
        }
        return {
            "campaign_id": self.policy.campaign_id,
            "lifecycle": self.state["lifecycle"],
            "stage": self.state["stage"],
            "active": self.state.get("active"),
            "live": live or {},
            "lane_leaders": {
                lane.value: self._lane(lane)["global_balanced_score"] for lane in LANE_ORDER_V5
            },
            "archive_leaders": archive_leaders,
            "completed_runs": len(self.state["outcomes"]),
            "deadline": self.state["deadline"],
            "observed_at": now.isoformat(),
        }
