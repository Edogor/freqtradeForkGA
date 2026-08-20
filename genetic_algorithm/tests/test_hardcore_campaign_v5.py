from datetime import UTC, datetime

import hashlib

from genetic_algorithm.orchestration.hardcore_campaign_v5 import (
    CampaignLaneV5,
    FitnessProfileV5,
    HardcoreCampaignControllerV5,
    confirmation_arms_v5,
    default_hardcore_campaign_policy_v5,
    diagnostic_arms_v5,
    next_lane_v5,
    select_profiles_v5,
)


class _Backend:
    def __init__(self):
        self.requests = []
        self.finished = {}
        self.stop_requests = []

    def queue(self, request):
        handle = f"h-{len(self.requests)}"
        self.requests.append(request)
        return handle

    def poll(self, handle):
        return self.finished.get(handle, {"status": "RUNNING"})

    def request_graceful_stop(self, handle, *, reason):
        self.stop_requests.append((handle, reason))


class _QueueValidationFailureBackend(_Backend):
    def queue(self, request):
        if request["lane"] == "15m":
            raise ValueError("invalid immutable archive assignment")
        return super().queue(request)


def _completed(score=1.0, valid=True):
    return {
        "status": "FINISHED",
        "evidence": {
            "valid": valid,
            "comparison_score": score,
            "productive_score": score,
            "clusters": 2,
            "worst_drawdown": 0.1,
            "stop_reason": "PLATEAU",
        },
    }


def _evidence(lane, profile, score, productive, clusters=3, drawdown=0.1):
    from genetic_algorithm.orchestration.hardcore_campaign_v5 import ProfileRunEvidenceV5

    return ProfileRunEvidenceV5(lane, profile, 1, score, productive, clusters, drawdown)


def test_three_lane_order_and_nine_paired_diagnostic_arms():
    assert next_lane_v5(CampaignLaneV5.FIFTEEN_MINUTES) is CampaignLaneV5.ONE_HOUR
    assert next_lane_v5(CampaignLaneV5.ONE_HOUR) is CampaignLaneV5.FOUR_HOURS
    assert next_lane_v5(CampaignLaneV5.FOUR_HOURS) is CampaignLaneV5.FIFTEEN_MINUTES
    arms = diagnostic_arms_v5(
        seed_by_lane={
            CampaignLaneV5.FIFTEEN_MINUTES: 1,
            CampaignLaneV5.ONE_HOUR: 2,
            CampaignLaneV5.FOUR_HOURS: 3,
        }
    )
    assert len(arms) == 9
    assert {seed for lane, _profile, seed, _shape in arms if lane is CampaignLaneV5.ONE_HOUR} == {2}


def test_profile_selection_uses_productive_frontier_for_sub_material_ties():
    lane = CampaignLaneV5.ONE_HOUR
    evidence = [
        _evidence(lane, FitnessProfileV5.EDGE, 25.00, 0.08),
        _evidence(lane, FitnessProfileV5.BALANCED, 25.10, 0.09),
        _evidence(lane, FitnessProfileV5.PRODUCTIVE, 25.16, 0.11),
    ]
    decision = select_profiles_v5(lane, evidence)
    assert decision.ordered_profiles[0] is FitnessProfileV5.PRODUCTIVE
    arms = confirmation_arms_v5(
        [
            decision,
            select_profiles_v5(
                CampaignLaneV5.FIFTEEN_MINUTES,
                [
                    _evidence(CampaignLaneV5.FIFTEEN_MINUTES, profile, 1, 0.1)
                    for profile in FitnessProfileV5
                ],
            ),
            select_profiles_v5(
                CampaignLaneV5.FOUR_HOURS,
                [
                    _evidence(CampaignLaneV5.FOUR_HOURS, profile, 1, 0.1)
                    for profile in FitnessProfileV5
                ],
            ),
        ],
        second_seed_by_lane={
            CampaignLaneV5.FIFTEEN_MINUTES: 4,
            CampaignLaneV5.ONE_HOUR: 5,
            CampaignLaneV5.FOUR_HOURS: 6,
        },
    )
    assert len(arms) == 6


def test_controller_keeps_three_lane_order_and_canary_requires_valid_each_lane(tmp_path):
    backend = _Backend()
    policy = default_hardcore_campaign_policy_v5(
        campaign_id="canary-v5",
        automation_root=tmp_path / "campaign",
        config_15m=tmp_path / "15m.yaml",
        config_1h=tmp_path / "1h.yaml",
        config_4h=tmp_path / "4h.yaml",
        canary=True,
    )
    now = datetime(2026, 8, 20, tzinfo=UTC)
    controller = HardcoreCampaignControllerV5(policy=policy, backend=backend, started_at=now)
    controller.tick(now=now)
    assert backend.requests[-1]["lane"] == "15m"
    assert backend.requests[-1]["shape"] == {
        "population_size": 4,
        "generations": 3,
        "cross_niche_offspring": 2,
    }
    backend.finished["h-0"] = _completed()
    controller.tick(now=now)
    assert backend.requests[-1]["lane"] == "1h"
    backend.finished["h-1"] = _completed()
    controller.tick(now=now)
    assert backend.requests[-1]["lane"] == "4h"
    backend.finished["h-2"] = _completed(valid=False)
    assert controller.tick(now=now)["lifecycle"] == "BLOCKED"


def test_controller_kill_switch_waits_for_active_attempt(tmp_path):
    backend = _Backend()
    policy = default_hardcore_campaign_policy_v5(
        campaign_id="stop-v5",
        automation_root=tmp_path / "campaign",
        config_15m=tmp_path / "15m.yaml",
        config_1h=tmp_path / "1h.yaml",
        config_4h=tmp_path / "4h.yaml",
    )
    now = datetime(2026, 8, 20, tzinfo=UTC)
    controller = HardcoreCampaignControllerV5(policy=policy, backend=backend, started_at=now)
    controller.tick(now=now)
    (tmp_path / "campaign" / "STOP_HARDCORE_CAMPAIGN_V5").touch()
    assert controller.tick(now=now)["lifecycle"] == "RUNNING"
    assert backend.stop_requests == [("h-0", "CAMPAIGN_DEADLINE")]
    backend.finished["h-0"] = _completed()
    assert controller.tick(now=now)["lifecycle"] == "COMPLETED"


def test_controller_suspends_only_the_prequeue_failure_lane_and_persists_outcome(tmp_path):
    backend = _QueueValidationFailureBackend()
    policy = default_hardcore_campaign_policy_v5(
        campaign_id="queue-failure-v5",
        automation_root=tmp_path / "campaign",
        config_15m=tmp_path / "15m.yaml",
        config_1h=tmp_path / "1h.yaml",
        config_4h=tmp_path / "4h.yaml",
    )
    now = datetime(2026, 8, 20, tzinfo=UTC)
    controller = HardcoreCampaignControllerV5(policy=policy, backend=backend, started_at=now)

    status = controller.tick(now=now)
    assert status["lifecycle"] == "RUNNING"
    assert controller.state["active"] is None
    assert controller.state["lanes"]["15m"]["suspended_reason"] == "QUEUE_VALUEERROR"
    assert controller.state["next_lane"] == "1h"
    outcome = controller.state["outcomes"][0]
    assert outcome["failure_class"] == "DETERMINISTIC_CONFIG"
    path = tmp_path / "campaign" / "runs" / outcome["run_id"] / "evolution_outcome_v5.json"
    assert path.is_file()
    checksum = path.with_suffix(path.suffix + ".sha256")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == checksum.read_text().strip()

    controller.tick(now=now)
    assert backend.requests[-1]["lane"] == "1h"


def test_v5_archive_keeps_strict_candidate_and_its_seed_metadata():
    from genetic_algorithm.orchestration.hardcore_campaign_v5 import (
        HardcoreCampaignControllerV5,
    )

    candidate = {
        "candidate_id": "candidate-1",
        "evolutionary_phenotype_hash": "a" * 64,
        "scores": {"balanced": 1.0, "edge": 1.0, "productive": 1.0},
        "edge_value": 0.2,
        "activity_value": 0.3,
        "productive_value": 0.1,
        "logic_tokens": ["indicator:RSI"],
        "evolution_seed_path": "/tmp/seed.json",
        "evolution_seed_sha256": "b" * 64,
        "pairs": [{"Q": 0.2, "A": 0.3, "F": 0.1, "R": 0.1, "D": 0.1, "U": 0.1} for _ in range(6)],
    }
    lane_state = {"archive": []}
    HardcoreCampaignControllerV5._update_archive(lane_state, {"candidates": [candidate]})
    assert lane_state["archive"][0]["candidate"]["evolution_seed_path"] == "/tmp/seed.json"
