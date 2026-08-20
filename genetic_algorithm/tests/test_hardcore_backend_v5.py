import hashlib
from pathlib import Path
from types import SimpleNamespace

import genetic_algorithm.orchestration.hardcore_backend_v5 as backend_v5
import pytest
from genetic_algorithm.orchestration.hardcore_backend_v5 import V2HardcoreAttemptBackendV5
from genetic_algorithm.config.schema import validate_resolved_config_v2_or_raise


def test_v5_backend_materializes_every_lane_and_profile(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    backend = V2HardcoreAttemptBackendV5(
        state_path=tmp_path / "attempts.sqlite3",
        automation_root=tmp_path / "campaign",
        repo_root=repo,
        python_executable=repo / ".venv/bin/python",
    )
    try:
        for lane, profile in (("15m", "edge"), ("1h", "balanced"), ("4h", "productive")):
            config = backend._materialize(
                {
                    "campaign_id": "v5-test",
                    "run_id": f"test-{lane}",
                    "lane": lane,
                    "profile": profile,
                    "seed": 7,
                    "shape": {
                        "population_size": 10,
                        "generations": 12,
                        "cross_niche_offspring": 2,
                    },
                    "config_path": str(
                        repo / f"genetic_algorithm/config/presets/hardcore_multipair_v5_{lane}.yaml"
                    ),
                }
            )
            assert config["backtesting"]["timeframe"] == lane
            assert config["genetic_algorithm"]["population_size"] == 10
            assert config["raw_multipair_score"]["policy_version"] == "raw-multipair-score-v5"
    finally:
        backend.close()


def test_v5_backend_accepts_reduced_canary_shape(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    backend = V2HardcoreAttemptBackendV5(
        state_path=tmp_path / "attempts.sqlite3",
        automation_root=tmp_path / "campaign",
        repo_root=repo,
        python_executable=repo / ".venv/bin/python",
    )
    try:
        config = backend._materialize(
            {
                "campaign_id": "v5-test",
                "run_id": "canary-15m",
                "lane": "15m",
                "profile": "edge",
                "seed": 7,
                "shape": {"population_size": 4, "generations": 3, "cross_niche_offspring": 2},
                "config_path": str(repo / "genetic_algorithm/config/presets/hardcore_multipair_v5_15m.yaml"),
            }
        )
        assert config["genetic_algorithm"]["population_size"] == 4
        assert config["genetic_algorithm"]["generations"] == 3
    finally:
        backend.close()


def test_v5_backend_queues_honest_intraday_4h_panel(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    backend = V2HardcoreAttemptBackendV5(
        state_path=tmp_path / "attempts.sqlite3",
        automation_root=tmp_path / "campaign",
        repo_root=repo,
        python_executable=repo / ".venv/bin/python",
    )
    try:
        handle = backend.queue(
            {
                "campaign_id": "v5-test",
                "run_id": "test-4h",
                "lane": "4h",
                "profile": "productive",
                "seed": 7,
                "shape": {
                    "population_size": 10,
                    "generations": 12,
                    "cross_niche_offspring": 2,
                },
                "config_path": str(
                    repo / "genetic_algorithm/config/presets/hardcore_multipair_v5_4h.yaml"
                ),
            }
        )
        assert handle.startswith("hardcore-v5-")
        state = backend.store.get(handle)
        assert state.worker_binding is not None
        assert state.worker_binding.argv[0] == str(repo / ".venv/bin/python")
        restored_request, restored_config = backend._restore_request(state)
        assert restored_request["lane"] == "4h"
        assert restored_config["backtesting"]["timeframe"] == "4h"
    finally:
        backend.close()


def test_v5_child_runtime_profile_accepts_4h_panel(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    backend = V2HardcoreAttemptBackendV5(
        state_path=tmp_path / "attempts.sqlite3",
        automation_root=tmp_path / "campaign",
        repo_root=repo,
        python_executable=repo / ".venv/bin/python",
    )
    try:
        config = backend._materialize(
            {
                "campaign_id": "v5-test",
                "run_id": "test-4h",
                "lane": "4h",
                "profile": "edge",
                "seed": 7,
                "shape": {
                    "population_size": 10,
                    "generations": 12,
                    "cross_niche_offspring": 2,
                },
                "config_path": str(
                    repo / "genetic_algorithm/config/presets/hardcore_multipair_v5_4h.yaml"
                ),
            }
        )
        config["safety_profile"]["name"] = "hardcore_multipair_child_v1"
        validate_resolved_config_v2_or_raise(config)
    finally:
        backend.close()


def test_v5_archive_seeds_match_bridge_assignments_not_entire_archive(tmp_path, monkeypatch):
    """Only assigned archive parents may enter the immutable worker spec.

    This is the production resume path after a lane has collected its first
    twelve archive candidates.  The worker validates seed IDs against bridge
    assignments, so returning all twelve would reject an otherwise valid run.
    """

    def load_seed(payload: bytes):
        return SimpleNamespace(candidate_id=payload.decode("ascii"))

    monkeypatch.setattr(
        backend_v5.FrozenEvolutionSeedV2,
        "model_validate_json",
        staticmethod(load_seed),
    )
    monkeypatch.setattr(backend_v5, "validate_evolution_seed", lambda seed, config: None)

    entries = []
    for candidate_id, niche in (
        ("edge-1", "EDGE"),
        ("edge-2", "EDGE"),
        ("edge-3", "EDGE"),
        ("activity-1", "ACTIVITY"),
        ("activity-2", "ACTIVITY"),
        ("productive-1", "PRODUCTIVE"),
    ):
        path = tmp_path / f"{candidate_id}.json"
        path.write_bytes(candidate_id.encode("ascii"))
        entries.append(
            {
                "evolution_seed_path": str(path),
                "evolution_seed_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "niche": niche,
            }
        )
    config = {
        "generic_island_model": {
            "islands": [{"name": f"bridge-{index}"} for index in range(1, 4)],
            "archive_seeding": {},
        }
    }

    seeds = V2HardcoreAttemptBackendV5._archive_seeds({"archive_seeds": entries}, config)

    assert [seed.candidate_id for seed in seeds] == [
        "edge-1",
        "edge-2",
        "activity-1",
        "activity-2",
    ]
    for assignment in config["generic_island_model"]["archive_seeding"]["assignments"].values():
        assert [row["candidate_id"] for row in assignment] == [
            "edge-1",
            "edge-2",
            "activity-1",
            "activity-2",
        ]


def test_v5_retries_quarantine_only_an_unprepared_conflicting_request(tmp_path):
    request_path = tmp_path / "attempt-0" / "run_request_v5.json"
    request_path.parent.mkdir(parents=True)
    request_path.write_bytes(b"old-preflight-request")
    request_path.with_suffix(".json.sha256").write_text("old\n")

    V2HardcoreAttemptBackendV5._quarantine_conflicting_unprepared_request(
        request_path, b"new-preflight-request"
    )

    assert not request_path.exists()
    quarantined = list(request_path.parent.glob("run_request_v5.preflight-rejected-*.json"))
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == b"old-preflight-request"
    V2HardcoreAttemptBackendV5._persist_immutable_request(request_path, b"new-preflight-request")
    assert request_path.read_bytes() == b"new-preflight-request"

    request_path.write_bytes(b"executed-request")
    worker_spec = request_path.parent / "worker" / "worker_spec.json"
    worker_spec.parent.mkdir()
    worker_spec.write_text("prepared")
    with pytest.raises(ValueError, match="immutable V5 request differs"):
        V2HardcoreAttemptBackendV5._quarantine_conflicting_unprepared_request(
            request_path, b"different-request"
        )
