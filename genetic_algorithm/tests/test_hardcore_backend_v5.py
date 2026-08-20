from pathlib import Path

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
