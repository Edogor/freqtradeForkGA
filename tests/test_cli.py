"""Tests for the unified CLI (cli.py) and its integration with the registry."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from genetic_algorithm.cli import _detect_ga_type, main
from genetic_algorithm.orchestration.attempt_state_v2 import AttemptLifecycleStatus
from genetic_algorithm.run_ga import main as legacy_runner_main


# ---------------------------------------------------------------------------
# _detect_ga_type
# ---------------------------------------------------------------------------

class TestDetectGaType:
    def test_generic_island(self):
        cfg = {"generic_island_model": {"enabled": True}, "island_model": {"enabled": False}}
        assert _detect_ga_type(cfg) == "generic_island"

    def test_island(self):
        cfg = {"generic_island_model": {"enabled": False}, "island_model": {"enabled": True}}
        assert _detect_ga_type(cfg) == "island"

    def test_standard(self):
        cfg = {"generic_island_model": {"enabled": False}, "island_model": {"enabled": False}}
        assert _detect_ga_type(cfg) == "standard"

    def test_missing_keys_defaults_standard(self):
        assert _detect_ga_type({}) == "standard"


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

class TestCliHelp:
    def test_no_args_returns_0(self):
        """No command prints help and returns 0."""
        rc = main([])
        assert rc == 0

    def test_unknown_command(self):
        """Unrecognized command should exit nonzero."""
        with pytest.raises(SystemExit):
            main(["bogus_command"])


# ---------------------------------------------------------------------------
# 'run' command integration
# ---------------------------------------------------------------------------

class TestCmdRun:
    @pytest.fixture
    def config_file(self, tmp_path):
        cfg = {
            "backtesting": {"pairs": ["BTC/USDT"], "timerange": "20250101-20250201"},
            "genetic_algorithm": {"population_size": 5, "generations": 1, "elite_size": 1},
            "parallel_evaluation": {"enabled": False},
            "holdout_validation": {"enabled": False},
            "holdout_monitoring": {"enabled": False},
        }
        p = tmp_path / "test_config.yaml"
        p.write_text(yaml.dump(cfg))
        return str(p)

    @pytest.fixture
    def registry_path(self, tmp_path):
        return tmp_path / "registry.json"

    def test_run_registers_and_completes(self, config_file, registry_path):
        """run command delegates to the canonical exact-attempt runner."""
        terminal = MagicMock(
            attempt_id="attempt-test",
            status=AttemptLifecycleStatus.SUCCEEDED,
            artifact_root="/tmp/attempt-test",
        )
        with patch(
            "genetic_algorithm.orchestration.runner_v2.run_standard_attempt",
            return_value=terminal,
        ) as run_attempt:
            rc = main(["run", config_file, "--name", "test_exp", "--no-monitor", "--yes"])

            assert rc == 0
            run_attempt.assert_called_once()
            assert run_attempt.call_args.kwargs["experiment_name"] == "test_exp"

    def test_run_marks_fail_on_exception(self, config_file, registry_path):
        """Runner exceptions are surfaced without a legacy-registry fallback."""
        with patch(
            "genetic_algorithm.orchestration.runner_v2.run_standard_attempt",
            side_effect=RuntimeError("boom"),
        ):
            rc = main(["run", config_file, "--name", "fail_exp", "--no-monitor", "--yes"])
            assert rc == 1

    def test_run_missing_config(self):
        """Non-existent config should return 1."""
        rc = main(["run", "/tmp/definitely_not_a_real_config_xyz.yaml"])
        assert rc == 1

    def test_legacy_runner_delegates_without_registering_again(self, config_file):
        terminal = MagicMock(
            attempt_id="legacy-wrapper-attempt",
            status=AttemptLifecycleStatus.SUCCEEDED,
            artifact_root="/tmp/legacy-wrapper-attempt",
        )
        with patch(
            "genetic_algorithm.orchestration.runner_v2.run_standard_attempt",
            return_value=terminal,
        ) as run_attempt:
            rc = legacy_runner_main(["--config", config_file, "--yes", "--no-monitor"])

        assert rc == 0
        run_attempt.assert_called_once()

    def test_legacy_runner_refuses_uncontracted_resume(self, config_file):
        with patch(
            "genetic_algorithm.orchestration.runner_v2.run_standard_attempt"
        ) as run_attempt:
            rc = legacy_runner_main(
                ["--config", config_file, "--resume", "checkpoint.json", "--yes"]
            )

        assert rc == 2
        run_attempt.assert_not_called()


# ---------------------------------------------------------------------------
# 'experiment list' command
# ---------------------------------------------------------------------------

class TestCmdExperiment:
    def test_experiment_without_subcommand_is_rejected_cleanly(self):
        assert main(["experiment"]) == 1

    def test_experiment_list_empty(self):
        with patch(
            "genetic_algorithm.orchestration.experiment_catalog_v2.ExperimentCatalogV2"
        ) as Catalog:
            Catalog.return_value.list_records.return_value = []
            rc = main(["experiment", "list"])
            assert rc == 0

    def test_experiment_show(self):
        record = MagicMock()
        record.model_dump_json.return_value = '{"experiment_id":"exp1"}'
        with patch(
            "genetic_algorithm.orchestration.experiment_catalog_v2.ExperimentCatalogV2"
        ) as Catalog:
            Catalog.return_value.get_record.return_value = record
            rc = main(["experiment", "show", "exp1"])
            assert rc == 0


# ---------------------------------------------------------------------------
# 'queue' command
# ---------------------------------------------------------------------------

class TestCmdQueue:
    def test_queue_add(self, tmp_path):
        cfg = tmp_path / "q.yaml"
        cfg.write_text(yaml.dump({"backtesting": {"pairs": ["BTC/USDT"]}}))
        prepared = MagicMock(attempt_id="queued-v2-attempt")
        with patch(
            "genetic_algorithm.orchestration.runner_v2.prepare_and_queue_standard_attempt",
            return_value=prepared,
        ) as prepare:
            rc = main(["queue", "add", str(cfg)])
            assert rc == 0
            prepare.assert_called_once()

    def test_queue_add_rejects_unpersisted_tags(self, tmp_path):
        cfg = tmp_path / "q.yaml"
        cfg.write_text(yaml.dump({"backtesting": {"pairs": ["BTC/USDT"]}}))
        with patch(
            "genetic_algorithm.orchestration.runner_v2.prepare_and_queue_standard_attempt"
        ) as prepare:
            rc = main(["queue", "add", str(cfg), "--tag", "batch1"])

        assert rc == 2
        prepare.assert_not_called()

    def test_queue_status(self):
        with patch(
            "genetic_algorithm.orchestration.attempt_state_v2.AttemptStateStoreV2"
        ) as Store:
            Store.return_value.count_attempts.return_value = 0
            Store.return_value.path = Path("/tmp/v2-state.sqlite3")
            rc = main(["queue", "status"])
            assert rc == 0


# ---------------------------------------------------------------------------
# 'monitor --once' command
# ---------------------------------------------------------------------------

class TestCmdMonitor:
    def test_monitor_once(self):
        with patch(
            "genetic_algorithm.orchestration.monitor.ExperimentMonitor"
        ) as Monitor:
            rc = main(
                [
                    "monitor",
                    "--once",
                    "--filter",
                    "failed",
                    "--tag",
                    "wave-42",
                    "--include-legacy",
                ]
            )

        assert rc == 0
        Monitor.assert_called_once_with(
            experiment_id=None,
            state_path=None,
            include_legacy=True,
            statuses=["failed"],
            tags=["wave-42"],
        )
        Monitor.return_value.snapshot.assert_called_once_with()

    def test_monitor_all_includes_non_active_catalog_states(self):
        with patch(
            "genetic_algorithm.orchestration.monitor.ExperimentMonitor"
        ) as Monitor:
            rc = main(["monitor", "--once", "--filter", "all"])

        assert rc == 0
        assert Monitor.call_args.kwargs["statuses"] == [
            "running",
            "queued",
            "completed",
            "failed",
            "cancelled",
            "unknown",
        ]


# ---------------------------------------------------------------------------
# 'data' command
# ---------------------------------------------------------------------------

class TestCmdData:
    def test_data_cleanup_dry_run(self):
        with patch("genetic_algorithm.orchestration.lifecycle.DataLifecycle") as MockLC:
            mock_lc = MagicMock()
            mock_lc.run.return_value = {"archived": 0, "deleted": 0, "dry_run": True}
            MockLC.return_value = mock_lc

            rc = main(["data", "cleanup", "--dry-run"])
            assert rc == 0

    def test_backfill_and_export_use_sqlite_catalog(self, tmp_path):
        legacy = tmp_path / "registry.json"
        legacy.write_text(
            json.dumps(
                {
                    "version": 1,
                    "experiments": {
                        "legacy-cli": {
                            "experiment_id": "legacy-cli",
                            "status": "completed",
                            "best_fitness": 0.7,
                        }
                    },
                }
            )
        )
        state = tmp_path / "state.sqlite3"
        destination = tmp_path / "catalog-export.json"

        first = main(
            [
                "data",
                "backfill",
                "--registry",
                str(legacy),
                "--state-db",
                str(state),
            ]
        )
        second = main(
            [
                "data",
                "backfill",
                "--registry",
                str(legacy),
                "--state-db",
                str(state),
            ]
        )
        exported = main(
            [
                "data",
                "export",
                "--output",
                str(destination),
                "--state-db",
                str(state),
            ]
        )

        assert first == second == exported == 0
        payload = json.loads(destination.read_text())
        assert payload["experiments"][0]["experiment_id"] == "legacy-cli"
