"""Tests for the unified CLI (cli.py) and its integration with the registry."""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from genetic_algorithm.cli import main, _detect_ga_type


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
        """run command should register, start, call engine, then complete."""
        mock_ga_instance = MagicMock()
        mock_ga_instance.evolve.return_value = []

        with patch("genetic_algorithm.orchestration.registry.ExperimentRegistry") as MockReg, \
             patch("genetic_algorithm.core.evolution.GeneticAlgorithm", return_value=mock_ga_instance):
            mock_reg = MagicMock()
            MockReg.return_value = mock_reg

            rc = main(["run", config_file, "--name", "test_exp", "--no-monitor", "--yes"])

            assert rc == 0
            mock_reg.register.assert_called_once()
            call_kwargs = mock_reg.register.call_args
            assert call_kwargs[0][0] == "test_exp"

            mock_reg.start.assert_called_once()
            mock_ga_instance.evolve.assert_called_once()
            mock_reg.complete.assert_called_once()

    def test_run_marks_fail_on_exception(self, config_file, registry_path):
        """If the engine raises, the experiment should be marked failed."""
        mock_ga_instance = MagicMock()
        mock_ga_instance.evolve.side_effect = RuntimeError("boom")

        with patch("genetic_algorithm.orchestration.registry.ExperimentRegistry") as MockReg, \
             patch("genetic_algorithm.core.evolution.GeneticAlgorithm", return_value=mock_ga_instance):
            mock_reg = MagicMock()
            MockReg.return_value = mock_reg

            # The CLI catches the exception and returns 1
            rc = main(["run", config_file, "--name", "fail_exp", "--no-monitor", "--yes"])
            assert rc == 1

            mock_reg.fail.assert_called_once()
            assert "boom" in str(mock_reg.fail.call_args)

    def test_run_missing_config(self):
        """Non-existent config should return 1."""
        rc = main(["run", "/tmp/definitely_not_a_real_config_xyz.yaml"])
        assert rc == 1


# ---------------------------------------------------------------------------
# 'experiment list' command
# ---------------------------------------------------------------------------

class TestCmdExperiment:
    def test_experiment_list_empty(self):
        with patch("genetic_algorithm.orchestration.registry.ExperimentRegistry") as MockReg:
            mock_reg = MagicMock()
            mock_reg.list.return_value = []
            MockReg.return_value = mock_reg

            rc = main(["experiment", "list"])
            assert rc == 0

    def test_experiment_show(self):
        with patch("genetic_algorithm.orchestration.registry.ExperimentRegistry") as MockReg:
            mock_reg = MagicMock()
            mock_reg.get.return_value = {
                "experiment_id": "exp1",
                "status": "running",
                "tags": ["t1"],
            }
            MockReg.return_value = mock_reg

            rc = main(["experiment", "show", "exp1"])
            assert rc == 0


# ---------------------------------------------------------------------------
# 'queue' command
# ---------------------------------------------------------------------------

class TestCmdQueue:
    def test_queue_add(self, tmp_path):
        cfg = tmp_path / "q.yaml"
        cfg.write_text(yaml.dump({"backtesting": {"pairs": ["BTC/USDT"]}}))

        with patch("genetic_algorithm.orchestration.registry.ExperimentRegistry") as MockReg:
            mock_reg = MagicMock()
            MockReg.return_value = mock_reg

            rc = main(["queue", "add", str(cfg), "--tag", "batch1"])
            assert rc == 0
            mock_reg.register.assert_called_once()

    def test_queue_status(self):
        with patch("genetic_algorithm.orchestration.registry.ExperimentRegistry") as MockReg:
            mock_reg = MagicMock()
            mock_reg.summary.return_value = {"queued": 2, "running": 1}
            MockReg.return_value = mock_reg

            rc = main(["queue", "status"])
            assert rc == 0


# ---------------------------------------------------------------------------
# 'monitor --once' command
# ---------------------------------------------------------------------------

class TestCmdMonitor:
    def test_monitor_once(self):
        with patch("genetic_algorithm.orchestration.registry.ExperimentRegistry") as MockReg, \
             patch("os.system"):
            mock_reg = MagicMock()
            mock_reg.list.return_value = []
            mock_reg.summary.return_value = {}
            MockReg.return_value = mock_reg

            rc = main(["monitor", "--once"])
            assert rc == 0


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
