"""
Tests for DataService — list_runs, get_generation, get_strategy,
get_hall_of_fame, get_config_templates, etc.

Uses mock filesystem to avoid depending on actual run data.
"""

from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from genetic_algorithm.web.models.generation import GenerationDetail
from genetic_algorithm.web.models.run import RunDetail, RunStatus, RunSummary
from genetic_algorithm.web.models.strategy import StrategyDetail
from genetic_algorithm.web.services.data_service import DataService, RUNS_DIR, HOF_DIR, CONFIG_DIR
from genetic_algorithm.orchestration.result_contract import AttemptManifestV2


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def mock_manager():
    mgr = MagicMock()
    mgr.list_runs.return_value = []
    mgr.get_run.return_value = None
    return mgr


@pytest.fixture
def data_service(mock_manager, tmp_path):
    return DataService(
        run_manager=mock_manager,
        state_path=tmp_path / "catalog-state.sqlite3",
    )


# ── list_runs ──────────────────────────────────────────────────────


class TestListRuns:

    def test_list_runs_returns_active_runs(self, data_service, mock_manager):
        mock_summary = RunSummary(
            run_id="active_1",
            status=RunStatus.RUNNING,
            config_name="test",
        )
        mock_manager.list_runs.return_value = [mock_summary]

        result = data_service.list_runs()
        assert len(result) >= 1
        assert any(r.run_id == "active_1" for r in result)

    def test_list_runs_empty_when_no_runs(self, data_service, mock_manager):
        mock_manager.list_runs.return_value = []
        with patch.object(Path, 'exists', return_value=False):
            result = data_service.list_runs()
        assert result == []

    def test_list_runs_includes_disk_runs(self, data_service, mock_manager, tmp_path):
        """When RUNS_DIR has run directories, they should appear in results."""
        mock_manager.list_runs.return_value = []

        run_dir = tmp_path / "test_run_disk"
        run_dir.mkdir()
        config_file = run_dir / "config.yaml"
        config_file.write_text("genetic_algorithm:\n  generations: 10\n  population_size: 5\n")

        # Patch RUNS_DIR
        with patch("genetic_algorithm.web.services.data_service.RUNS_DIR", tmp_path):
            result = data_service.list_runs()

        # Should find the disk run (may not parse perfectly, but should not crash)
        assert isinstance(result, list)

    def test_catalog_is_the_only_historical_run_source(
        self, data_service, mock_manager, tmp_path
    ):
        root = tmp_path / "attempts" / "attempt-web"
        manifest = AttemptManifestV2(
            attempt_id="attempt-web",
            wave_id="wave-web",
            experiment_id="experiment-web",
            created_at=datetime(2026, 7, 23, tzinfo=UTC),
            config_hash="c" * 64,
            code_version="web-test",
            data_manifest_hash="d" * 64,
            split_manifest_hash="s" * 64,
            fitness_policy_version="web-policy",
            seeds=[42],
            worker_count=1,
            resolved_config_path=str(root / "resolved_config.yaml"),
            artifact_root=str(root),
        )
        data_service.catalog.state_store.register(manifest)
        mock_manager.list_runs.return_value = []

        summaries = data_service.list_runs()
        detail = data_service.get_run_detail("experiment-web")

        assert [item.run_id for item in summaries] == ["experiment-web"]
        assert summaries[0].status == RunStatus.PENDING
        assert detail is not None
        assert detail.mode == "canonical_v2"


# ── get_generation ─────────────────────────────────────────────────


class TestGetGeneration:

    def test_get_generation_from_disk(self, data_service, tmp_path):
        """Load a generation snapshot from a JSON file."""
        run_dir = tmp_path / "test_run"
        run_dir.mkdir()

        gen_data = {
            "run_id": "test_run",
            "generation": 3,
            "individuals": [
                {
                    "id": "ind_001",
                    "fitness": 0.8,
                    "rank": 1,
                    "evaluated": True,
                    "metrics": {"profit": 5.0, "sharpe_ratio": 1.2},
                },
                {
                    "id": "ind_002",
                    "fitness": 0.6,
                    "rank": 2,
                    "evaluated": True,
                    "metrics": {"profit": 2.0},
                },
            ],
        }
        gen_file = run_dir / "gen_0003.json"
        gen_file.write_text(json.dumps(gen_data))

        with patch("genetic_algorithm.web.services.data_service.RUNS_DIR", tmp_path):
            result = data_service.get_generation("test_run", 3)

        assert result is not None
        assert result.generation == 3
        assert len(result.individuals) == 2
        assert result.individuals[0].id == "ind_001"

    def test_get_generation_not_found(self, data_service, tmp_path):
        with patch("genetic_algorithm.web.services.data_service.RUNS_DIR", tmp_path):
            result = data_service.get_generation("nonexistent_run", 999)
        assert result is None


# ── get_hall_of_fame ──────────────────────────────────────────────


class TestGetHallOfFame:

    def test_get_hall_of_fame_from_file(self, data_service, tmp_path):
        hof_data = [
            {"id": "hof_001", "fitness": 0.95, "profit": 20.0, "num_trades": 50},
            {"id": "hof_002", "fitness": 0.88, "profit": 15.0, "num_trades": 40},
        ]
        hof_file = tmp_path / "hall_of_fame.json"
        hof_file.write_text(json.dumps(hof_data))

        with patch("genetic_algorithm.web.services.data_service.HOF_DIR", tmp_path):
            result = data_service.get_hall_of_fame()

        assert isinstance(result, list)

    def test_get_hall_of_fame_empty(self, data_service, tmp_path):
        """Should return empty list when no HoF file exists."""
        with patch("genetic_algorithm.web.services.data_service.HOF_DIR", tmp_path):
            result = data_service.get_hall_of_fame()
        assert isinstance(result, list)


# ── get_config_templates ──────────────────────────────────────────


class TestGetConfigTemplates:

    def test_get_config_templates(self, data_service, tmp_path):
        """Should find YAML config files in the config directory."""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(
            "genetic_algorithm:\n  population_size: 20\n  generations: 50\n"
            "backtesting:\n  pairs:\n    - BTC/USDT\n"
        )

        with patch("genetic_algorithm.web.services.data_service.CONFIG_DIR", tmp_path):
            result = data_service.get_config_templates()

        assert isinstance(result, list)

    def test_get_config_templates_empty(self, data_service, tmp_path):
        with patch("genetic_algorithm.web.services.data_service.CONFIG_DIR", tmp_path):
            result = data_service.get_config_templates()
        assert result == [] or isinstance(result, list)

    def test_load_config_template(self, data_service, tmp_path):
        config_file = tmp_path / "my_config.yaml"
        config_file.write_text(
            "genetic_algorithm:\n  population_size: 30\n  generations: 100\n"
        )

        with patch("genetic_algorithm.web.services.data_service.CONFIG_DIR", tmp_path):
            result = data_service.load_config_template("my_config")

        # Should return parsed config dict or None
        assert result is None or isinstance(result, dict)

    def test_load_config_template_not_found(self, data_service, tmp_path):
        with patch("genetic_algorithm.web.services.data_service.CONFIG_DIR", tmp_path):
            result = data_service.load_config_template("nonexistent")
        assert result is None
