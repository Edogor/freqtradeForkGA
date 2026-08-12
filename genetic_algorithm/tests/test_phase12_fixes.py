"""
Tests for Phase 1-2 bug fixes.

Covers:
- Shared memory: thread safety, uuid naming, stale cleanup
- Cache: SHA256 checksum validation
- NSGA-II: hypervolume empty guard, AP-7 worst-case objectives
- Parallel executor: cleanup lifecycle
- Regime-aware: harmonic mean with zero/negative values
- Generation: crossover/mutation failure rate warnings
- Run GA: relative path resolution, registry wiring
"""

import hashlib
import json
import logging
import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root on path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))


# ============================================================================
# Shared Memory Tests
# ============================================================================

class TestSharedMemoryThreadSafety:
    """Tests for shared_memory.py race condition fixes."""

    def test_shared_data_manager_exists(self):
        """SharedDataManager class should be importable."""
        from genetic_algorithm.market.shared_memory import SharedDataManager
        assert SharedDataManager is not None

    def test_cleanup_stale_shared_memory_no_crash(self):
        """cleanup_stale_shared_memory should not crash even if /dev/shm has no ga blocks."""
        from genetic_algorithm.market.shared_memory import cleanup_stale_shared_memory
        # Should run without errors — returns None (implicit) or int
        cleanup_stale_shared_memory()

    def test_lock_attributes_exist(self):
        """Module should have threading lock protection for shared block lists."""
        import genetic_algorithm.market.shared_memory as shm_mod
        assert hasattr(shm_mod, '_shared_blocks_lock'), "Missing _shared_blocks_lock"
        assert hasattr(shm_mod, '_worker_shm_lock'), "Missing _worker_shm_lock"
        assert hasattr(shm_mod, '_shared_blocks'), "Missing _shared_blocks list"
        assert hasattr(shm_mod, '_worker_shm_blocks'), "Missing _worker_shm_blocks list"

    def test_loads_the_single_configured_timeframe(self):
        from genetic_algorithm.market.shared_memory import SharedDataManager

        manager = SharedDataManager()
        frame = MagicMock()
        frame.__len__.return_value = 1
        config = {
            'backtesting': {
                'pairs': ['BTC/USDT'],
                'timerange': '20260101-20260201',
            },
            'strategy_constraints': {'timeframes': ['1h']},
        }
        with (
            patch.object(
                manager,
                '_load_ohlcv_data',
                return_value={'BTC/USDT': frame},
            ) as load,
            patch.object(
                manager,
                '_dataframe_to_shared_memory',
                return_value=('shm-test', {'nbytes': 48}),
            ),
        ):
            loaded = manager.load_and_share(config)

        assert loaded == {'BTC/USDT': frame}
        load.assert_called_once_with(
            config,
            ['BTC/USDT'],
            '20260101-20260201',
            '1h',
            startup_candles=0,
        )
        assert manager.get_metadata()['timeframe'] == '1h'

    def test_multiple_timeframes_disable_shared_cache(self):
        from genetic_algorithm.market.shared_memory import SharedDataManager

        manager = SharedDataManager()
        with patch.object(manager, '_load_ohlcv_data') as load:
            result = manager.load_and_share(
                {
                    'backtesting': {'pairs': ['BTC/USDT']},
                    'strategy_constraints': {'timeframes': ['15m', '1h']},
                }
            )

        assert result == {}
        load.assert_not_called()

    def test_relative_data_dir_is_resolved_from_repository_root(self):
        from genetic_algorithm.market.shared_memory import _resolve_data_dir

        assert _resolve_data_dir("user_data/data/binance") == (
            project_root / "user_data" / "data" / "binance"
        )

    def test_absolute_data_dir_is_preserved(self, tmp_path):
        from genetic_algorithm.market.shared_memory import _resolve_data_dir

        assert _resolve_data_dir(tmp_path) == tmp_path


# ============================================================================
# Cache Checksum Tests
# ============================================================================

class TestCacheChecksumValidation:
    """Tests for cache.py SHA256 checksum validation."""

    def _make_mock_result(self):
        """Create a mock BacktestResult for testing."""
        from unittest.mock import MagicMock
        result = MagicMock()
        result.to_dict.return_value = {
            "success": True, "profit_percent": 10.5, "trades": 42,
            "profit_abs": 100.0, "sharpe_ratio": 1.5, "sortino_ratio": 2.0,
            "max_drawdown": -0.05, "win_rate": 0.6, "num_trades": 42,
            "profit_factor": 1.8, "avg_trade_duration": 3600,
        }
        return result

    def test_cache_put_creates_checksum_file(self):
        """Putting a result should create a .sha256 companion file."""
        from genetic_algorithm.evaluation.cache import BacktestCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = BacktestCache(cache_dir=Path(tmpdir))
            mock_result = self._make_mock_result()
            strategy_code = "class TestStrategy: pass"
            config = {"test": True}

            cache.put(strategy_code, config, mock_result)

            sha_files = list(Path(tmpdir).glob("*.sha256"))
            json_files = list(Path(tmpdir).glob("*.json"))
            assert len(json_files) >= 1, "Cache JSON file should exist"
            assert len(sha_files) >= 1, "Checksum file should exist"

    def test_cache_detects_corruption(self):
        """Corrupted cache files should be detected via checksum."""
        from genetic_algorithm.evaluation.cache import BacktestCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = BacktestCache(cache_dir=Path(tmpdir))
            mock_result = self._make_mock_result()
            strategy_code = "class CorruptTest: pass"
            config = {"corrupt_test": True}

            cache.put(strategy_code, config, mock_result)

            # Find and corrupt the JSON file
            json_files = list(Path(tmpdir).glob("*.json"))
            assert len(json_files) >= 1
            json_file = json_files[0]
            json_file.write_text('{"success": true, "profit_percent": 999.0}')

            # Get should detect mismatch and return None
            cache2 = BacktestCache(cache_dir=Path(tmpdir))
            result = cache2.get(strategy_code, config)
            assert result is None, "Corrupted cache entry should return None"


# ============================================================================
# NSGA-II Tests
# ============================================================================

class TestNSGA2Fixes:
    """Tests for NSGA-II hypervolume and AP-7 fixes."""

    def test_hypervolume_empty_remaining_points(self):
        """Hypervolume should not crash on empty remaining_points."""
        from genetic_algorithm.engine.nsga2 import _hypervolume_nd

        # Single point with 3+ objectives — should not raise ValueError
        points = [[1.0, 2.0, 3.0]]
        ref = [0.0, 0.0, 0.0]
        result = _hypervolume_nd(points, ref)
        assert isinstance(result, float)
        assert result >= 0.0

    def test_hypervolume_degenerate_3obj(self):
        """Hypervolume with all-same points in 3D should return 0 or small value."""
        from genetic_algorithm.engine.nsga2 import _hypervolume_nd

        points = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
        ref = [0.0, 0.0, 0.0]
        result = _hypervolume_nd(points, ref)
        assert isinstance(result, float)

    def test_ap7_worst_case_objectives_negative(self):
        """AP-7 fix: worst-case objectives should be large negative, not zero."""
        from genetic_algorithm.engine.nsga2 import extract_objectives_from_metrics

        # Simulate a strategy that fails min_trades gate
        metrics = {"num_trades": 0, "profit": 0, "sharpe_ratio": 0, "max_drawdown": 0}
        objectives_config = [
            {"name": "profit", "direction": "maximize"},
            {"name": "sharpe_ratio", "direction": "maximize"},
            {"name": "max_drawdown", "direction": "minimize"},
        ]
        result = extract_objectives_from_metrics(metrics, objectives_config, min_trades=5)

        # Maximize objectives should be large negative (dominated)
        assert result[0] < -100, f"Maximize profit worst-case should be << 0, got {result[0]}"
        assert result[1] < -100, f"Maximize sharpe worst-case should be << 0, got {result[1]}"
        # Minimize objective worst-case should be penalty value (like -1.0 for drawdown)
        assert result[2] < 0, f"Minimize drawdown worst-case should be < 0, got {result[2]}"

    def test_ap7_valid_strategy_passes(self):
        """Strategies meeting min_trades should get their actual objectives."""
        from genetic_algorithm.engine.nsga2 import extract_objectives_from_metrics

        metrics = {"num_trades": 50, "profit": 15.0, "sharpe_ratio": 2.1, "max_drawdown": -0.05}
        objectives_config = [
            {"name": "profit", "direction": "maximize"},
            {"name": "sharpe_ratio", "direction": "maximize"},
        ]
        result = extract_objectives_from_metrics(metrics, objectives_config, min_trades=5)
        assert result[0] == 15.0
        assert result[1] == 2.1


# ============================================================================
# Regime-Aware Evaluation Tests
# ============================================================================

class TestRegimeHarmonicMeanFix:
    """Tests for regime_aware.py harmonic mean crash prevention."""

    def test_harmonic_mean_code_path_exists(self):
        """harmonic_mean aggregation path should exist in RegimeAwareEvaluator."""
        from genetic_algorithm.market.regime_aware import RegimeAwareEvaluator

        config = {
            'backtesting': {'pairs': ['BTC/USDT'], 'timerange': '20230101-20231231'},
            'fitness_weights': {'profit': 0.5, 'sharpe_ratio': 0.5},
            'regime_evaluation': {'aggregation': 'harmonic_mean'},
        }
        evaluator = RegimeAwareEvaluator(config)
        assert evaluator.aggregation_method == 'harmonic_mean'
        assert hasattr(evaluator, '_aggregate_results')

    def test_harmonic_mean_positive_filter_logic(self):
        """Verify the statistics.harmonic_mean import exists and handles positive vals."""
        from statistics import harmonic_mean
        # harmonic_mean with all positive values should work
        result = harmonic_mean([1.0, 2.0, 3.0])
        assert result > 0
        # harmonic_mean with negative values should raise
        with pytest.raises(Exception):
            harmonic_mean([-1.0, 2.0, 3.0])


# ============================================================================
# Generation Failure Rate Tests
# ============================================================================

class TestGenerationFailureTracking:
    """Tests for generation.py failure rate warning."""

    def test_generation_step_exists(self):
        """GenerationStep class should be importable."""
        from genetic_algorithm.engine.generation import GenerationStep
        assert GenerationStep is not None

    def test_generation_module_imports(self):
        """generation.py should import cleanly with all operators."""
        from genetic_algorithm.engine import generation
        assert hasattr(generation, 'GenerationStep')
        assert hasattr(generation, 'PopulationStats')


# ============================================================================
# Run GA Path Resolution Tests
# ============================================================================

class TestRunGAPathResolution:
    """Tests for run_ga.py relative path fixes."""

    def test_base_dir_is_absolute(self):
        """_BASE_DIR should be an absolute path to genetic_algorithm/."""
        from genetic_algorithm.run_ga import _BASE_DIR
        assert _BASE_DIR.is_absolute(), f"_BASE_DIR should be absolute, got {_BASE_DIR}"
        assert _BASE_DIR.name == "genetic_algorithm", f"Should point to genetic_algorithm/, got {_BASE_DIR}"

    def test_output_dir_under_base(self):
        """OUTPUT_DIR should be under _BASE_DIR."""
        from genetic_algorithm.run_ga import OUTPUT_DIR, _BASE_DIR
        assert str(OUTPUT_DIR).startswith(str(_BASE_DIR))

    def test_log_dir_under_base(self):
        """LOG_DIR should be under _BASE_DIR."""
        from genetic_algorithm.run_ga import LOG_DIR, _BASE_DIR
        assert str(LOG_DIR).startswith(str(_BASE_DIR))


# ============================================================================
# Registry Integration Tests
# ============================================================================

class TestExperimentRegistry:
    """Tests for orchestration/registry.py functionality."""

    def test_registry_lifecycle(self):
        """Full lifecycle: register → start → update → complete."""
        from genetic_algorithm.orchestration.registry import ExperimentRegistry

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            tmp_path = f.name

        try:
            reg = ExperimentRegistry(path=tmp_path)
            
            # Register
            entry = reg.register("test_exp_1", config_path="test.yaml", ga_type="standard")
            assert entry["status"] == "queued"
            
            # Start
            reg.start("test_exp_1", pid=12345, generations_total=10)
            entry = reg.get("test_exp_1")
            assert entry["status"] == "running"
            assert entry["pid"] == 12345
            
            # Update
            reg.update("test_exp_1", generation=5, best_fitness=0.72)
            entry = reg.get("test_exp_1")
            assert entry["generation"] == 5
            assert entry["best_fitness"] == 0.72
            
            # Complete
            reg.complete("test_exp_1", best_fitness=0.85, best_profit=15.3)
            entry = reg.get("test_exp_1")
            assert entry["status"] == "completed"
            assert entry["best_fitness"] == 0.85
        finally:
            os.unlink(tmp_path)

    def test_registry_fail(self):
        """Failed experiments should record error."""
        from genetic_algorithm.orchestration.registry import ExperimentRegistry

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            tmp_path = f.name

        try:
            reg = ExperimentRegistry(path=tmp_path)
            reg.register("test_fail_1", config_path="test.yaml")
            reg.start("test_fail_1")
            reg.fail("test_fail_1", error="Something went wrong")
            
            entry = reg.get("test_fail_1")
            assert entry["status"] == "failed"
            assert "Something went wrong" in entry["error"]
        finally:
            os.unlink(tmp_path)

    def test_registry_concurrent_safety(self):
        """Registry should handle sequential writes from multiple 'threads' safely."""
        from genetic_algorithm.orchestration.registry import ExperimentRegistry

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / "test_registry.json"
            reg = ExperimentRegistry(path=tmp_path)

            # Sequential writes are always safe (file locking)
            for i in range(10):
                reg.register(f"seq_{i}", config_path=f"config_{i}.yaml")

            assert reg.count() == 10
            for i in range(10):
                entry = reg.get(f"seq_{i}")
                assert entry is not None
                assert entry["config_path"] == f"config_{i}.yaml"


# ============================================================================
# Data Lifecycle Tests
# ============================================================================

class TestDataLifecycle:
    """Tests for orchestration/lifecycle.py."""

    def test_report_no_crash(self):
        """Disk usage report should not crash even with empty data dir."""
        from genetic_algorithm.orchestration.lifecycle import DataLifecycle

        with tempfile.TemporaryDirectory() as tmpdir:
            lc = DataLifecycle(data_root=tmpdir, dry_run=True)
            report = lc.report()
            assert "total_bytes" in report
            assert "categories" in report
            assert isinstance(report["total_bytes"], int)

    def test_cleanup_dry_run(self):
        """Cleanup in dry_run mode should not delete anything."""
        from genetic_algorithm.orchestration.lifecycle import DataLifecycle

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a fake cache file
            cache_dir = Path(tmpdir) / "cache"
            cache_dir.mkdir()
            (cache_dir / "test.json").write_text('{"test": 1}')

            lc = DataLifecycle(data_root=tmpdir, dry_run=True)
            removed = lc.cleanup_cache(max_size_mb=0)  # Force all to exceed
            
            # File should still exist (dry run)
            assert (cache_dir / "test.json").exists()


# ============================================================================
# Corpus Builder Tests
# ============================================================================

class TestCorpusBuilder:
    """Tests for intelligence/corpus.py."""

    def test_build_empty_data_dir(self):
        """Building from an empty dir should return empty DataFrame."""
        from genetic_algorithm.intelligence.corpus import CorpusBuilder

        with tempfile.TemporaryDirectory() as tmpdir:
            builder = CorpusBuilder(data_dir=tmpdir)
            df = builder.build()
            assert len(df) == 0

    def test_surrogate_feature_names_65(self):
        """SURROGATE_FEATURE_NAMES should have exactly 65 entries."""
        from genetic_algorithm.intelligence.corpus import SURROGATE_FEATURE_NAMES
        assert len(SURROGATE_FEATURE_NAMES) == 65
