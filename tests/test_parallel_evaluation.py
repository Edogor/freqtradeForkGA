"""
Tests for Parallel Evaluation module.

This test suite verifies the parallel evaluation functionality,
including worker initialization, parallel batch evaluation, and
proper integration with the evolution process.
"""

import os
import pytest
import time
from collections import OrderedDict
from concurrent.futures import Future
from unittest.mock import MagicMock, patch
from typing import Dict, Any, List

# Import the modules to test
from genetic_algorithm.evaluation.parallel import (
    ParallelEvaluator,
    ParallelEvaluationResult,
    is_parallel_available,
    get_recommended_workers,
    _evaluate_strategy_in_worker,
    _init_worker,
    _isolate_worker_strategy_directory,
    _prepopulate_shared_bt_data_cache,
)
from genetic_algorithm.core.strategy_gene import StrategyGene, IndicatorGene, ConditionGene
from genetic_algorithm.core.individual import Individual


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def minimal_config() -> Dict[str, Any]:
    """Minimal configuration for testing parallel evaluation."""
    return {
        'genetic_algorithm': {
            'mode': 'single_objective',
            'population_size': 10,
            'generations': 5,
        },
        'parallel_evaluation': {
            'enabled': True,
            'num_workers': 2,
        },
        'backtesting': {
            'timerange': '20180110-20180130',
            'stake_amount': 0.1,
            'pairs': ['UNITTEST/BTC'],
            'max_open_trades': 3,
            'fee': 0.001,
        },
        'walk_forward': {
            'enabled': False,
        },
        'fitness_weights': {
            'profit': 0.3,
            'sharpe_ratio': 0.2,
            'drawdown': 0.2,
            'win_rate': 0.15,
            'trade_frequency': 0.15,
        },
        'fitness_penalties': {
            'min_trades': 5,
            'max_drawdown': 0.30,
            'min_win_rate': 0.30,
            'complexity_weight': 0.01,
        },
        'strategy_constraints': {
            'timeframes': ['5m', '15m', '1h'],
        },
    }


@pytest.fixture
def sample_strategy_gene() -> StrategyGene:
    """Create a sample strategy gene for testing."""
    return StrategyGene(
        generation=0,
        individual_id=0,
        indicators=[
            IndicatorGene(type='RSI', parameters={'period': 14}, instance_id='RSI_0'),
            IndicatorGene(type='EMA', parameters={'period': 20}, instance_id='EMA_0'),
        ],
        entry_conditions=[
            ConditionGene(indicator='RSI_0', operator='<', threshold=30, logic='AND'),
        ],
        exit_conditions=[
            ConditionGene(indicator='RSI_0', operator='>', threshold=70, logic='AND'),
        ],
        timeframe='5m',
        stoploss=-0.10,
        minimal_roi={"0": 0.04, "30": 0.02, "60": 0.01},
    )


@pytest.fixture
def sample_individuals(sample_strategy_gene) -> List[Individual]:
    """Create a list of sample individuals for testing."""
    individuals = []
    for i in range(5):
        gene = sample_strategy_gene.copy()
        gene.individual_id = i
        ind = Individual(strategy_gene=gene)
        individuals.append(ind)
    return individuals


# ============================================================================
# Unit Tests - Utility Functions
# ============================================================================

class TestUtilityFunctions:
    """Tests for utility functions."""
    
    def test_is_parallel_available(self):
        """Test that parallel availability check works."""
        # This should return True on most systems
        result = is_parallel_available()
        assert isinstance(result, bool)
    
    def test_get_recommended_workers(self):
        """Test recommended worker count calculation."""
        workers = get_recommended_workers()
        assert isinstance(workers, int)
        assert workers >= 1
        # Should be less than or equal to CPU count
        assert workers <= os.cpu_count()


# ============================================================================
# Unit Tests - ParallelEvaluator
# ============================================================================

class TestParallelEvaluatorInit:
    """Tests for ParallelEvaluator initialization."""
    
    def test_init_with_explicit_workers(self, minimal_config):
        """Test initialization with explicit worker count."""
        evaluator = ParallelEvaluator(minimal_config, num_workers=4)
        assert evaluator.num_workers == 4
    
    def test_init_with_config_workers(self, minimal_config):
        """Test initialization with config-specified worker count."""
        minimal_config['parallel_evaluation']['num_workers'] = 3
        evaluator = ParallelEvaluator(minimal_config)
        assert evaluator.num_workers == 3
    
    def test_init_auto_workers(self, minimal_config):
        """Test initialization with auto-detected worker count."""
        minimal_config['parallel_evaluation']['num_workers'] = None
        evaluator = ParallelEvaluator(minimal_config)
        assert evaluator.num_workers >= 1
        assert evaluator.num_workers <= os.cpu_count()
    
    def test_init_nsga2_mode(self, minimal_config):
        """Test initialization in NSGA-II mode."""
        minimal_config['genetic_algorithm']['mode'] = 'nsga2'
        minimal_config['nsga2'] = {
            'objectives': [
                {'name': 'profit', 'type': 'maximize', 'scale': 100.0},
                {'name': 'max_drawdown', 'type': 'minimize', 'scale': 1.0},
            ]
        }
        evaluator = ParallelEvaluator(minimal_config)
        assert evaluator.nsga2_mode is True
        assert len(evaluator.objectives_config) == 2


# ============================================================================
# Unit Tests - Strategy Gene Serialization
# ============================================================================

class TestStrategyGeneSerialization:
    """Tests for strategy gene serialization (required for multiprocessing)."""
    
    def test_to_dict_roundtrip(self, sample_strategy_gene):
        """Test that strategy gene survives dict serialization."""
        gene_dict = sample_strategy_gene.to_dict()
        restored = StrategyGene.from_dict(gene_dict)
        
        assert restored.generation == sample_strategy_gene.generation
        assert restored.individual_id == sample_strategy_gene.individual_id
        assert len(restored.indicators) == len(sample_strategy_gene.indicators)
        assert len(restored.entry_conditions) == len(sample_strategy_gene.entry_conditions)
        assert restored.timeframe == sample_strategy_gene.timeframe
        assert restored.stoploss == sample_strategy_gene.stoploss
    
    def test_dict_is_pickleable(self, sample_strategy_gene):
        """Test that strategy gene dict can be pickled (required for multiprocessing)."""
        import pickle
        
        gene_dict = sample_strategy_gene.to_dict()
        pickled = pickle.dumps(gene_dict)
        unpickled = pickle.loads(pickled)
        
        assert unpickled == gene_dict


# ============================================================================
# Unit Tests - Worker Function
# ============================================================================

class TestWorkerFunction:
    """Tests for the worker evaluation function."""
    
    def test_worker_not_initialized_returns_error(self, sample_strategy_gene):
        """Test that worker returns error when not initialized."""
        # Reset global worker state
        import genetic_algorithm.evaluation.parallel as parallel_module
        parallel_module._worker_evaluator = None
        
        gene_dict = sample_strategy_gene.to_dict()
        result = _evaluate_strategy_in_worker(gene_dict, 0)
        
        assert result['success'] is False
        assert 'not initialized' in result['error'].lower()

    def test_metrics_error_is_returned_as_worker_failure(
        self, sample_strategy_gene, monkeypatch
    ):
        """Evaluator failures returned in metrics must not become successes."""
        import genetic_algorithm.evaluation.parallel as parallel_module

        evaluator = MagicMock()
        evaluator.evaluate.return_value = (0.0, {'error': 'backtest failed'})
        monkeypatch.setattr(parallel_module, '_worker_evaluator', evaluator)

        result = _evaluate_strategy_in_worker(sample_strategy_gene.to_dict(), 0)

        assert result['success'] is False
        assert result['error'] == 'backtest failed'

    def test_worker_uses_batch_unique_evaluation_id(
        self, sample_strategy_gene, monkeypatch
    ):
        """Duplicate island-local ids must not share a generated class name."""
        import genetic_algorithm.evaluation.parallel as parallel_module

        observed_ids = []
        evaluator = MagicMock()

        def evaluate(gene):
            observed_ids.append(gene.individual_id)
            return 1.0, {}

        evaluator.evaluate.side_effect = evaluate
        monkeypatch.setattr(parallel_module, '_worker_evaluator', evaluator)

        source = sample_strategy_gene.to_dict()
        source['individual_id'] = 0
        result = _evaluate_strategy_in_worker(source, 17)

        assert result['success'] is True
        assert observed_ids == [17]
        assert source['individual_id'] == 0

    def test_workers_receive_process_private_strategy_directories(self, tmp_path):
        config_a = {
            'storage': {'generated_strategy_dir': str(tmp_path / 'generated')}
        }
        config_b = {
            'storage': {'generated_strategy_dir': str(tmp_path / 'generated')}
        }

        first = _isolate_worker_strategy_directory(config_a, worker_pid=101)
        second = _isolate_worker_strategy_directory(config_b, worker_pid=202)

        assert first != second
        assert first.name == 'worker-101'
        assert second.name == 'worker-202'
        assert config_a['storage']['generated_strategy_dir'] == str(first)
        assert config_b['storage']['generated_strategy_dir'] == str(second)

    def test_batch_defensively_counts_metrics_error_as_failure(
        self, minimal_config, sample_individuals
    ):
        """The parent rejects an inconsistent success flag from a worker."""
        evaluator = ParallelEvaluator(minimal_config, num_workers=1)
        future = Future()
        future.set_result({
            'index': 0,
            'fitness': 0.75,
            'metrics': {'error': 'worker-reported failure'},
            'success': True,
        })
        executor = MagicMock()
        executor.submit.return_value = future
        evaluator._check_pool_health = MagicMock(return_value=True)
        evaluator._get_executor = MagicMock(return_value=executor)

        individual = sample_individuals[0]
        result = evaluator.evaluate_batch([individual])

        assert result.successful == 0
        assert result.failed == 1
        assert individual.metrics['error'] == 'worker-reported failure'

    def test_independent_pair_shared_cache_survives_multiple_candidates(self):
        """All six singleton overrides reuse shared OHLCV across candidates."""
        development_pairs = ['BTC/USDT', 'SOL/USDT', 'XRP/USDT']
        validation_pairs = ['BNB/USDT', 'ETH/USDT', 'PEPE/USDT']
        panel = development_pairs + validation_pairs
        shared_frames = {pair: object() for pair in panel}
        config = {
            'backtesting': {'timerange': '20230509-20260327'},
            'strategy_constraints': {'timeframes': ['15m']},
            'pair_validation': {
                'enabled': True,
                'evaluation_mode': 'independent_pairs',
                'training_pairs': development_pairs,
                'validation_pairs': validation_pairs,
            },
        }
        backtester = MagicMock()
        backtester._bt_data_cache = OrderedDict()
        backtester._bt_data_cache_max = 3
        evaluator = MagicMock()
        evaluator.backtester = backtester

        entry_count = _prepopulate_shared_bt_data_cache(
            evaluator,
            config,
            shared_frames,
            {'timeframe': '15m'},
        )

        # full panel + development group + validation group + six singletons
        assert entry_count == 9
        assert len(backtester._bt_data_cache) == 9
        assert backtester._bt_data_cache_max >= 9

        disk_loads = []
        for candidate_id in range(2):
            for pair in panel:
                singleton = (pair,)
                key = (
                    singleton,
                    '20230509-20260327',
                    '15m',
                    '',
                    singleton,
                )
                cached = backtester._bt_data_cache.get(key)
                if cached is None:
                    disk_loads.append((candidate_id, pair))
                    continue
                backtester._bt_data_cache.move_to_end(key)
                data, _timerange = cached
                assert data[pair] is shared_frames[pair]

        assert disk_loads == []


# ============================================================================
# Integration Tests - Parallel Evaluation
# ============================================================================

class TestParallelEvaluationIntegration:
    """Integration tests for parallel evaluation."""
    
    @pytest.mark.slow
    def test_evaluate_batch_empty_list(self, minimal_config):
        """Test evaluating empty list returns immediately."""
        evaluator = ParallelEvaluator(minimal_config, num_workers=2)
        result = evaluator.evaluate_batch([])
        
        assert result.successful == 0
        assert result.failed == 0
        assert result.total_time == 0.0
    
    @pytest.mark.slow
    def test_evaluate_batch_marks_individuals(self, minimal_config, sample_individuals):
        """Test that batch evaluation marks individuals as evaluated."""
        # This test may be slow as it actually runs backtests
        evaluator = ParallelEvaluator(minimal_config, num_workers=2)
        
        # Take just 2 individuals for speed
        individuals = sample_individuals[:2]
        
        result = evaluator.evaluate_batch(individuals)
        
        # All should be evaluated (success or failure)
        for ind in individuals:
            assert ind.evaluated is True
        
        assert result.successful + result.failed == len(individuals)


# ============================================================================
# Unit Tests - ParallelEvaluationResult
# ============================================================================

class TestParallelEvaluationResult:
    """Tests for ParallelEvaluationResult dataclass."""
    
    def test_result_creation(self):
        """Test creating a result object."""
        result = ParallelEvaluationResult(
            successful=10,
            failed=2,
            total_time=5.5,
            speedup_estimate=3.2
        )
        
        assert result.successful == 10
        assert result.failed == 2
        assert result.total_time == 5.5
        assert result.speedup_estimate == 3.2


# ============================================================================
# Tests for Evolution Integration
# ============================================================================

class TestEvolutionIntegration:
    """Tests for integration with evolution.py."""
    
    def test_import_parallel_in_evolution(self):
        """Test that evolution.py can import parallel module."""
        from genetic_algorithm.core.evolution import ParallelEvaluator, is_parallel_available
        assert ParallelEvaluator is not None
        assert callable(is_parallel_available)


# ============================================================================
# Benchmark Tests (marked as slow)
# ============================================================================

class TestBenchmark:
    """Benchmark tests for measuring parallel speedup."""
    
    @pytest.mark.slow
    @pytest.mark.benchmark
    def test_parallel_vs_sequential_speedup(self, minimal_config, sample_individuals):
        """
        Benchmark: Compare sequential vs parallel evaluation time.
        
        This test measures the actual speedup achieved by parallel evaluation.
        It's marked as slow because it runs actual backtests.
        """
        from genetic_algorithm.evaluation.fitness import FitnessEvaluator
        
        # Use 5 individuals for benchmarking
        individuals = sample_individuals
        
        # Reset evaluation state
        for ind in individuals:
            ind.evaluated = False
            ind.fitness = None
        
        # Sequential evaluation
        seq_evaluator = FitnessEvaluator(minimal_config)
        seq_start = time.time()
        
        for ind in individuals:
            fitness, metrics = seq_evaluator.evaluate(ind.strategy_gene)
            ind.set_fitness(fitness, metrics)
        
        seq_time = time.time() - seq_start
        
        # Reset for parallel evaluation
        for ind in individuals:
            ind.evaluated = False
            ind.fitness = None
        
        # Parallel evaluation
        par_evaluator = ParallelEvaluator(minimal_config, num_workers=2)
        par_result = par_evaluator.evaluate_batch(individuals)
        par_time = par_result.total_time
        
        # Calculate speedup
        speedup = seq_time / par_time if par_time > 0 else 0
        
        print(f"\nBenchmark Results:")
        print(f"  Sequential: {seq_time:.2f}s")
        print(f"  Parallel:   {par_time:.2f}s")
        print(f"  Speedup:    {speedup:.2f}x")
        
        # Parallel should be faster with multiple individuals
        # (might not be faster with very few individuals due to overhead)
        assert par_result.successful > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
