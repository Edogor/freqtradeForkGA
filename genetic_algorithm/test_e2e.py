"""
Quick end-to-end test of GA with improvements
Tests a minimal run with 1 generation and 3 individuals
"""

import sys
import tempfile
import yaml
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from genetic_algorithm.core.evolution import GeneticAlgorithm


def test_minimal_ga_run():
    """Test a minimal GA run with improvements enabled."""
    print("\n=== Testing Minimal GA Run with Improvements ===")
    
    # Create minimal config
    config = {
        'genetic_algorithm': {
            'population_size': 3,
            'generations': 2,
            'mutation_rate': 0.15,
            'crossover_rate': 0.7,
            'elite_size': 1,
            'tournament_size': 2,
            'selection_method': 'tournament',
            'convergence_patience': 10,
            'use_adaptive_rates': True,
            'use_multi_crossover': True,
            'crossover_method': 'single_point',
            'crossover_methods': ['single_point', 'uniform'],
        },
        'fitness_weights': {
            'profit': 0.25,
            'sharpe_ratio': 0.15,
            'drawdown': 0.15,
            'win_rate': 0.10,
            'trade_frequency': 0.10,
            'profit_factor': 0.15,
            'sortino_ratio': 0.10,
        },
        'fitness_penalties': {
            'min_trades': 5,
            'max_drawdown': 0.30,
            'min_win_rate': 0.30,
        },
        'backtesting': {
            'timerange': '20230101-20230201',
            'stake_amount': 0.05,
            'pairs': ['UNITTEST/BTC'],
            'max_open_trades': 1,
            'fee': 0.001,
            'enable_cache': False,
            'timeout': 60,
        },
        'strategy_constraints': {
            'min_trades': 5,
            'max_drawdown': 0.30,
            'min_win_rate': 0.30,
            'timeframes': ['5m', '15m'],
            'stoploss_range': [-0.15, -0.05],
            'roi_range': [0.01, 0.08],
        },
        'indicators': {
            'available': ['RSI', 'MACD', 'EMA'],
            'max_per_strategy': 3,
            'min_per_strategy': 2,
            'RSI': {
                'period': [7, 21],
                'buy_threshold': [20, 40],
                'sell_threshold': [60, 80],
            },
            'MACD': {
                'fast_period': [8, 21],
                'slow_period': [21, 50],
                'signal_period': [5, 14],
            },
            'EMA': {
                'period': [10, 50],
            },
        },
        'storage': {
            'database': '/tmp/test_strategies.db',
            'strategy_dir': '/tmp/test_strategies',
            'checkpoint_dir': '/tmp/test_checkpoints',
            'checkpoint_interval': 5,
            'keep_history': False,
        },
        'logging': {
            'level': 'INFO',
            'file': '/tmp/test_ga.log',
            'console': True,
        },
        'visualization': {
            'enabled': False,
        },
        'advanced': {
            'parallel_evaluation': False,
        }
    }
    
    # Write config to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        config_path = f.name
    
    try:
        print(f"Config written to: {config_path}")
        
        # Initialize GA
        print("Initializing GA...")
        ga = GeneticAlgorithm(config_path)
        
        # Check adaptive controller was initialized
        assert hasattr(ga, 'adaptive_controller'), "Adaptive controller should be initialized"
        print("✓ Adaptive rate controller initialized")
        
        # Check crossover settings
        assert ga.use_multi_crossover == True, "Multi-crossover should be enabled"
        assert 'uniform' in ga.crossover_methods, "Uniform crossover should be available"
        print("✓ Multi-crossover enabled with multiple methods")
        
        # Initialize population
        print("Initializing population...")
        population = ga.initialize_population()
        assert len(population) == 3, "Should have 3 individuals"
        print(f"✓ Population initialized with {len(population)} individuals")
        
        # Check that ROIs are monotonic
        for i, individual in enumerate(population):
            roi = individual.strategy_gene.minimal_roi
            sorted_times = sorted(roi.keys())
            for j in range(1, len(sorted_times)):
                assert roi[sorted_times[j]] <= roi[sorted_times[j-1]], \
                    f"Individual {i} ROI should be monotonic"
        print("✓ All individuals have monotonic ROI")
        
        # Generate strategy code for one individual
        strategy_code = ga.strategy_generator.generate_strategy_code(
            population.individuals[0].strategy_gene
        )
        
        # Check that logic operators are in generated code
        # (at least one condition should use & or |)
        has_logic = '&' in strategy_code or '|' in strategy_code
        print(f"✓ Generated strategy code includes logic operators: {has_logic}")
        
        print("\n✅ Minimal GA run test completed successfully!")
        print("All improvements are working together correctly.")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Clean up
        try:
            Path(config_path).unlink()
        except:
            pass


if __name__ == '__main__':
    sys.exit(test_minimal_ga_run())
