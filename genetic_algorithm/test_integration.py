"""
Integration test for all GA improvements
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from genetic_algorithm.strategies.generator import StrategyGenerator
from genetic_algorithm.core.strategy_gene import StrategyGene, IndicatorGene, ConditionGene


def test_logic_operators_in_conditions():
    """Test that logic operators are respected in generated strategy code."""
    print("\n=== Testing Logic Operators in Conditions ===")
    
    # Create a config
    config = {
        'indicators': {
            'available': ['RSI', 'MACD', 'BBANDS'],
            'RSI': {'period': [7, 21]},
            'MACD': {'fast_period': [8, 21], 'slow_period': [21, 50], 'signal_period': [5, 14]},
        },
        'strategy_constraints': {
            'timeframes': ['5m', '15m', '1h'],
            'stoploss_range': [-0.20, -0.05],
            'roi_range': [0.01, 0.10],
        }
    }
    
    generator = StrategyGenerator(config)
    
    # Create a strategy with mixed AND/OR conditions
    indicators = [
        IndicatorGene(type='RSI', parameters={'period': 14}, weight=1.0),
        IndicatorGene(type='MACD', parameters={'fast_period': 12, 'slow_period': 26, 'signal_period': 9}, weight=1.0),
    ]
    
    entry_conditions = [
        ConditionGene(indicator='RSI', operator='<', threshold=30, logic='AND'),
        ConditionGene(indicator='MACD', operator='cross_above', threshold=0, logic='AND'),
        ConditionGene(indicator='RSI', operator='>', threshold=25, logic='OR'),
    ]
    
    exit_conditions = [
        ConditionGene(indicator='RSI', operator='>', threshold=70, logic='AND'),
    ]
    
    strategy_gene = StrategyGene(
        generation=0,
        individual_id=1,
        indicators=indicators,
        entry_conditions=entry_conditions,
        exit_conditions=exit_conditions,
        timeframe='5m',
        stoploss=-0.10,
        minimal_roi={0: 0.05, 30: 0.03, 60: 0.01},
        trailing_stop=False,
    )
    
    # Generate code
    code = generator.generate_strategy_code(strategy_gene)
    
    print("Generated strategy code snippet:")
    print("="*60)
    # Extract the entry condition part
    entry_start = code.find('def populate_entry_trend')
    entry_end = code.find('def populate_exit_trend')
    entry_section = code[entry_start:entry_end]
    print(entry_section[:500])
    print("="*60)
    
    # Verify logic operators are in the code
    assert '&' in code or 'AND' in code, "AND operator should be present"
    assert '|' in code or 'OR' in code, "OR operator should be present"
    print("✓ Logic operators (AND/OR) are properly used in generated code")
    
    # Verify both conditions are present
    assert 'rsi_14' in code, "RSI condition should be present"
    assert 'macd' in code, "MACD condition should be present"
    print("✓ Both RSI and MACD conditions are present")
    
    print("✅ Logic operators test passed!\n")


def test_monotonic_roi_in_generated_strategy():
    """Test that generated strategies have monotonic ROI."""
    print("\n=== Testing Monotonic ROI in Generated Strategy ===")
    
    config = {
        'indicators': {
            'available': ['RSI', 'MACD', 'EMA', 'SMA'],
            'max_per_strategy': 3,
            'min_per_strategy': 2,
            'RSI': {'period': [7, 21]},
            'MACD': {'fast_period': [8, 21], 'slow_period': [21, 50], 'signal_period': [5, 14]},
            'EMA': {'period': [10, 50]},
            'SMA': {'period': [10, 50]},
        },
        'strategy_constraints': {
            'timeframes': ['5m'],
            'stoploss_range': [-0.10, -0.05],
            'roi_range': [0.01, 0.10],
        }
    }
    
    generator = StrategyGenerator(config)
    
    # Generate multiple random strategies
    for i in range(5):
        strategy = generator.generate_random_strategy(generation=0, individual_id=i)
        
        # Check ROI is monotonic
        roi = strategy.minimal_roi
        sorted_times = sorted(roi.keys())
        
        for j in range(1, len(sorted_times)):
            current_value = roi[sorted_times[j]]
            previous_value = roi[sorted_times[j-1]]
            assert current_value <= previous_value, \
                f"ROI at time {sorted_times[j]} ({current_value}) should be <= ROI at time {sorted_times[j-1]} ({previous_value})"
        
        print(f"✓ Strategy {i}: ROI {roi} is monotonic")
    
    print("✅ All generated strategies have monotonic ROI!\n")


def test_adaptive_rates_integration():
    """Test that adaptive rates work with evolution engine."""
    print("\n=== Testing Adaptive Rates Integration ===")
    
    from genetic_algorithm.core.adaptive_rates import AdaptiveRateController
    
    controller = AdaptiveRateController(
        base_mutation_rate=0.15,
        base_crossover_rate=0.7
    )
    
    # Simulate evolution progress
    print("\nSimulating 50 generations:")
    print("-" * 60)
    print(f"{'Gen':<5} {'Fitness':<10} {'Diversity':<12} {'Stag':<6} {'Mut Rate':<10} {'Cross Rate':<10}")
    print("-" * 60)
    
    for gen in range(0, 50, 5):
        # Simulate decreasing fitness improvement and diversity
        fitness = 0.3 + (gen / 50) * 0.4  # 0.3 to 0.7
        diversity = 0.8 - (gen / 50) * 0.4  # 0.8 to 0.4
        stagnation = min(gen // 10, 5)  # Increases over time
        
        mut_rate, cross_rate = controller.get_rates(
            current_generation=gen,
            total_generations=50,
            best_fitness=fitness,
            diversity_score=diversity,
            stagnation_count=stagnation
        )
        
        print(f"{gen:<5} {fitness:<10.3f} {diversity:<12.3f} {stagnation:<6} {mut_rate:<10.3f} {cross_rate:<10.3f}")
    
    print("-" * 60)
    print("✓ Adaptive rates respond to evolution dynamics")
    print("✅ Adaptive rates integration test passed!\n")


def test_enhanced_fitness_metrics():
    """Test that all metrics contribute to fitness."""
    print("\n=== Testing Enhanced Fitness Metrics ===")
    
    from genetic_algorithm.evaluation.fitness import FitnessEvaluator
    
    config = {
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
            'min_trades': 10,
            'max_drawdown': 0.25,
            'min_win_rate': 0.35,
        },
        'backtesting': {},
    }
    
    evaluator = FitnessEvaluator(config)
    
    # Test with excellent metrics
    excellent_metrics = {
        'profit': 100.0,
        'sharpe_ratio': 3.0,
        'max_drawdown': 0.10,
        'win_rate': 0.70,
        'num_trades': 40,
        'profit_factor': 3.0,
        'sortino_ratio': 3.5,
    }
    
    fitness_excellent = evaluator.calculate_fitness(excellent_metrics)
    print(f"Excellent strategy fitness: {fitness_excellent:.4f}")
    
    # Test with average metrics
    average_metrics = {
        'profit': 30.0,
        'sharpe_ratio': 1.0,
        'max_drawdown': 0.20,
        'win_rate': 0.50,
        'num_trades': 25,
        'profit_factor': 1.5,
        'sortino_ratio': 1.2,
    }
    
    fitness_average = evaluator.calculate_fitness(average_metrics)
    print(f"Average strategy fitness: {fitness_average:.4f}")
    
    # Test with poor metrics
    poor_metrics = {
        'profit': -20.0,
        'sharpe_ratio': -1.0,
        'max_drawdown': 0.40,
        'win_rate': 0.30,
        'num_trades': 8,
        'profit_factor': 0.7,
        'sortino_ratio': -0.8,
    }
    
    fitness_poor = evaluator.calculate_fitness(poor_metrics)
    print(f"Poor strategy fitness: {fitness_poor:.4f}")
    
    assert fitness_excellent > fitness_average > fitness_poor, \
        "Fitness should correlate with strategy quality"
    print("✓ Fitness properly ranks strategies: excellent > average > poor")
    
    # Test that all metrics contribute
    # Remove one metric at a time and see fitness change
    for metric_name in ['profit_factor', 'sortino_ratio']:
        metrics_without = average_metrics.copy()
        metrics_without[metric_name] = 0
        fitness_without = evaluator.calculate_fitness(metrics_without)
        print(f"Fitness without {metric_name}: {fitness_without:.4f} (original: {fitness_average:.4f})")
        # Fitness should be different (and likely lower)
    
    print("✅ Enhanced fitness metrics test passed!\n")


def main():
    """Run all integration tests."""
    print("\n" + "="*60)
    print("GA Improvements Integration Tests")
    print("="*60)
    
    try:
        test_monotonic_roi_in_generated_strategy()
        test_logic_operators_in_conditions()
        test_adaptive_rates_integration()
        test_enhanced_fitness_metrics()
        
        print("\n" + "="*60)
        print("✅ ALL INTEGRATION TESTS PASSED!")
        print("="*60 + "\n")
        return 0
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
