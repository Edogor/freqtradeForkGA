"""
Simple tests for GA improvements (no pytest dependency)
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from genetic_algorithm.utils.roi_helper import (
    generate_monotonic_roi, is_monotonic_roi, fix_monotonic_roi, mutate_roi
)
from genetic_algorithm.core.adaptive_rates import AdaptiveRateController


def test_monotonic_roi():
    """Test ROI generation and validation."""
    print("\n=== Testing Monotonic ROI ===")
    
    # Test generation
    roi = generate_monotonic_roi()
    print(f"Generated ROI: {roi}")
    assert is_monotonic_roi(roi), "Generated ROI should be monotonic"
    print("✓ ROI is monotonically decreasing")
    
    # Test detection
    good_roi = {0: 0.10, 30: 0.06, 60: 0.03, 120: 0.01}
    assert is_monotonic_roi(good_roi), "Good ROI should be validated"
    print("✓ Valid ROI correctly identified")
    
    bad_roi = {0: 0.05, 30: 0.10, 60: 0.03}
    assert not is_monotonic_roi(bad_roi), "Bad ROI should be detected"
    print("✓ Invalid ROI correctly detected")
    
    # Test fixing
    fixed_roi = fix_monotonic_roi(bad_roi)
    assert is_monotonic_roi(fixed_roi), "Fixed ROI should be monotonic"
    print(f"✓ Fixed ROI: {fixed_roi}")
    
    # Test mutation
    for i in range(5):
        mutated = mutate_roi(good_roi)
        assert is_monotonic_roi(mutated), f"Mutation {i+1} should preserve monotonic property"
    print("✓ All 5 mutations preserved monotonic property")
    
    print("✅ All ROI tests passed!\n")


def test_adaptive_rates():
    """Test adaptive rate controller."""
    print("\n=== Testing Adaptive Rates ===")
    
    controller = AdaptiveRateController(base_mutation_rate=0.15, base_crossover_rate=0.7)
    print(f"Initialized controller: mut={controller.base_mutation_rate}, cross={controller.base_crossover_rate}")
    
    # Test early generation
    mut1, cross1 = controller.get_rates(
        current_generation=1,
        total_generations=50,
        best_fitness=0.5,
        diversity_score=0.8,
        stagnation_count=0
    )
    print(f"Early generation (1/50): mut={mut1:.3f}, cross={cross1:.3f}")
    
    # Test late generation
    mut2, cross2 = controller.get_rates(
        current_generation=45,
        total_generations=50,
        best_fitness=0.8,
        diversity_score=0.6,
        stagnation_count=0
    )
    print(f"Late generation (45/50): mut={mut2:.3f}, cross={cross2:.3f}")
    assert mut2 <= mut1, "Late generation should have lower or equal mutation"
    print("✓ Mutation rate decreases over generations")
    
    # Test with stagnation
    mut3, cross3 = controller.get_rates(
        current_generation=10,
        total_generations=50,
        best_fitness=0.5,
        diversity_score=0.6,
        stagnation_count=5
    )
    print(f"With stagnation (count=5): mut={mut3:.3f}, cross={cross3:.3f}")
    
    # Test with low diversity
    mut4, cross4 = controller.get_rates(
        current_generation=10,
        total_generations=50,
        best_fitness=0.5,
        diversity_score=0.2,
        stagnation_count=0
    )
    print(f"With low diversity (0.2): mut={mut4:.3f}, cross={cross4:.3f}")
    
    print("✅ All adaptive rates tests passed!\n")


def test_fitness_improvements():
    """Test enhanced fitness calculation."""
    print("\n=== Testing Fitness Improvements ===")
    
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
    
    # Test with good metrics
    good_metrics = {
        'profit': 50.0,
        'sharpe_ratio': 2.0,
        'max_drawdown': 0.15,
        'win_rate': 0.60,
        'num_trades': 30,
        'profit_factor': 2.5,
        'sortino_ratio': 2.5,
    }
    
    fitness = evaluator.calculate_fitness(good_metrics)
    print(f"Fitness with good metrics: {fitness:.4f}")
    assert fitness > 0.5, "Good metrics should yield decent fitness"
    print("✓ Good metrics produce good fitness")
    
    # Test with poor metrics
    poor_metrics = {
        'profit': -10.0,
        'sharpe_ratio': -0.5,
        'max_drawdown': 0.30,
        'win_rate': 0.30,
        'num_trades': 5,
        'profit_factor': 0.8,
        'sortino_ratio': -0.5,
    }
    
    fitness_poor = evaluator.calculate_fitness(poor_metrics)
    print(f"Fitness with poor metrics: {fitness_poor:.4f}")
    assert fitness_poor < fitness, "Poor metrics should yield lower fitness"
    print("✓ Poor metrics produce lower fitness")
    
    # Test normalization functions
    assert evaluator._normalize_profit_factor(0.5) == 0.0
    assert evaluator._normalize_profit_factor(3.0) == 1.0
    print("✓ Profit factor normalization works")
    
    assert evaluator._normalize_sortino(-2) == 0.0
    assert evaluator._normalize_sortino(5) == 1.0
    print("✓ Sortino ratio normalization works")
    
    print("✅ All fitness tests passed!\n")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("Testing GA Improvements")
    print("="*60)
    
    try:
        test_monotonic_roi()
        test_adaptive_rates()
        test_fitness_improvements()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60 + "\n")
        return 0
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
