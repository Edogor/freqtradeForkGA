"""
Tests for GA improvements

Tests:
- Monotonic ROI generation and mutation
- Adaptive rates controller
- Enhanced fitness calculation
"""

import pytest
from genetic_algorithm.utils.roi_helper import (
    generate_monotonic_roi, is_monotonic_roi, fix_monotonic_roi, mutate_roi
)
from genetic_algorithm.core.adaptive_rates import AdaptiveRateController


class TestMonotonicROI:
    """Test ROI helper functions."""
    
    def test_generate_monotonic_roi(self):
        """Test that generated ROI is monotonically decreasing."""
        roi = generate_monotonic_roi()
        assert is_monotonic_roi(roi), "Generated ROI should be monotonic"
        
        # Check structure
        assert 0 in roi, "ROI should have time 0"
        assert len(roi) >= 2, "ROI should have multiple time points"
        
        # Check values are reasonable
        for time, value in roi.items():
            assert 0 < value < 1, f"ROI value {value} at time {time} should be between 0 and 1"
    
    def test_is_monotonic_roi(self):
        """Test monotonic ROI detection."""
        # Valid monotonic ROI
        good_roi = {0: 0.10, 30: 0.06, 60: 0.03, 120: 0.01}
        assert is_monotonic_roi(good_roi)
        
        # Invalid ROI (increases)
        bad_roi = {0: 0.05, 30: 0.10, 60: 0.03}
        assert not is_monotonic_roi(bad_roi)
        
        # Equal values are OK
        equal_roi = {0: 0.05, 30: 0.05, 60: 0.03}
        assert is_monotonic_roi(equal_roi)
    
    def test_fix_monotonic_roi(self):
        """Test fixing non-monotonic ROI."""
        bad_roi = {0: 0.05, 30: 0.10, 60: 0.03, 120: 0.08}
        fixed_roi = fix_monotonic_roi(bad_roi)
        
        assert is_monotonic_roi(fixed_roi), "Fixed ROI should be monotonic"
        assert fixed_roi[0] == bad_roi[0], "First value should stay the same"
    
    def test_mutate_roi(self):
        """Test ROI mutation maintains monotonic property."""
        original_roi = {0: 0.08, 30: 0.05, 60: 0.02, 120: 0.01}
        
        # Mutate multiple times
        for _ in range(10):
            mutated = mutate_roi(original_roi)
            assert is_monotonic_roi(mutated), "Mutated ROI should be monotonic"
            
            # Values should be in reasonable range
            for time, value in mutated.items():
                assert 0 < value < 0.2, f"ROI value {value} should be reasonable"
    
    def test_generate_with_custom_params(self):
        """Test ROI generation with custom parameters."""
        roi_range = (0.02, 0.15)
        time_points = [0, 15, 45, 90, 180]
        
        roi = generate_monotonic_roi(roi_range, time_points)
        
        assert len(roi) == len(time_points)
        assert all(time in roi for time in time_points)
        assert is_monotonic_roi(roi)


class TestAdaptiveRates:
    """Test adaptive mutation and crossover rates."""
    
    def test_initialization(self):
        """Test controller initialization."""
        controller = AdaptiveRateController()
        
        assert controller.base_mutation_rate == 0.15
        assert controller.base_crossover_rate == 0.7
        assert len(controller.best_fitness_history) == 0
    
    def test_early_generation_rates(self):
        """Test that early generations have higher mutation."""
        controller = AdaptiveRateController(base_mutation_rate=0.15)
        
        # Early generation (1 out of 50)
        mut_rate, cross_rate = controller.get_rates(
            current_generation=1,
            total_generations=50,
            best_fitness=0.5,
            diversity_score=0.8,
            stagnation_count=0
        )
        
        # Should be close to base rate or higher
        assert mut_rate >= 0.10, "Early generation should have decent mutation rate"
    
    def test_late_generation_rates(self):
        """Test that late generations have lower mutation."""
        controller = AdaptiveRateController(base_mutation_rate=0.15)
        
        # Late generation (45 out of 50)
        mut_rate, cross_rate = controller.get_rates(
            current_generation=45,
            total_generations=50,
            best_fitness=0.8,
            diversity_score=0.6,
            stagnation_count=0
        )
        
        # Should be lower than base rate
        assert mut_rate <= 0.15, "Late generation should have lower mutation"
    
    def test_stagnation_increases_mutation(self):
        """Test that stagnation increases mutation rate."""
        controller = AdaptiveRateController(base_mutation_rate=0.15)
        
        # No stagnation
        mut_rate_no_stag, _ = controller.get_rates(
            current_generation=10,
            total_generations=50,
            best_fitness=0.5,
            diversity_score=0.6,
            stagnation_count=0
        )
        
        # With stagnation
        mut_rate_with_stag, _ = controller.get_rates(
            current_generation=10,
            total_generations=50,
            best_fitness=0.5,
            diversity_score=0.6,
            stagnation_count=5
        )
        
        assert mut_rate_with_stag > mut_rate_no_stag, \
            "Stagnation should increase mutation rate"
    
    def test_low_diversity_increases_mutation(self):
        """Test that low diversity increases mutation rate."""
        controller = AdaptiveRateController(base_mutation_rate=0.15)
        
        # High diversity
        mut_rate_high_div, _ = controller.get_rates(
            current_generation=10,
            total_generations=50,
            best_fitness=0.5,
            diversity_score=0.8,
            stagnation_count=0
        )
        
        # Low diversity
        mut_rate_low_div, _ = controller.get_rates(
            current_generation=10,
            total_generations=50,
            best_fitness=0.5,
            diversity_score=0.2,
            stagnation_count=0
        )
        
        assert mut_rate_low_div > mut_rate_high_div, \
            "Low diversity should increase mutation rate"
    
    def test_rate_bounds(self):
        """Test that rates stay within bounds."""
        controller = AdaptiveRateController(
            min_mutation_rate=0.05,
            max_mutation_rate=0.30,
            min_crossover_rate=0.5,
            max_crossover_rate=0.9
        )
        
        # Extreme conditions
        mut_rate, cross_rate = controller.get_rates(
            current_generation=1,
            total_generations=50,
            best_fitness=0.1,
            diversity_score=0.1,
            stagnation_count=10
        )
        
        assert 0.05 <= mut_rate <= 0.30, "Mutation rate should be within bounds"
        assert 0.5 <= cross_rate <= 0.9, "Crossover rate should be within bounds"


class TestFitnessImprovements:
    """Test enhanced fitness calculation."""
    
    def test_fitness_includes_all_metrics(self):
        """Test that fitness uses all available metrics."""
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
        
        metrics = {
            'profit': 50.0,
            'sharpe_ratio': 2.0,
            'max_drawdown': 0.15,
            'win_rate': 0.60,
            'num_trades': 30,
            'profit_factor': 2.5,
            'sortino_ratio': 2.5,
        }
        
        fitness = evaluator.calculate_fitness(metrics)
        
        assert fitness > 0, "Fitness should be positive for good metrics"
        assert fitness <= 1.0, "Normalized fitness should not exceed 1.0"
    
    def test_profit_factor_normalization(self):
        """Test profit factor normalization."""
        from genetic_algorithm.evaluation.fitness import FitnessEvaluator
        
        config = {'fitness_weights': {}, 'fitness_penalties': {}, 'backtesting': {}}
        evaluator = FitnessEvaluator(config)
        
        # Test various profit factors
        assert evaluator._normalize_profit_factor(0.5) == 0.0
        assert evaluator._normalize_profit_factor(3.0) == 1.0
        assert 0 < evaluator._normalize_profit_factor(1.5) < 1.0
        assert 0 < evaluator._normalize_profit_factor(2.0) < 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
