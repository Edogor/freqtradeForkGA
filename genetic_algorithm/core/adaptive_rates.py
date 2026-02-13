"""
Adaptive Mutation and Crossover Rates

Dynamically adjusts mutation and crossover rates based on:
- Generation progress
- Fitness stagnation
- Population diversity
"""

from typing import Tuple


class AdaptiveRateController:
    """
    Controls adaptive mutation and crossover rates.
    
    Strategy:
    - Early generations: Higher mutation for exploration
    - Later generations: Lower mutation for exploitation
    - Stagnation: Increase mutation to escape local optima
    - Low diversity: Increase mutation to promote variation
    """
    
    def __init__(self, 
                 base_mutation_rate: float = 0.15,
                 base_crossover_rate: float = 0.7,
                 min_mutation_rate: float = 0.05,
                 max_mutation_rate: float = 0.30,
                 min_crossover_rate: float = 0.5,
                 max_crossover_rate: float = 0.9):
        """
        Initialize adaptive rate controller.
        
        Args:
            base_mutation_rate: Starting mutation rate
            base_crossover_rate: Starting crossover rate
            min_mutation_rate: Minimum allowed mutation rate
            max_mutation_rate: Maximum allowed mutation rate
            min_crossover_rate: Minimum allowed crossover rate
            max_crossover_rate: Maximum allowed crossover rate
        """
        self.base_mutation_rate = base_mutation_rate
        self.base_crossover_rate = base_crossover_rate
        self.min_mutation_rate = min_mutation_rate
        self.max_mutation_rate = max_mutation_rate
        self.min_crossover_rate = min_crossover_rate
        self.max_crossover_rate = max_crossover_rate
        
        # Track history
        self.best_fitness_history = []
        self.diversity_history = []
        
    def get_rates(self, 
                  current_generation: int,
                  total_generations: int,
                  best_fitness: float,
                  diversity_score: float,
                  stagnation_count: int) -> Tuple[float, float]:
        """
        Calculate adaptive mutation and crossover rates.
        
        Args:
            current_generation: Current generation number (0-indexed)
            total_generations: Total number of generations
            best_fitness: Best fitness in current generation
            diversity_score: Population diversity score (0-1)
            stagnation_count: Number of generations without improvement
            
        Returns:
            Tuple of (mutation_rate, crossover_rate)
        """
        # Track history
        self.best_fitness_history.append(best_fitness)
        self.diversity_history.append(diversity_score)
        
        # Calculate progress (0 to 1)
        progress = current_generation / max(total_generations - 1, 1)
        
        # Base rates change with progress
        # Early: higher mutation (exploration)
        # Late: lower mutation (exploitation)
        progress_factor = 1.0 - (progress * 0.5)  # Reduce by up to 50%
        
        mutation_rate = self.base_mutation_rate * progress_factor
        crossover_rate = self.base_crossover_rate
        
        # Adjust for stagnation
        # If fitness hasn't improved for several generations, increase mutation
        if stagnation_count > 3:
            stagnation_boost = min(0.15, stagnation_count * 0.03)
            mutation_rate += stagnation_boost
        
        # Adjust for low diversity
        # If diversity is low, increase mutation to add variation
        if diversity_score < 0.3:
            diversity_boost = (0.3 - diversity_score) * 0.5
            mutation_rate += diversity_boost
        
        # Adjust crossover inversely to mutation
        # When mutation is high, slightly reduce crossover
        if mutation_rate > self.base_mutation_rate * 1.2:
            crossover_rate *= 0.95
        
        # Clamp to valid ranges
        mutation_rate = max(self.min_mutation_rate, 
                          min(self.max_mutation_rate, mutation_rate))
        crossover_rate = max(self.min_crossover_rate,
                           min(self.max_crossover_rate, crossover_rate))
        
        return mutation_rate, crossover_rate
    
    def reset(self):
        """Reset history for new run."""
        self.best_fitness_history.clear()
        self.diversity_history.clear()
