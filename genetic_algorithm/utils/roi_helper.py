"""
ROI Helper Functions

Functions to generate and validate monotonically decreasing ROI tables
as required by FreqTrade.
"""

import random
from typing import Dict


def generate_monotonic_roi(roi_range: tuple = (0.01, 0.10), 
                          time_points: list = None) -> Dict[str, float]:
    """
    Generate a monotonically decreasing ROI table.
    
    FreqTrade requires ROI values to be monotonically decreasing over time.
    This function ensures ROI values always decrease or stay the same.
    
    Args:
        roi_range: Tuple of (min_roi, max_roi) as decimal values
        time_points: List of time points in minutes (default: [0, 30, 60, 120])
        
    Returns:
        Dictionary mapping time (minutes as string) to ROI value
    """
    if time_points is None:
        time_points = [0, 30, 60, 120]
    
    min_roi, max_roi = roi_range
    
    # Constants for ROI generation
    INITIAL_ROI_MULTIPLIER = 2.0  # Start with double the minimum ROI
    
    # Generate descending values
    descending_roi_values = []
    
    # Start with highest ROI at time 0
    current_roi = random.uniform(min_roi * INITIAL_ROI_MULTIPLIER, max_roi)
    descending_roi_values.append(current_roi)
    
    for i in range(1, len(time_points)):
        # Each subsequent value should be lower
        # Ensure it stays within range and is lower than previous
        max_next = min(current_roi * 0.9, max_roi * 0.6)  # At most 90% of previous
        min_next = max(min_roi, current_roi * 0.3)  # At least 30% of previous or min_roi
        
        # Ensure min_next <= max_next
        if min_next > max_next:
            min_next = max_next * 0.9
        
        current_roi = random.uniform(min_next, max_next)
        descending_roi_values.append(current_roi)
    
    # Create dictionary with time points as string keys (required by FreqTrade schema validation)
    roi_dict = {str(time_points[i]): descending_roi_values[i] for i in range(len(time_points))}
    
    # Validate monotonic property
    assert is_monotonic_roi(roi_dict), "Generated ROI is not monotonic"
    
    return roi_dict


def is_monotonic_roi(roi: Dict[str, float]) -> bool:
    """
    Check if ROI table is monotonically decreasing.
    
    Args:
        roi: Dictionary mapping time (as string) to ROI value
        
    Returns:
        True if ROI values decrease (or stay same) over time, False otherwise
    """
    # Sort by time (convert to int for proper numerical sorting)
    sorted_times = sorted(roi.keys(), key=lambda x: int(x))
    
    # Check that values are monotonically decreasing
    for i in range(1, len(sorted_times)):
        if roi[sorted_times[i]] > roi[sorted_times[i-1]]:
            return False
    
    return True


def fix_monotonic_roi(roi: Dict[str, float]) -> Dict[str, float]:
    """
    Fix a non-monotonic ROI table by adjusting values to be monotonic.
    
    Args:
        roi: Dictionary mapping time (as string) to ROI value
        
    Returns:
        Fixed ROI dictionary with monotonically decreasing values
    """
    if is_monotonic_roi(roi):
        return roi.copy()
    
    # Sort by time (convert to int for proper numerical sorting)
    sorted_times = sorted(roi.keys(), key=lambda x: int(x))
    fixed_roi = {}
    
    # First value stays the same
    fixed_roi[sorted_times[0]] = roi[sorted_times[0]]
    
    # Ensure subsequent values are not greater than previous
    for i in range(1, len(sorted_times)):
        current_value = roi[sorted_times[i]]
        previous_value = fixed_roi[sorted_times[i-1]]
        
        # If current is greater, set it to previous (or slightly less)
        if current_value > previous_value:
            fixed_roi[sorted_times[i]] = previous_value * 0.95
        else:
            fixed_roi[sorted_times[i]] = current_value
    
    return fixed_roi


def mutate_roi(roi: Dict[str, float], roi_range: tuple = (0.01, 0.10), 
               mutation_strength: float = 0.2) -> Dict[str, float]:
    """
    Mutate an ROI table while maintaining monotonic property.
    
    Args:
        roi: Current ROI dictionary (with string keys)
        roi_range: Tuple of (min_roi, max_roi) as decimal values
        mutation_strength: How much to vary values (0.0-1.0)
        
    Returns:
        Mutated ROI dictionary that is still monotonically decreasing
    """
    # Constants for mutation bounds
    MIN_ROI_FRACTION = 0.5  # Ensure values stay at least 50% of min ROI
    MAX_PREVIOUS_FRACTION = 0.99  # Ensure values stay below 99% of previous
    INITIAL_ROI_MULTIPLIER = 2.0  # For first value
    
    min_roi, max_roi = roi_range
    # Sort by time (convert to int for proper numerical sorting)
    sorted_times = sorted(roi.keys(), key=lambda x: int(x))
    mutated_roi = {}
    
    # Mutate the first (highest) value
    current_value = roi[sorted_times[0]]
    variation = current_value * mutation_strength
    new_value = current_value + random.uniform(-variation, variation)
    new_value = max(min_roi * INITIAL_ROI_MULTIPLIER, min(max_roi, new_value))
    mutated_roi[sorted_times[0]] = new_value
    
    # Mutate subsequent values, ensuring they stay below previous
    for i in range(1, len(sorted_times)):
        current_value = roi[sorted_times[i]]
        previous_value = mutated_roi[sorted_times[i-1]]
        
        # Mutate within constrained range
        variation = current_value * mutation_strength
        new_value = current_value + random.uniform(-variation, variation)
        
        # Ensure it's less than or equal to previous and within bounds
        new_value = max(min_roi * MIN_ROI_FRACTION, min(previous_value * MAX_PREVIOUS_FRACTION, new_value))
        mutated_roi[sorted_times[i]] = new_value
    
    # Verify monotonic property
    assert is_monotonic_roi(mutated_roi), "Mutated ROI is not monotonic"
    
    return mutated_roi
