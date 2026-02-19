"""
Walk-Forward Timerange Utilities

Provides functions for splitting timeranges into training and validation windows
for walk-forward optimization.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TimeWindow:
    """
    Represents a time window for training or validation.
    """
    start_date: datetime
    end_date: datetime
    window_type: str  # 'train' or 'validate'
    window_index: int  # Index of this window in the sequence
    
    def to_freqtrade_format(self) -> str:
        """
        Convert to FreqTrade timerange format (YYYYMMDD-YYYYMMDD).
        
        Returns:
            Timerange string in FreqTrade format
        """
        start_str = self.start_date.strftime('%Y%m%d')
        end_str = self.end_date.strftime('%Y%m%d')
        return f"{start_str}-{end_str}"
    
    def get_duration_days(self) -> int:
        """
        Get window duration in days.
        
        Returns:
            Number of days in this window
        """
        return (self.end_date - self.start_date).days
    
    def __str__(self) -> str:
        """String representation."""
        return (f"{self.window_type.capitalize()} Window {self.window_index}: "
                f"{self.to_freqtrade_format()} ({self.get_duration_days()} days)")


@dataclass
class WalkForwardWindow:
    """
    Represents a walk-forward window with both training and validation periods.
    """
    train_window: TimeWindow
    validation_window: TimeWindow
    window_index: int  # Index of this walk-forward window
    
    def __str__(self) -> str:
        """String representation."""
        return (f"Walk-Forward Window {self.window_index}:\n"
                f"  Train: {self.train_window.to_freqtrade_format()}\n"
                f"  Validate: {self.validation_window.to_freqtrade_format()}")


def parse_timerange(timerange_str: str) -> Tuple[datetime, datetime]:
    """
    Parse FreqTrade timerange string to datetime objects.
    
    Args:
        timerange_str: Timerange string in format "YYYYMMDD-YYYYMMDD"
        
    Returns:
        Tuple of (start_date, end_date)
        
    Raises:
        ValueError: If timerange format is invalid
    """
    try:
        parts = timerange_str.split('-')
        if len(parts) != 2:
            raise ValueError(f"Invalid timerange format: {timerange_str}")
        
        start_date = datetime.strptime(parts[0], '%Y%m%d')
        end_date = datetime.strptime(parts[1], '%Y%m%d')
        
        if start_date >= end_date:
            raise ValueError(f"Start date must be before end date: {timerange_str}")
        
        return start_date, end_date
    except Exception as e:
        raise ValueError(f"Failed to parse timerange '{timerange_str}': {e}")


def create_walk_forward_windows(
    timerange: str,
    train_days: int,
    validation_days: int,
    step_days: int,
    mode: str = 'rolling',
    min_train_trades: Optional[int] = None
) -> List[WalkForwardWindow]:
    """
    Create walk-forward windows from a timerange.
    
    Args:
        timerange: Full timerange string (YYYYMMDD-YYYYMMDD)
        train_days: Number of days for training window
        validation_days: Number of days for validation window
        step_days: Number of days to slide forward for next window
        mode: 'rolling' (fixed window) or 'anchored' (expanding window)
        min_train_trades: Minimum trades required (not used here, for future use)
        
    Returns:
        List of WalkForwardWindow objects
        
    Raises:
        ValueError: If parameters are invalid
    """
    # Parse timerange
    start_date, end_date = parse_timerange(timerange)
    total_days = (end_date - start_date).days
    
    # Validate parameters
    if train_days <= 0:
        raise ValueError(f"train_days must be positive, got {train_days}")
    if validation_days <= 0:
        raise ValueError(f"validation_days must be positive, got {validation_days}")
    if step_days <= 0:
        raise ValueError(f"step_days must be positive, got {step_days}")
    if mode not in ['rolling', 'anchored']:
        raise ValueError(f"mode must be 'rolling' or 'anchored', got '{mode}'")
    
    # Check if we have enough data for at least one window
    min_required = train_days + validation_days
    if total_days < min_required:
        raise ValueError(
            f"Timerange too short for walk-forward: {total_days} days available, "
            f"but need at least {min_required} days (train={train_days} + validate={validation_days})"
        )
    
    windows: List[WalkForwardWindow] = []
    window_index = 0
    
    # Current position in the timeline
    current_pos = start_date
    
    logger.info(f"Creating walk-forward windows:")
    logger.info(f"  Mode: {mode}")
    logger.info(f"  Train: {train_days} days, Validate: {validation_days} days, Step: {step_days} days")
    logger.info(f"  Total timerange: {timerange} ({total_days} days)")
    
    while True:
        # Calculate training window
        if mode == 'rolling':
            # Rolling window: fixed size training period
            train_start = current_pos
            train_end = train_start + timedelta(days=train_days)
        else:  # anchored
            # Anchored window: expanding training period from start
            train_start = start_date
            train_end = current_pos + timedelta(days=train_days)
        
        # Calculate validation window (always follows training)
        val_start = train_end
        val_end = val_start + timedelta(days=validation_days)
        
        # Check if validation window fits within timerange
        if val_end > end_date:
            logger.info(f"Stopping: validation window would exceed timerange")
            break
        
        # Create windows
        train_window = TimeWindow(
            start_date=train_start,
            end_date=train_end,
            window_type='train',
            window_index=window_index
        )
        
        val_window = TimeWindow(
            start_date=val_start,
            end_date=val_end,
            window_type='validate',
            window_index=window_index
        )
        
        wf_window = WalkForwardWindow(
            train_window=train_window,
            validation_window=val_window,
            window_index=window_index
        )
        
        windows.append(wf_window)
        logger.debug(f"Created {wf_window}")
        
        # Move to next position
        current_pos = current_pos + timedelta(days=step_days)
        window_index += 1
        
        # Safety check: avoid infinite loops
        if window_index > 100:
            logger.warning(f"Stopping after 100 windows (safety limit)")
            break
    
    if not windows:
        raise ValueError(
            f"No walk-forward windows could be created with given parameters. "
            f"Try reducing train_days, validation_days, or increasing total timerange."
        )
    
    logger.info(f"Created {len(windows)} walk-forward windows")
    return windows


def validate_walk_forward_config(config: Dict[str, Any]) -> None:
    """
    Validate walk-forward configuration.
    
    Args:
        config: Configuration dictionary with walk_forward section
        
    Raises:
        ValueError: If configuration is invalid
    """
    if 'walk_forward' not in config:
        raise ValueError("walk_forward section missing from config")
    
    wf_config = config['walk_forward']
    
    # Check required fields
    required_fields = ['enabled', 'train_days', 'validation_days', 'step_days']
    for field in required_fields:
        if field not in wf_config:
            raise ValueError(f"walk_forward.{field} is required")
    
    # Validate field types and values
    if not isinstance(wf_config['enabled'], bool):
        raise ValueError("walk_forward.enabled must be boolean")
    
    if not isinstance(wf_config['train_days'], int) or wf_config['train_days'] <= 0:
        raise ValueError("walk_forward.train_days must be positive integer")
    
    if not isinstance(wf_config['validation_days'], int) or wf_config['validation_days'] <= 0:
        raise ValueError("walk_forward.validation_days must be positive integer")
    
    if not isinstance(wf_config['step_days'], int) or wf_config['step_days'] <= 0:
        raise ValueError("walk_forward.step_days must be positive integer")
    
    # Validate optional fields
    if 'mode' in wf_config and wf_config['mode'] not in ['rolling', 'anchored']:
        raise ValueError("walk_forward.mode must be 'rolling' or 'anchored'")
    
    if 'aggregation' in wf_config and wf_config['aggregation'] not in ['mean', 'min', 'harmonic_mean', 'weighted']:
        raise ValueError("walk_forward.aggregation must be one of: mean, min, harmonic_mean, weighted")
    
    logger.info("Walk-forward configuration validated successfully")


def aggregate_validation_scores(
    scores: List[float],
    method: str = 'mean',
    weights: Optional[List[float]] = None
) -> float:
    """
    Aggregate validation scores from multiple windows.
    
    Args:
        scores: List of validation fitness scores
        method: Aggregation method ('mean', 'min', 'harmonic_mean', 'weighted')
        weights: Optional weights for weighted aggregation (must sum to 1.0)
        
    Returns:
        Aggregated fitness score
        
    Raises:
        ValueError: If method is invalid or weights don't match scores
    """
    if not scores:
        return 0.0
    
    # Filter out None and negative scores (failed backtests)
    valid_scores = [s for s in scores if s is not None and s >= 0]
    
    if not valid_scores:
        logger.warning("No valid scores to aggregate")
        return 0.0
    
    if method == 'mean':
        # Simple arithmetic mean
        result = sum(valid_scores) / len(valid_scores)
        
    elif method == 'min':
        # Conservative: worst-case performance
        result = min(valid_scores)
        
    elif method == 'harmonic_mean':
        # Harmonic mean: penalizes inconsistency
        # More conservative than arithmetic mean
        if any(s == 0 for s in valid_scores):
            result = 0.0
        else:
            result = len(valid_scores) / sum(1.0 / s for s in valid_scores)
    
    elif method == 'weighted':
        # Weighted average (e.g., more weight to recent windows)
        if weights is None:
            # Default: linear weights favoring recent windows
            weights = [(i + 1) / sum(range(1, len(valid_scores) + 1)) 
                      for i in range(len(valid_scores))]
        
        if len(weights) != len(valid_scores):
            raise ValueError(
                f"Number of weights ({len(weights)}) must match number of scores ({len(valid_scores)})"
            )
        
        # Normalize weights to sum to 1.0
        weight_sum = sum(weights)
        if weight_sum == 0:
            raise ValueError("Weights sum to zero")
        normalized_weights = [w / weight_sum for w in weights]
        
        result = sum(s * w for s, w in zip(valid_scores, normalized_weights))
    
    else:
        raise ValueError(f"Unknown aggregation method: {method}")
    
    logger.debug(f"Aggregated {len(valid_scores)} scores using {method}: {result:.4f}")
    return result


def get_walk_forward_summary(windows: List[WalkForwardWindow]) -> Dict[str, Any]:
    """
    Get summary statistics for walk-forward windows.
    
    Args:
        windows: List of walk-forward windows
        
    Returns:
        Dictionary with summary statistics
    """
    if not windows:
        return {
            'num_windows': 0,
            'total_train_days': 0,
            'total_validate_days': 0,
            'avg_train_days': 0,
            'avg_validate_days': 0,
        }
    
    total_train_days = sum(w.train_window.get_duration_days() for w in windows)
    total_validate_days = sum(w.validation_window.get_duration_days() for w in windows)
    
    return {
        'num_windows': len(windows),
        'total_train_days': total_train_days,
        'total_validate_days': total_validate_days,
        'avg_train_days': total_train_days / len(windows),
        'avg_validate_days': total_validate_days / len(windows),
        'first_train_start': windows[0].train_window.start_date.strftime('%Y-%m-%d'),
        'last_validate_end': windows[-1].validation_window.end_date.strftime('%Y-%m-%d'),
    }
