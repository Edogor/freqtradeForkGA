"""
Tests for Walk-Forward Optimization

Tests the timerange splitting, window creation, and walk-forward evaluation logic.
"""

import pytest
from datetime import datetime, timedelta
from genetic_algorithm.utils.timerange import (
    parse_timerange,
    create_walk_forward_windows,
    validate_walk_forward_config,
    aggregate_validation_scores,
    get_walk_forward_summary,
    TimeWindow,
    WalkForwardWindow,
)


class TestParseTimerange:
    """Test timerange parsing."""
    
    def test_valid_timerange(self):
        """Test parsing valid timerange."""
        start, end = parse_timerange("20240101-20240201")
        assert start == datetime(2024, 1, 1)
        assert end == datetime(2024, 2, 1)
    
    def test_invalid_format(self):
        """Test invalid timerange format."""
        with pytest.raises(ValueError, match="Invalid timerange format"):
            parse_timerange("20240101")
        
        with pytest.raises(ValueError, match="Invalid timerange format"):
            parse_timerange("20240101-20240201-20240301")
    
    def test_invalid_dates(self):
        """Test invalid date values."""
        with pytest.raises(ValueError):
            parse_timerange("invalid-20240201")
    
    def test_start_after_end(self):
        """Test start date after end date."""
        with pytest.raises(ValueError, match="Start date must be before end date"):
            parse_timerange("20240201-20240101")


class TestTimeWindow:
    """Test TimeWindow dataclass."""
    
    def test_to_freqtrade_format(self):
        """Test conversion to FreqTrade format."""
        window = TimeWindow(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 2, 1),
            window_type='train',
            window_index=0
        )
        assert window.to_freqtrade_format() == "20240101-20240201"
    
    def test_get_duration_days(self):
        """Test duration calculation."""
        window = TimeWindow(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            window_type='train',
            window_index=0
        )
        assert window.get_duration_days() == 30


class TestCreateWalkForwardWindows:
    """Test walk-forward window creation."""
    
    def test_rolling_window_basic(self):
        """Test basic rolling window creation."""
        windows = create_walk_forward_windows(
            timerange="20240101-20240301",  # 60 days
            train_days=30,
            validation_days=10,
            step_days=10,
            mode='rolling'
        )
        
        # Should create 3 windows:
        # W1: Train 1-30, Val 31-40
        # W2: Train 11-40, Val 41-50
        # W3: Train 21-50, Val 51-60
        assert len(windows) == 3
        
        # Check first window
        assert windows[0].train_window.get_duration_days() == 30
        assert windows[0].validation_window.get_duration_days() == 10
        assert windows[0].train_window.start_date == datetime(2024, 1, 1)
        assert windows[0].validation_window.end_date == datetime(2024, 2, 10)
    
    def test_anchored_window_basic(self):
        """Test basic anchored (expanding) window creation."""
        windows = create_walk_forward_windows(
            timerange="20240101-20240301",  # 60 days
            train_days=30,
            validation_days=10,
            step_days=10,
            mode='anchored'
        )
        
        # Anchored mode: training always starts from beginning
        assert len(windows) == 3
        
        # First window
        assert windows[0].train_window.start_date == datetime(2024, 1, 1)
        
        # All windows should start from the same date in anchored mode
        assert windows[1].train_window.start_date == datetime(2024, 1, 1)
        assert windows[2].train_window.start_date == datetime(2024, 1, 1)
        
        # Training windows should be expanding
        assert windows[1].train_window.get_duration_days() > windows[0].train_window.get_duration_days()
    
    def test_insufficient_data(self):
        """Test error when timerange is too short."""
        with pytest.raises(ValueError, match="Timerange too short"):
            create_walk_forward_windows(
                timerange="20240101-20240115",  # Only 15 days
                train_days=30,
                validation_days=10,
                step_days=10,
                mode='rolling'
            )
    
    def test_invalid_parameters(self):
        """Test validation of parameters."""
        with pytest.raises(ValueError, match="train_days must be positive"):
            create_walk_forward_windows(
                timerange="20240101-20240301",
                train_days=0,
                validation_days=10,
                step_days=10,
                mode='rolling'
            )
        
        with pytest.raises(ValueError, match="mode must be"):
            create_walk_forward_windows(
                timerange="20240101-20240301",
                train_days=30,
                validation_days=10,
                step_days=10,
                mode='invalid'
            )
    
    def test_window_indices(self):
        """Test that window indices are correctly assigned."""
        windows = create_walk_forward_windows(
            timerange="20240101-20240301",
            train_days=30,
            validation_days=10,
            step_days=10,
            mode='rolling'
        )
        
        for i, window in enumerate(windows):
            assert window.window_index == i
            assert window.train_window.window_index == i
            assert window.validation_window.window_index == i
    
    def test_no_overlapping_validation(self):
        """Test that validation windows don't overlap with their training windows."""
        windows = create_walk_forward_windows(
            timerange="20240101-20240301",
            train_days=30,
            validation_days=10,
            step_days=10,
            mode='rolling'
        )
        
        for window in windows:
            # Validation should start right after training ends
            assert window.validation_window.start_date == window.train_window.end_date
            # Validation should not overlap with training
            assert window.validation_window.start_date >= window.train_window.end_date


class TestAggregateValidationScores:
    """Test validation score aggregation methods."""
    
    def test_mean_aggregation(self):
        """Test mean aggregation."""
        scores = [0.5, 0.6, 0.7]
        result = aggregate_validation_scores(scores, method='mean')
        assert result == pytest.approx(0.6, rel=1e-5)
    
    def test_min_aggregation(self):
        """Test min (conservative) aggregation."""
        scores = [0.5, 0.6, 0.7]
        result = aggregate_validation_scores(scores, method='min')
        assert result == 0.5
    
    def test_harmonic_mean_aggregation(self):
        """Test harmonic mean aggregation."""
        scores = [0.5, 0.6, 0.7]
        result = aggregate_validation_scores(scores, method='harmonic_mean')
        # Harmonic mean of 0.5, 0.6, 0.7 ≈ 0.583
        assert result == pytest.approx(0.583, rel=1e-2)
    
    def test_harmonic_mean_with_zero(self):
        """Test harmonic mean with zero score."""
        scores = [0.0, 0.6, 0.7]
        result = aggregate_validation_scores(scores, method='harmonic_mean')
        assert result == 0.0
    
    def test_weighted_aggregation(self):
        """Test weighted aggregation with custom weights."""
        scores = [0.5, 0.6, 0.7]
        weights = [0.2, 0.3, 0.5]  # More weight to recent
        result = aggregate_validation_scores(scores, method='weighted', weights=weights)
        expected = 0.5 * 0.2 + 0.6 * 0.3 + 0.7 * 0.5
        assert result == pytest.approx(expected, rel=1e-5)
    
    def test_weighted_aggregation_default_weights(self):
        """Test weighted aggregation with default (linear) weights."""
        scores = [0.5, 0.6, 0.7]
        result = aggregate_validation_scores(scores, method='weighted')
        # Default weights favor recent: [1/6, 2/6, 3/6]
        expected = 0.5 * (1/6) + 0.6 * (2/6) + 0.7 * (3/6)
        assert result == pytest.approx(expected, rel=1e-5)
    
    def test_empty_scores(self):
        """Test aggregation with empty scores."""
        result = aggregate_validation_scores([], method='mean')
        assert result == 0.0
    
    def test_filter_invalid_scores(self):
        """Test filtering of None and negative scores."""
        scores = [0.5, None, -0.1, 0.7]
        result = aggregate_validation_scores(scores, method='mean')
        # Should only use 0.5 and 0.7
        assert result == pytest.approx(0.6, rel=1e-5)
    
    def test_invalid_method(self):
        """Test invalid aggregation method."""
        with pytest.raises(ValueError, match="Unknown aggregation method"):
            aggregate_validation_scores([0.5, 0.6], method='invalid')


class TestValidateWalkForwardConfig:
    """Test walk-forward configuration validation."""
    
    def test_valid_config(self):
        """Test valid configuration."""
        config = {
            'walk_forward': {
                'enabled': True,
                'train_days': 60,
                'validation_days': 15,
                'step_days': 15,
                'mode': 'rolling',
                'aggregation': 'mean'
            }
        }
        # Should not raise
        validate_walk_forward_config(config)
    
    def test_missing_section(self):
        """Test missing walk_forward section."""
        config = {}
        with pytest.raises(ValueError, match="walk_forward section missing"):
            validate_walk_forward_config(config)
    
    def test_missing_required_fields(self):
        """Test missing required fields."""
        config = {
            'walk_forward': {
                'enabled': True,
                'train_days': 60
                # Missing validation_days and step_days
            }
        }
        with pytest.raises(ValueError, match="validation_days is required"):
            validate_walk_forward_config(config)
    
    def test_invalid_field_types(self):
        """Test invalid field types."""
        config = {
            'walk_forward': {
                'enabled': 'true',  # Should be boolean
                'train_days': 60,
                'validation_days': 15,
                'step_days': 15
            }
        }
        with pytest.raises(ValueError, match="enabled must be boolean"):
            validate_walk_forward_config(config)
    
    def test_invalid_mode(self):
        """Test invalid mode value."""
        config = {
            'walk_forward': {
                'enabled': True,
                'train_days': 60,
                'validation_days': 15,
                'step_days': 15,
                'mode': 'invalid'
            }
        }
        with pytest.raises(ValueError, match="mode must be"):
            validate_walk_forward_config(config)
    
    def test_invalid_aggregation(self):
        """Test invalid aggregation value."""
        config = {
            'walk_forward': {
                'enabled': True,
                'train_days': 60,
                'validation_days': 15,
                'step_days': 15,
                'aggregation': 'invalid'
            }
        }
        with pytest.raises(ValueError, match="aggregation must be one of"):
            validate_walk_forward_config(config)


class TestGetWalkForwardSummary:
    """Test walk-forward summary statistics."""
    
    def test_summary_with_windows(self):
        """Test summary generation with valid windows."""
        windows = create_walk_forward_windows(
            timerange="20240101-20240301",
            train_days=30,
            validation_days=10,
            step_days=10,
            mode='rolling'
        )
        
        summary = get_walk_forward_summary(windows)
        
        assert summary['num_windows'] == 3
        assert summary['avg_train_days'] == 30
        assert summary['avg_validate_days'] == 10
        assert 'first_train_start' in summary
        assert 'last_validate_end' in summary
    
    def test_summary_empty_windows(self):
        """Test summary with empty windows list."""
        summary = get_walk_forward_summary([])
        
        assert summary['num_windows'] == 0
        assert summary['total_train_days'] == 0
        assert summary['total_validate_days'] == 0


class TestWalkForwardIntegration:
    """Integration tests for walk-forward with FitnessEvaluator."""
    
    def test_walk_forward_config_validation(self):
        """Test that walk-forward configuration validates correctly."""
        valid_config = {
            'walk_forward': {
                'enabled': True,
                'train_days': 30,
                'validation_days': 10,
                'step_days': 10,
                'mode': 'rolling',
                'aggregation': 'mean'
            }
        }
        # Should not raise
        validate_walk_forward_config(valid_config)
    
    def test_disabled_walk_forward_by_default(self):
        """Test that walk-forward is disabled by default in config."""
        import yaml
        from pathlib import Path
        
        config_path = Path(__file__).parent / 'config' / 'ga_config.yaml'
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Walk-forward should exist but be disabled by default
        assert 'walk_forward' in config
        assert config['walk_forward']['enabled'] == False


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
