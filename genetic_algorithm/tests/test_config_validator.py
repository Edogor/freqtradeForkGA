"""
Tests for GA Config Validator

Covers: basic validation, anti-pattern detection, preflight checks,
and edge cases for all validation rules.
"""

import pytest

from genetic_algorithm.utils.config_validator import (
    validate_ga_config,
    _check_experiment_anti_patterns,
    validate_and_log,
)


# =============================================================================
# HELPERS
# =============================================================================


def _base_config(**overrides):
    """Create a minimal valid config, with optional overrides merged in."""
    config = {
        'genetic_algorithm': {
            'population_size': 15,
            'generations': 20,
            'mutation_rate': 0.15,
            'crossover_rate': 0.8,
            'elite_size': 3,
            'tournament_size': 3,
            'mode': 'single_objective',
        },
        'backtesting': {
            'pairs': ['BTC/USDT', 'ETH/USDT'],
            'timerange': '20230101-20260101',
            'datadir': 'user_data/data/binance',
            'fee': 0.001,
        },
    }
    for key, val in overrides.items():
        if isinstance(val, dict) and key in config:
            config[key].update(val)
        else:
            config[key] = val
    return config


# =============================================================================
# BASIC VALIDATION TESTS
# =============================================================================


class TestBasicValidation:

    def test_valid_config_no_errors(self):
        errors, warnings = validate_ga_config(_base_config())
        assert errors == []

    def test_missing_ga_section(self):
        errors, _ = validate_ga_config({'backtesting': {'pairs': ['BTC/USDT']}})
        assert any("genetic_algorithm" in e for e in errors)

    def test_missing_backtesting_section(self):
        errors, _ = validate_ga_config({'genetic_algorithm': _base_config()['genetic_algorithm']})
        assert any("backtesting" in e for e in errors)

    def test_population_size_too_small(self):
        config = _base_config()
        config['genetic_algorithm']['population_size'] = 1
        errors, _ = validate_ga_config(config)
        assert any("population_size" in e for e in errors)

    def test_population_size_too_large(self):
        config = _base_config()
        config['genetic_algorithm']['population_size'] = 100001
        errors, _ = validate_ga_config(config)
        assert any("population_size" in e for e in errors)

    def test_mutation_rate_out_of_range(self):
        config = _base_config()
        config['genetic_algorithm']['mutation_rate'] = 1.5
        errors, _ = validate_ga_config(config)
        assert any("mutation_rate" in e for e in errors)

    def test_elite_size_exceeds_population(self):
        config = _base_config()
        config['genetic_algorithm']['elite_size'] = 20
        config['genetic_algorithm']['population_size'] = 15
        errors, _ = validate_ga_config(config)
        assert any("elite_size" in e for e in errors)

    def test_tournament_size_exceeds_population(self):
        config = _base_config()
        config['genetic_algorithm']['tournament_size'] = 20
        errors, _ = validate_ga_config(config)
        assert any("tournament_size" in e for e in errors)

    def test_invalid_mode(self):
        config = _base_config()
        config['genetic_algorithm']['mode'] = 'invalid_mode'
        errors, _ = validate_ga_config(config)
        assert any("mode" in e for e in errors)

    def test_empty_pairs(self):
        config = _base_config()
        config['backtesting']['pairs'] = []
        errors, _ = validate_ga_config(config)
        assert any("pairs" in e for e in errors)

    def test_bad_timerange_format(self):
        config = _base_config()
        config['backtesting']['timerange'] = 'not-a-date'
        errors, _ = validate_ga_config(config)
        # Should not crash, may produce warning

    def test_high_fee_warning(self):
        config = _base_config()
        config['backtesting']['fee'] = 0.10
        _, warnings = validate_ga_config(config)
        assert any("fee" in w for w in warnings)


# =============================================================================
# FITNESS WEIGHTS TESTS
# =============================================================================


class TestFitnessWeightsValidation:

    def test_negative_weight_is_error(self):
        config = _base_config()
        config['fitness_weights'] = {'profit': -0.5, 'sharpe_ratio': 0.5}
        errors, _ = validate_ga_config(config)
        assert any("negative" in e for e in errors)

    def test_sum_far_from_one_warns(self):
        config = _base_config()
        config['fitness_weights'] = {'profit': 0.1, 'sharpe_ratio': 0.1}
        _, warnings = validate_ga_config(config)
        assert any("sum" in w.lower() for w in warnings)


# =============================================================================
# WALK-FORWARD VALIDATION
# =============================================================================


class TestWalkForwardValidation:

    def test_short_train_days(self):
        config = _base_config()
        config['walk_forward'] = {'enabled': True, 'train_days': 3, 'validation_days': 7}
        errors, _ = validate_ga_config(config)
        assert any("train_days" in e for e in errors)

    def test_short_validation_days(self):
        config = _base_config()
        config['walk_forward'] = {'enabled': True, 'train_days': 30, 'validation_days': 0}
        errors, _ = validate_ga_config(config)
        assert any("validation_days" in e for e in errors)


# =============================================================================
# STRATEGY CONSTRAINTS VALIDATION
# =============================================================================


class TestStrategyConstraintValidation:

    def test_inverted_stoploss_range(self):
        config = _base_config()
        config['strategy_constraints'] = {'stoploss_range': [-0.05, -0.20]}
        errors, _ = validate_ga_config(config)
        assert any("stoploss_range" in e for e in errors)

    def test_positive_stoploss_upper_warns(self):
        config = _base_config()
        config['strategy_constraints'] = {'stoploss_range': [-0.20, 0.05]}
        _, warnings = validate_ga_config(config)
        assert any("stoploss" in w.lower() for w in warnings)


# =============================================================================
# INDICATOR VALIDATION
# =============================================================================


class TestIndicatorValidation:

    def test_min_exceeds_max(self):
        config = _base_config()
        config['indicators'] = {'min_per_strategy': 8, 'max_per_strategy': 4}
        errors, _ = validate_ga_config(config)
        assert any("min_per_strategy" in e for e in errors)


# =============================================================================
# ANTI-PATTERN DETECTION TESTS
# =============================================================================


class TestAntiPatternWarnings:

    def test_ap1_pop_over_15_standard_ga(self):
        config = _base_config()
        config['genetic_algorithm']['population_size'] = 20
        _, warnings = validate_ga_config(config)
        assert any("ANTI-PATTERN" in w and "population_size" in w for w in warnings)

    def test_ap2_island_pop_over_6(self):
        config = _base_config()
        config['genetic_algorithm']['island_model'] = {
            'enabled': True,
            'population_per_island': 8,
        }
        _, warnings = validate_ga_config(config)
        assert any("ANTI-PATTERN" in w and "island" in w.lower() for w in warnings)

    def test_ap3_llm_plus_rank_selection(self):
        config = _base_config()
        config['genetic_algorithm']['selection_method'] = 'rank'
        config['advanced'] = {'llm': {'enabled': True}}
        _, warnings = validate_ga_config(config)
        assert any("ANTI-PATTERN" in w and "rank" in w for w in warnings)

    def test_ap4_component_crossover_with_island(self):
        config = _base_config()
        config['genetic_algorithm']['crossover_method'] = 'component'
        config['genetic_algorithm']['island_model'] = {'enabled': True}
        _, warnings = validate_ga_config(config)
        assert any("ANTI-PATTERN" in w and "component" in w for w in warnings)

    def test_ap5_low_patience_low_elite(self):
        config = _base_config()
        config['genetic_algorithm']['early_stopping'] = {'patience': 4}
        config['genetic_algorithm']['elite_size'] = 2
        _, warnings = validate_ga_config(config)
        assert any("ANTI-PATTERN" in w and "patience" in w for w in warnings)

    def test_ap6_mc_permutations_over_15(self):
        config = _base_config()
        config['monte_carlo'] = {'enabled': True, 'num_permutations': 30}
        _, warnings = validate_ga_config(config)
        assert any("ANTI-PATTERN" in w and "permutation" in w.lower() for w in warnings)

    def test_ap7_three_plus_pairs(self):
        config = _base_config()
        config['backtesting']['pairs'] = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
        _, warnings = validate_ga_config(config)
        assert any("ANTI-PATTERN" in w and "pair" in w.lower() for w in warnings)

    def test_ap8_nsga2_with_fitness_sharing(self):
        config = _base_config()
        config['genetic_algorithm']['mode'] = 'nsga2'
        config['genetic_algorithm']['fitness_sharing'] = True
        _, warnings = validate_ga_config(config)
        assert any("ANTI-PATTERN" in w and "fitness_sharing" in w for w in warnings)

    def test_ap9_island_plus_walk_forward(self):
        config = _base_config()
        config['genetic_algorithm']['island_model'] = {'enabled': True}
        config['walk_forward'] = {'enabled': True}
        _, warnings = validate_ga_config(config)
        assert any("ANTI-PATTERN" in w and "walk_forward" in w.lower() for w in warnings)

    def test_ap10_island_plus_monte_carlo(self):
        config = _base_config()
        config['genetic_algorithm']['island_model'] = {'enabled': True}
        config['monte_carlo'] = {'enabled': True}
        _, warnings = validate_ga_config(config)
        assert any("ANTI-PATTERN" in w and "monte_carlo" in w.lower() for w in warnings)

    def test_no_antipatterns_for_good_config(self):
        config = _base_config()
        _, warnings = validate_ga_config(config)
        assert not any("ANTI-PATTERN" in w for w in warnings)


# =============================================================================
# PAIR VALIDATION TESTS
# =============================================================================


class TestPairValidation:

    def test_valid_pair_split(self):
        config = _base_config()
        config['pair_validation'] = {
            'enabled': True,
            'training_pairs': ['BTC/USDT'],
            'validation_pairs': ['ETH/USDT'],
            'weight_train': 0.6,
            'weight_val': 0.4,
        }
        errors, _ = validate_ga_config(config)
        assert not any("pair_validation" in e for e in errors)

    def test_empty_training_pairs(self):
        config = _base_config()
        config['pair_validation'] = {
            'enabled': True,
            'training_pairs': [],
            'validation_pairs': ['ETH/USDT'],
        }
        errors, _ = validate_ga_config(config)
        assert any("training_pairs" in e for e in errors)

    def test_weights_sum_not_one(self):
        config = _base_config()
        config['pair_validation'] = {
            'enabled': True,
            'training_pairs': ['BTC/USDT'],
            'validation_pairs': ['ETH/USDT'],
            'weight_train': 0.8,
            'weight_val': 0.8,
        }
        _, warnings = validate_ga_config(config)
        assert any("weight" in w.lower() for w in warnings)


# =============================================================================
# VALIDATE AND LOG
# =============================================================================


class TestValidateAndLog:

    def test_valid_config_returns_true(self):
        assert validate_and_log(_base_config()) is True

    def test_invalid_config_returns_false(self):
        assert validate_and_log({}) is False
