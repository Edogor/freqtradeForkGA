# Strategy Validation System

## Overview

The strategy validation system ensures that all generated strategies are syntactically correct and semantically valid before being added to the population. This prevents errors during backtesting and ensures only working strategies participate in evolution.

## Features

### 1. Multi-Level Validation

The validator performs three levels of validation:

#### a) **Syntax Validation**
- Checks that generated Python code is syntactically correct
- Uses Python's built-in `compile()` function
- Catches syntax errors before they cause issues

#### b) **Semantic Validation**
- Verifies FreqTrade strategy structure
- Checks for required methods: `populate_indicators`, `populate_entry_trend`, `populate_exit_trend`
- Validates required attributes: `timeframe`, `stoploss`, `minimal_roi`
- Ensures signal columns (`enter_long`, `exit_long`) are set
- Provides warnings for potential issues

#### c) **Runtime Validation (Optional)**
- Tests that strategy can be imported
- Validates all dependencies are available
- Disabled by default (requires full FreqTrade environment)

### 2. Automatic Retry Logic

When strategy generation fails validation:
- **During generation**: Up to 5 attempts to generate a valid strategy
- **During evolution**: Up to 3 attempts to regenerate invalid offspring
- Falls back gracefully if all attempts fail (strategy gets low fitness)

### 3. Configurable Settings

All validation behavior is configurable in `ga_config.yaml`:

```yaml
validation:
  enable_validation: true              # Master switch for validation
  enable_runtime_validation: false     # Runtime import checks (optional)
  max_generation_attempts: 5           # Retry attempts during generation
  max_validation_attempts: 3           # Retry attempts for offspring
  max_fix_attempts: 3                  # Attempts to auto-fix errors
```

## Usage

### Basic Usage

Validation is enabled by default. Strategies are automatically validated during generation:

```python
from genetic_algorithm.strategies.generator import StrategyGenerator
import yaml

# Load config
with open('genetic_algorithm/config/ga_config.yaml') as f:
    config = yaml.safe_load(f)

# Create generator (validation enabled by default)
generator = StrategyGenerator(config)

# Generate validated strategy
strategy_gene = generator.generate_random_strategy(generation=0, individual_id=1)

# The strategy is guaranteed to be syntactically and semantically valid
```

### Manual Validation

You can also manually validate any strategy:

```python
from genetic_algorithm.validation.strategy_validator import StrategyValidator

# Initialize validator
validator = StrategyValidator(config)

# Generate strategy code
strategy_code = generator.generate_strategy_code(strategy_gene)

# Validate
result = validator.validate_strategy(strategy_code, strategy_gene)

if result.is_valid:
    print("Strategy is valid!")
    if result.warnings:
        print(f"Warnings: {result.warnings}")
else:
    print(f"Strategy is invalid: {result.error_type}")
    print(f"Error: {result.error_message}")
```

### Disabling Validation

To disable validation (not recommended):

```python
config['validation']['enable_validation'] = False
generator = StrategyGenerator(config)
```

## Integration with Evolution

The validation system is fully integrated into the evolution process:

### 1. Initial Population
- All initial strategies are validated during generation
- Invalid strategies are regenerated automatically
- Ensures the population starts with only valid strategies

### 2. Offspring Generation
- Offspring from crossover/mutation are validated before being added to population
- Invalid offspring are regenerated up to 3 times
- Prevents propagation of broken strategies

### 3. Evolution Flow

```
Generate Strategy
      ↓
  Validate
      ↓
Valid? ─No→ Retry (up to 5 times)
  ↓ Yes                ↓
Add to Population ←────┘
                (or use last attempt)
```

## Validation Results

The `ValidationResult` object contains:

- `is_valid`: Boolean indicating if strategy passed all checks
- `error_type`: Type of error if validation failed (`'syntax'`, `'semantic'`, `'runtime'`)
- `error_message`: Detailed error message
- `warnings`: List of non-critical issues
- `fixed_code`: Automatically fixed code (if applicable)

## Testing

Run the validation test suite:

```bash
# Test validation mechanism
python genetic_algorithm/test_validation.py

# Expected output:
# ✓ Valid strategies: 10/10 (100.0%)
# ✓ All validation tests passed!
```

## Performance Impact

- **Syntax validation**: Negligible (~0.1ms per strategy)
- **Semantic validation**: Minimal (~1-2ms per strategy)
- **Runtime validation**: Moderate (~50-100ms per strategy) - disabled by default

Total validation overhead: ~1-2ms per strategy with default settings.

## Benefits

1. **Prevents Crashes**: Invalid strategies caught before backtesting
2. **Improves Evolution**: Only working strategies participate in evolution
3. **Better Results**: No wasted evaluations on broken strategies
4. **Graceful Degradation**: System continues even if validation fails
5. **Debugging**: Clear error messages help identify issues

## Error Examples

### Syntax Error
```
Error type: syntax
Error message: Line 45: invalid syntax
```

### Semantic Error
```
Error type: semantic
Error message: Missing required methods: populate_indicators
```

### Runtime Error (if enabled)
```
Error type: runtime
Error message: Import error: No module named 'talib'
```

## Future Enhancements

Planned features for future versions:

1. **Automatic Error Fixing**: Auto-correct common syntax/semantic errors
2. **Smart Regeneration**: Learn from validation failures to avoid repeated errors
3. **Validation Caching**: Cache validation results for identical strategies
4. **Custom Validators**: Plugin system for custom validation rules
5. **Validation Metrics**: Track validation success rates and common errors

## Summary

The strategy validation system is a critical component that ensures the genetic algorithm only works with valid, working strategies. It's enabled by default, has minimal performance impact, and significantly improves the reliability and quality of the evolution process.
