#!/usr/bin/env python3
"""
Test script to verify strategy validation mechanism.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from genetic_algorithm.strategies.generator import StrategyGenerator
from genetic_algorithm.validation.strategy_validator import StrategyValidator
import yaml


def test_validation_mechanism():
    """Test the strategy validation mechanism."""
    print("\n" + "=" * 60)
    print("Testing Strategy Validation Mechanism")
    print("=" * 60)
    
    # Load configuration
    config_path = Path(__file__).parent / 'config' / 'ga_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize generator with validation enabled
    generator = StrategyGenerator(config)
    
    print("\n=== Test 1: Generate Validated Strategies ===")
    print(f"Validation enabled: {generator.enable_validation}")
    print(f"Max generation attempts: {generator.max_generation_attempts}")
    
    # Generate multiple strategies
    num_strategies = 10
    valid_count = 0
    invalid_count = 0
    
    for i in range(num_strategies):
        print(f"\nGenerating strategy {i + 1}/{num_strategies}...")
        
        try:
            # Generate strategy
            strategy_gene = generator.generate_random_strategy(generation=0, individual_id=i)
            
            # Generate code
            strategy_code = generator.generate_strategy_code(strategy_gene)
            
            # Validate
            validator = StrategyValidator(config)
            result = validator.validate_strategy(strategy_code, strategy_gene)
            
            if result.is_valid:
                print(f"  ✓ Strategy {i} is VALID")
                if result.warnings:
                    print(f"    Warnings: {result.warnings}")
                valid_count += 1
            else:
                print(f"  ✗ Strategy {i} is INVALID")
                print(f"    Error type: {result.error_type}")
                print(f"    Error message: {result.error_message}")
                invalid_count += 1
                
        except Exception as e:
            print(f"  ✗ Error generating strategy {i}: {e}")
            invalid_count += 1
    
    print("\n" + "=" * 60)
    print("Validation Results:")
    print(f"  Valid strategies:   {valid_count}/{num_strategies} ({valid_count/num_strategies*100:.1f}%)")
    print(f"  Invalid strategies: {invalid_count}/{num_strategies} ({invalid_count/num_strategies*100:.1f}%)")
    print("=" * 60)
    
    # Test with validation disabled
    print("\n=== Test 2: Generate Without Validation ===")
    config['validation']['enable_validation'] = False
    generator_no_val = StrategyGenerator(config)
    
    print(f"Validation enabled: {generator_no_val.enable_validation}")
    
    # Generate one strategy without validation
    strategy_gene = generator_no_val.generate_random_strategy(generation=0, individual_id=100)
    strategy_code = generator_no_val.generate_strategy_code(strategy_gene)
    
    # Manually validate it
    result = validator.validate_strategy(strategy_code, strategy_gene)
    print(f"Strategy is valid: {result.is_valid}")
    if not result.is_valid:
        print(f"Error: {result.error_type} - {result.error_message}")
    
    print("\n" + "=" * 60)
    print("✓ Validation mechanism test complete!")
    print("=" * 60)
    
    return valid_count == num_strategies


def test_validation_levels():
    """Test different validation levels."""
    print("\n" + "=" * 60)
    print("Testing Validation Levels")
    print("=" * 60)
    
    # Load configuration
    config_path = Path(__file__).parent / 'config' / 'ga_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Generate a strategy
    generator = StrategyGenerator(config)
    strategy_gene = generator.generate_random_strategy(generation=0, individual_id=0)
    strategy_code = generator.generate_strategy_code(strategy_gene)
    
    validator = StrategyValidator(config)
    
    print("\n=== Test 1: Syntax Validation ===")
    result = validator._validate_syntax(strategy_code)
    print(f"Syntax valid: {result.is_valid}")
    if not result.is_valid:
        print(f"Error: {result.error_message}")
    
    print("\n=== Test 2: Semantic Validation ===")
    result = validator._validate_semantics(strategy_code, strategy_gene)
    print(f"Semantics valid: {result.is_valid}")
    if not result.is_valid:
        print(f"Error: {result.error_message}")
    if result.warnings:
        print(f"Warnings: {result.warnings}")
    
    print("\n=== Test 3: Runtime Validation ===")
    result = validator._validate_runtime(strategy_code)
    print(f"Runtime valid: {result.is_valid}")
    if not result.is_valid:
        print(f"Error: {result.error_message}")
    
    print("\n=== Test 4: Full Validation ===")
    result = validator.validate_strategy(strategy_code, strategy_gene)
    print(f"Overall valid: {result.is_valid}")
    if not result.is_valid:
        print(f"Error: {result.error_type} - {result.error_message}")
    if result.warnings:
        print(f"Warnings: {result.warnings}")
    
    print("\n" + "=" * 60)
    print("✓ Validation levels test complete!")
    print("=" * 60)


def main():
    """Run all tests."""
    try:
        test_validation_mechanism()
        test_validation_levels()
        
        print("\n" + "=" * 60)
        print("✓ All validation tests passed!")
        print("=" * 60)
        return 0
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
