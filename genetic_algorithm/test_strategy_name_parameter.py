"""
Test for strategy name parameter in StrategyGenerator.

Verifies that generate_strategy_code accepts an optional strategy_name parameter
and generates code with the correct class name.
"""

import pytest
import sys
sys.path.insert(0, '/home/runner/work/freqtradeForkGA/freqtradeForkGA')

from genetic_algorithm.strategies.generator import StrategyGenerator
from genetic_algorithm.core.strategy_gene import StrategyGene
import yaml


class TestStrategyNameParameter:
    """Test strategy_name parameter in generate_strategy_code."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Load config
        with open('/home/runner/work/freqtradeForkGA/freqtradeForkGA/genetic_algorithm/config/ga_config.yaml', 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.generator = StrategyGenerator(self.config)
    
    def test_default_strategy_name(self):
        """Test that default strategy name is generated from gene."""
        # Create a simple strategy gene
        strategy_gene = self.generator.generate_random_strategy(generation=5, individual_id=42)
        
        # Generate code without custom name
        code = self.generator.generate_strategy_code(strategy_gene)
        
        # Check that default name is used
        expected_name = "GAStrategy_Gen5_Ind42"
        assert f"class {expected_name}(IStrategy):" in code
        assert expected_name in code
    
    def test_custom_strategy_name(self):
        """Test that custom strategy name is used when provided."""
        # Create a simple strategy gene
        strategy_gene = self.generator.generate_random_strategy(generation=5, individual_id=42)
        
        # Generate code with custom name
        custom_name = "GAStrategy_Gen5_Ind42_W0_Val"
        code = self.generator.generate_strategy_code(strategy_gene, strategy_name=custom_name)
        
        # Check that custom name is used
        assert f"class {custom_name}(IStrategy):" in code
        assert custom_name in code
        
        # Check that default name is NOT used
        default_name = "GAStrategy_Gen5_Ind42"
        # The default name might appear in comments, but not in the class definition
        assert f"class {default_name}(IStrategy):" not in code
    
    def test_walk_forward_window_names(self):
        """Test generating code for walk-forward window names."""
        strategy_gene = self.generator.generate_random_strategy(generation=0, individual_id=10)
        
        # Generate code for multiple windows
        base_name = "GAStrategy_Gen0_Ind10"
        window_names = [
            f"{base_name}_W0_Val",
            f"{base_name}_W1_Val",
            f"{base_name}_W2_Val"
        ]
        
        for window_name in window_names:
            code = self.generator.generate_strategy_code(strategy_gene, strategy_name=window_name)
            
            # Verify correct class name
            assert f"class {window_name}(IStrategy):" in code
            
            # Verify the code is valid Python (at least syntactically)
            try:
                compile(code, '<string>', 'exec')
            except SyntaxError as e:
                pytest.fail(f"Generated code has syntax error: {e}")
    
    def test_strategy_name_none_uses_default(self):
        """Test that passing None explicitly uses default name."""
        strategy_gene = self.generator.generate_random_strategy(generation=3, individual_id=7)
        
        # Generate with explicit None
        code_with_none = self.generator.generate_strategy_code(strategy_gene, strategy_name=None)
        
        # Generate without parameter
        code_without_param = self.generator.generate_strategy_code(strategy_gene)
        
        # Both should generate the same code
        assert code_with_none == code_without_param
        
        # Both should have default name
        expected_name = "GAStrategy_Gen3_Ind7"
        assert f"class {expected_name}(IStrategy):" in code_with_none
        assert f"class {expected_name}(IStrategy):" in code_without_param


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
