"""
Strategy Validator

Validates generated strategies to ensure they are syntactically correct,
semantically valid, and can execute without errors.
"""

import ast
import logging
import tempfile
import sys
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass

from genetic_algorithm.core.strategy_gene import StrategyGene

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of strategy validation."""
    is_valid: bool
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    fixed_code: Optional[str] = None
    warnings: list = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class StrategyValidator:
    """
    Validates and fixes generated trading strategies.
    
    Performs multiple levels of validation:
    1. Syntax validation - checks Python syntax
    2. Semantic validation - checks FreqTrade compatibility
    3. Runtime validation - tests strategy can be instantiated
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize validator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.validation_config = self.config.get('validation', {})
        self.max_fix_attempts = self.validation_config.get('max_fix_attempts', 3)
        self.enable_runtime_validation = self.validation_config.get('enable_runtime_validation', False)
        
    def validate_strategy(self, strategy_code: str, strategy_gene: StrategyGene = None) -> ValidationResult:
        """
        Validate a generated strategy.
        
        Args:
            strategy_code: Python code of the strategy
            strategy_gene: Optional StrategyGene for additional validation
            
        Returns:
            ValidationResult with validation status and any errors
        """
        # Step 1: Syntax validation
        syntax_result = self._validate_syntax(strategy_code)
        if not syntax_result.is_valid:
            # Try to fix syntax errors
            fixed_result = self._try_fix_syntax_errors(strategy_code, syntax_result)
            if fixed_result and fixed_result.is_valid:
                return fixed_result
            return syntax_result
        
        # Step 2: Semantic validation
        semantic_result = self._validate_semantics(strategy_code, strategy_gene)
        if not semantic_result.is_valid:
            # Try to fix semantic errors
            fixed_result = self._try_fix_semantic_errors(strategy_code, semantic_result)
            if fixed_result and fixed_result.is_valid:
                return fixed_result
            return semantic_result
        
        # Step 3: Runtime validation (can the strategy be imported?) - Optional
        if self.enable_runtime_validation:
            runtime_result = self._validate_runtime(strategy_code)
            if not runtime_result.is_valid:
                return runtime_result
        
        # All validations passed
        return ValidationResult(
            is_valid=True,
            warnings=syntax_result.warnings + semantic_result.warnings
        )
    
    def _validate_syntax(self, strategy_code: str) -> ValidationResult:
        """
        Validate Python syntax of strategy code.
        
        Args:
            strategy_code: Python code to validate
            
        Returns:
            ValidationResult
        """
        try:
            compile(strategy_code, '<string>', 'exec')
            return ValidationResult(is_valid=True)
        except SyntaxError as e:
            logger.warning(f"Syntax error in strategy: {e}")
            return ValidationResult(
                is_valid=False,
                error_type='syntax',
                error_message=f"Line {e.lineno}: {e.msg}"
            )
        except Exception as e:
            logger.warning(f"Compilation error in strategy: {e}")
            return ValidationResult(
                is_valid=False,
                error_type='compilation',
                error_message=str(e)
            )
    
    def _validate_semantics(self, strategy_code: str, strategy_gene: StrategyGene = None) -> ValidationResult:
        """
        Validate semantic correctness of strategy.
        
        Checks:
        - Required methods exist
        - Required attributes exist
        - Strategy structure is correct
        
        Args:
            strategy_code: Python code to validate
            strategy_gene: Optional StrategyGene for cross-checking
            
        Returns:
            ValidationResult
        """
        warnings = []
        
        try:
            # Parse the code into AST
            tree = ast.parse(strategy_code)
            
            # Find the strategy class
            strategy_class = None
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    strategy_class = node
                    break
            
            if not strategy_class:
                return ValidationResult(
                    is_valid=False,
                    error_type='semantic',
                    error_message="No class definition found"
                )
            
            # Check required methods
            required_methods = ['populate_indicators', 'populate_entry_trend', 'populate_exit_trend']
            found_methods = set()
            
            for item in strategy_class.body:
                if isinstance(item, ast.FunctionDef):
                    found_methods.add(item.name)
            
            missing_methods = set(required_methods) - found_methods
            if missing_methods:
                return ValidationResult(
                    is_valid=False,
                    error_type='semantic',
                    error_message=f"Missing required methods: {', '.join(missing_methods)}"
                )
            
            # Check for required attributes
            required_attrs = ['timeframe', 'stoploss', 'minimal_roi']
            found_attrs = set()
            
            for item in strategy_class.body:
                if isinstance(item, ast.Assign):
                    for target in item.targets:
                        if isinstance(target, ast.Name):
                            found_attrs.add(target.id)
            
            missing_attrs = set(required_attrs) - found_attrs
            if missing_attrs:
                warnings.append(f"Missing recommended attributes: {', '.join(missing_attrs)}")
            
            # Check signal columns are set correctly
            entry_sets_signal = False
            exit_sets_signal = False
            
            for item in strategy_class.body:
                if isinstance(item, ast.FunctionDef):
                    if item.name == 'populate_entry_trend':
                        # Check if 'enter_long' is set somewhere in the function body
                        # Look for patterns like: dataframe.loc[conditions, 'enter_long'] = 1
                        # or dataframe['enter_long'] = ...
                        code_str = ast.get_source_segment(strategy_code, item) or ""
                        if 'enter_long' in code_str:
                            entry_sets_signal = True
                    
                    elif item.name == 'populate_exit_trend':
                        # Check if 'exit_long' is set somewhere in the function body
                        code_str = ast.get_source_segment(strategy_code, item) or ""
                        if 'exit_long' in code_str:
                            exit_sets_signal = True
            
            if not entry_sets_signal:
                warnings.append("Entry trend method doesn't appear to set 'enter_long' signal")
            
            if not exit_sets_signal:
                warnings.append("Exit trend method doesn't appear to set 'exit_long' signal")
            
            return ValidationResult(is_valid=True, warnings=warnings)
            
        except Exception as e:
            logger.warning(f"Semantic validation error: {e}")
            return ValidationResult(
                is_valid=False,
                error_type='semantic',
                error_message=f"AST parsing error: {str(e)}"
            )
    
    def _validate_runtime(self, strategy_code: str) -> ValidationResult:
        """
        Validate strategy can be imported and instantiated.
        
        Args:
            strategy_code: Python code to validate
            
        Returns:
            ValidationResult
        """
        # Create a temporary module and try to import it
        temp_dir = tempfile.mkdtemp()
        temp_file = Path(temp_dir) / "temp_strategy.py"
        
        try:
            # Write strategy to temp file
            with open(temp_file, 'w') as f:
                f.write(strategy_code)
            
            # Add temp directory to Python path temporarily
            sys.path.insert(0, temp_dir)
            
            try:
                # Try to import the module
                import importlib.util
                spec = importlib.util.spec_from_file_location("temp_strategy", temp_file)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # Find the strategy class
                    strategy_class = None
                    for name in dir(module):
                        obj = getattr(module, name)
                        if isinstance(obj, type) and name.startswith('GAStrategy'):
                            strategy_class = obj
                            break
                    
                    if not strategy_class:
                        return ValidationResult(
                            is_valid=False,
                            error_type='runtime',
                            error_message="No strategy class found in module"
                        )
                    
                    # Try to instantiate (without config - just check it doesn't crash)
                    # Note: Full instantiation requires FreqTrade config, so we just check imports work
                    return ValidationResult(is_valid=True)
                    
                else:
                    return ValidationResult(
                        is_valid=False,
                        error_type='runtime',
                        error_message="Failed to create module spec"
                    )
                    
            except ImportError as e:
                logger.warning(f"Import error in strategy: {e}")
                return ValidationResult(
                    is_valid=False,
                    error_type='runtime',
                    error_message=f"Import error: {str(e)}"
                )
            except Exception as e:
                logger.warning(f"Runtime error in strategy: {e}")
                return ValidationResult(
                    is_valid=False,
                    error_type='runtime',
                    error_message=f"Runtime error: {str(e)}"
                )
            
        finally:
            # Cleanup
            if temp_dir in sys.path:
                sys.path.remove(temp_dir)
            
            # Remove temp file and directory
            try:
                temp_file.unlink(missing_ok=True)
                Path(temp_dir).rmdir()
            except:
                pass
    
    def _try_fix_syntax_errors(self, strategy_code: str, error_result: ValidationResult) -> Optional[ValidationResult]:
        """
        Attempt to automatically fix common syntax errors.
        
        Args:
            strategy_code: Original code with errors
            error_result: ValidationResult with error details
            
        Returns:
            ValidationResult if fixed, None if unable to fix
        """
        # Common fixes:
        # 1. Missing colons at end of function/class definitions
        # 2. Incorrect indentation
        # 3. Missing parentheses
        
        logger.info("Attempting to fix syntax errors...")
        
        # For now, return None (no automatic fixes)
        # TODO: Implement common syntax fixes
        return None
    
    def _try_fix_semantic_errors(self, strategy_code: str, error_result: ValidationResult) -> Optional[ValidationResult]:
        """
        Attempt to automatically fix common semantic errors.
        
        Args:
            strategy_code: Original code with errors
            error_result: ValidationResult with error details
            
        Returns:
            ValidationResult if fixed, None if unable to fix
        """
        logger.info("Attempting to fix semantic errors...")
        
        # For now, return None (no automatic fixes)
        # TODO: Implement common semantic fixes like:
        # - Adding missing signal column assignments
        # - Fixing indicator references
        return None
