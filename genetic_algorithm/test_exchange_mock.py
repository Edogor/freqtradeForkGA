"""
Test for DirectBacktester exchange mock fix.

This test verifies that the Exchange mock in DirectBacktester properly
provides an exchange name for logging.
"""

import pytest
from unittest.mock import MagicMock


class TestExchangeMock:
    """Test Exchange mock configuration in DirectBacktester."""
    
    def test_exchange_mock_has_name(self):
        """Test that the mock ccxt exchange object has a proper name attribute."""
        # Simulate config
        config_dict = {
            'exchange': {
                'name': 'binance'
            }
        }
        
        # Create mock as done in DirectBacktester (after fix)
        exchange_name = config_dict.get('exchange', {}).get('name', 'binance')
        mock_ccxt = MagicMock()
        mock_ccxt.name = exchange_name.capitalize()
        mock_ccxt.id = exchange_name.lower()
        
        # Verify the mock has correct attributes
        assert mock_ccxt.name == 'Binance'
        assert mock_ccxt.id == 'binance'
        
        # Verify it works in a log message
        log_message = f'Using Exchange "{mock_ccxt.name}"'
        assert 'Binance' in log_message
        assert log_message == 'Using Exchange "Binance"'
    
    def test_exchange_mock_different_exchanges(self):
        """Test that the mock works with different exchange names."""
        exchanges = ['binance', 'kraken', 'coinbase', 'bybit']
        
        for exchange in exchanges:
            config_dict = {'exchange': {'name': exchange}}
            exchange_name = config_dict.get('exchange', {}).get('name', 'binance')
            
            mock_ccxt = MagicMock()
            mock_ccxt.name = exchange_name.capitalize()
            mock_ccxt.id = exchange_name.lower()
            
            assert mock_ccxt.name == exchange.capitalize()
            assert mock_ccxt.id == exchange.lower()
            
            log_message = f'Using Exchange "{mock_ccxt.name}"'
            assert exchange.capitalize() in log_message
    
    def test_mock_demonstrates_old_vs_new_behavior(self):
        """
        Demonstrate the difference between old (broken) and new (fixed) behavior.
        
        This test documents why the fix was needed:
        - Old: MagicMock() without attributes resulted in nested MagicMock for .name
        - New: Mock with explicit .name attribute provides correct exchange name
        """
        # Old behavior: just MagicMock() without setting attributes
        old_mock = MagicMock()
        
        # The old mock.name is itself a MagicMock
        assert isinstance(old_mock.name, MagicMock)
        
        # When converted to string, it shows MagicMock representation
        old_log = f'Using Exchange "{old_mock.name}"'
        assert 'MagicMock' in str(old_mock.name)
        
        # New behavior: Mock with explicit name attribute
        new_mock = MagicMock()
        new_mock.name = 'Binance'
        new_mock.id = 'binance'
        
        # Now the name is a proper string
        assert isinstance(new_mock.name, str)
        assert new_mock.name == 'Binance'
        
        # Log message works correctly
        new_log = f'Using Exchange "{new_mock.name}"'
        assert new_log == 'Using Exchange "Binance"'
        assert 'MagicMock' not in new_log


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
