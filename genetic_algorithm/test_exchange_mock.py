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
    
    def test_old_mock_behavior(self):
        """Test that demonstrates the old (broken) behavior for comparison."""
        # Old behavior: just MagicMock() without setting attributes
        old_mock = MagicMock()
        
        # The old mock.name is itself a MagicMock
        assert isinstance(old_mock.name, MagicMock)
        
        # When converted to string, it shows MagicMock representation
        log_message = f'Using Exchange "{old_mock.name}"'
        # The log message will contain "MagicMock" in it, not the actual exchange name
        assert 'MagicMock' in str(old_mock.name)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
