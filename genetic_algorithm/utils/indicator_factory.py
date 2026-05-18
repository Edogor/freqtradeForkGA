"""
Indicator Factory

Shared utility for creating random indicators used by both
strategy generation and mutation operations.
"""

import random
from typing import Dict, Any, Optional

from genetic_algorithm.core.strategy_gene import IndicatorGene


def _scale_period_for_timeframe(period_range: list, timeframe: Optional[str] = None) -> list:
    """Scale indicator period range based on strategy timeframe.

    Shorter timeframes need shorter lookback periods so indicators react
    faster.  The reference baseline is 1h (factor=1.0).

    Returns a new [min, max] range with values clamped to [3, original_max].
    """
    if not timeframe:
        return period_range

    _TF_MINUTES = {
        '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
        '1h': 60, '2h': 120, '4h': 240, '6h': 360, '1d': 1440,
    }
    tf_min = _TF_MINUTES.get(timeframe, 60)
    # Factor: 1h is baseline (factor=1.0), 5m gets factor ~0.42 (sqrt-based)
    # Using sqrt to avoid overly aggressive scaling on very short TFs
    import math
    factor = math.sqrt(tf_min / 60.0)
    factor = max(0.25, min(1.0, factor))  # clamp [0.25, 1.0]

    lo = max(3, int(period_range[0] * factor))
    hi = max(lo + 1, int(period_range[1] * factor))
    return [lo, hi]


def create_random_indicator(indicator_type: str, indicator_config: Dict[str, Any],
                            timeframe: Optional[str] = None) -> IndicatorGene:
    """
    Create a random indicator of the given type.
    
    This function consolidates the indicator generation logic that was
    previously duplicated in both generator.py and mutation.py.
    
    Args:
        indicator_type: Type of indicator to create (e.g., 'RSI', 'MACD')
        indicator_config: Configuration dict with parameter ranges for indicators
        timeframe: Base strategy timeframe (e.g., '5m'). When provided and
            the config has ``timeframe_adaptive_indicators: true``, period
            ranges are scaled so shorter TFs get faster indicators.
    
    Returns:
        IndicatorGene: A new indicator with random parameters
    """
    # Sanitize corrupted CDL type names (e.g., CDL_MORNINGSTAR_0_0 -> CDL_MORNINGSTAR)
    if indicator_type.startswith('CDL_'):
        from genetic_algorithm.core.strategy_gene import StrategyGene
        indicator_type = StrategyGene._strip_cdl_suffixes(indicator_type)
    
    ind_config = indicator_config.get(indicator_type, {})
    parameters = {}

    # Check if TF-adaptive scaling is enabled
    tf_adaptive = indicator_config.get('timeframe_adaptive_indicators',
                                       indicator_config.get('_tf_adaptive', False))
    _tf = timeframe if tf_adaptive else None

    def _period(key: str, default: list) -> int:
        """Get a random period, optionally TF-scaled."""
        raw = ind_config.get(key, default)
        scaled = _scale_period_for_timeframe(raw, _tf)
        return random.randint(*scaled)
    
    if indicator_type == 'RSI':
        parameters['period'] = _period('period', [7, 21])
    
    elif indicator_type == 'MACD':
        parameters['fast_period'] = _period('fast_period', [8, 21])
        parameters['slow_period'] = _period('slow_period', [21, 50])
        parameters['signal_period'] = _period('signal_period', [5, 14])
    
    elif indicator_type == 'BBANDS':
        parameters['period'] = _period('period', [15, 30])
        parameters['std_dev'] = random.uniform(*ind_config.get('std_dev', [1.5, 3.0]))
    
    elif indicator_type in ['EMA', 'SMA']:
        parameters['period'] = _period('period', [10, 50])
    
    elif indicator_type == 'STOCH':
        parameters['k_period'] = _period('k_period', [5, 21])
        parameters['d_period'] = _period('d_period', [3, 14])
    
    elif indicator_type in ['ATR', 'ADX', 'CCI']:
        parameters['period'] = _period('period', [10, 20])
    
    # New indicators for richer grammar
    elif indicator_type == 'MFI':
        # Money Flow Index (volume-weighted RSI)
        parameters['period'] = _period('period', [10, 20])
    
    elif indicator_type == 'OBV':
        # On-Balance Volume (no parameters, uses volume)
        parameters = {}
    
    elif indicator_type == 'WILLR':
        # Williams %R
        parameters['period'] = _period('period', [10, 20])
    
    elif indicator_type == 'ROC':
        # Rate of Change
        parameters['period'] = _period('period', [5, 20])
    
    elif indicator_type == 'TEMA':
        # Triple Exponential Moving Average
        parameters['period'] = _period('period', [10, 30])
    
    elif indicator_type == 'KAMA':
        # Kaufman Adaptive Moving Average
        parameters['period'] = _period('period', [10, 30])
    
    elif indicator_type == 'SAR':
        # Parabolic SAR
        parameters['acceleration'] = random.uniform(*ind_config.get('acceleration', [0.01, 0.05]))
        parameters['maximum'] = random.uniform(*ind_config.get('maximum', [0.1, 0.3]))
    
    elif indicator_type == 'AROON':
        # Aroon indicator (trend strength)
        parameters['period'] = _period('period', [10, 25])
    
    # === NEW INDICATORS ===
    
    elif indicator_type == 'SUPERTREND':
        # SuperTrend - trend following with dynamic stop
        parameters['period'] = _period('period', [7, 14])
        parameters['multiplier'] = random.uniform(*ind_config.get('multiplier', [2.0, 4.0]))
    
    elif indicator_type == 'ICHIMOKU':
        # Ichimoku Cloud - comprehensive trend/support/resistance
        parameters['tenkan_period'] = _period('tenkan_period', [7, 12])
        parameters['kijun_period'] = _period('kijun_period', [20, 30])
        parameters['senkou_b_period'] = _period('senkou_b_period', [40, 60])
    
    elif indicator_type == 'DONCHIAN':
        # Donchian Channels - breakout detection
        parameters['period'] = _period('period', [10, 30])
    
    elif indicator_type == 'VWAP':
        # Volume Weighted Average Price - intraday mean reversion anchor
        parameters['period'] = _period('period', [10, 30])
    
    elif indicator_type == 'CMF':
        # Chaikin Money Flow - volume-based momentum
        parameters['period'] = _period('period', [10, 25])
    
    elif indicator_type == 'VROC':
        # Volume Rate of Change
        parameters['period'] = _period('period', [5, 20])
    
    elif indicator_type == 'PSAR':
        # Parabolic SAR (alias for SAR)
        parameters['acceleration'] = random.uniform(*ind_config.get('acceleration', [0.01, 0.05]))
        parameters['maximum'] = random.uniform(*ind_config.get('maximum', [0.1, 0.3]))
    
    # === CANDLESTICK PATTERNS ===
    # TALib CDL* functions detect candlestick patterns
    # All patterns have no parameters - they analyze OHLC data directly
    
    elif indicator_type == 'CDL_ENGULFING':
        # Bullish/Bearish Engulfing pattern
        parameters = {}
    
    elif indicator_type == 'CDL_HAMMER':
        # Hammer / Hanging Man pattern
        parameters = {}
    
    elif indicator_type == 'CDL_DOJI':
        # Doji pattern (indecision)
        parameters = {}
    
    elif indicator_type == 'CDL_MORNINGSTAR':
        # Morning Star (bullish reversal)
        parameters['penetration'] = random.uniform(*ind_config.get('penetration', [0.0, 0.3]))
    
    elif indicator_type == 'CDL_EVENINGSTAR':
        # Evening Star (bearish reversal)
        parameters['penetration'] = random.uniform(*ind_config.get('penetration', [0.0, 0.3]))
    
    elif indicator_type == 'CDL_SHOOTINGSTAR':
        # Shooting Star (bearish reversal)
        parameters = {}
    
    elif indicator_type == 'CDL_HARAMI':
        # Harami pattern (reversal)
        parameters = {}
    
    elif indicator_type == 'CDL_PIERCING':
        # Piercing Line (bullish reversal)
        parameters = {}
    
    elif indicator_type == 'CDL_DARKCLOUD':
        # Dark Cloud Cover (bearish reversal)
        parameters = {}
    
    elif indicator_type == 'CDL_3WHITESOLDIERS':
        # Three White Soldiers (strong bullish)
        parameters = {}
    
    elif indicator_type == 'CDL_3BLACKCROWS':
        # Three Black Crows (strong bearish)
        parameters = {}
    
    # Validate MACD parameters (slow must be > fast)
    if indicator_type == 'MACD':
        if parameters.get('fast_period', 12) >= parameters.get('slow_period', 26):
            parameters['fast_period'] = max(8, parameters['slow_period'] - 5)
    
    return IndicatorGene(
        type=indicator_type,
        parameters=parameters,
        weight=random.uniform(0.3, 1.0)
    )
