"""
Auto-generated strategy by Genetic Algorithm
Generation: 0
Individual: 0
"""

from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import numpy as np

class GAStrategy_Gen0_Ind0(IStrategy):
    """Auto-generated GA strategy"""
    
    INTERFACE_VERSION = 3
    
    # Strategy parameters
    timeframe = '15m'
    stoploss = -0.18061553772933603
    minimal_roi = {0: 0.058352635524271154, 30: 0.04291445331828134, 60: 0.03717389378869976, 120: 0.032704119289751726}
    trailing_stop = False
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Add indicators"""
        dataframe['rsi_10'] = ta.RSI(dataframe, timeperiod=10)
        bollinger = ta.BBANDS(dataframe, timeperiod=17, nbdevup=1.8919026728536643, nbdevdn=1.8919026728536643)
        dataframe['bb_upperband'] = bollinger['upperband']
        dataframe['bb_middleband'] = bollinger['middleband']
        dataframe['bb_lowerband'] = bollinger['lowerband']
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Entry signals"""
        conditions = (
            ((dataframe['rsi_10'] < 28))
        )
        dataframe.loc[conditions, 'enter_long'] = 1

        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Exit signals"""
        conditions = (
            ((dataframe['rsi_10'] > 68))
        )
        dataframe.loc[conditions, 'exit_long'] = 1

        return dataframe
