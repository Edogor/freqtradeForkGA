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
    timeframe = '1h'
    stoploss = -0.07841653385220157
    minimal_roi = {0: 0.08044901656958701, 30: 0.05093035816478253, 60: 0.018529233882263352, 120: 0.014899894936198655}
    trailing_stop = True
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Add indicators"""
        dataframe['atr_10'] = ta.ATR(dataframe, timeperiod=10)
        dataframe['adx_15'] = ta.ADX(dataframe, timeperiod=15)
        dataframe['cci_14'] = ta.CCI(dataframe, timeperiod=14)
        macd = ta.MACD(dataframe, fastperiod=8, slowperiod=38, signalperiod=6)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']
        dataframe['ema_21'] = ta.EMA(dataframe, timeperiod=21)
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Entry signals"""
        conditions = (
            ((dataframe['adx_15'] > 28)) | 
            ((dataframe['macd'] > dataframe['macdsignal']))
        )
        dataframe.loc[conditions, 'enter_long'] = 1

        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Exit signals"""
        conditions = (
            ((dataframe['cci_14'] > 190)) & 
            ((dataframe['cci_14'] > 141)) & 
            ((dataframe['cci_14'] > 194))
        )
        dataframe.loc[conditions, 'exit_long'] = 1

        return dataframe
