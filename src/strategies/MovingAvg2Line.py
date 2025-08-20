from typing import Dict, Any
import pandas as pd
import numpy as np
import random
from .base import Strategy

MOVINGAVG2LINE_PARAMS = {
    'fast_length_range': (3, 100),    # Fast SMA period (default 9 in Pine Script)
    'slow_length_range': (5, 100),   # Slow SMA period (default 18 in Pine Script)
}


class MovingAvg2LineStrategy(Strategy):
    """
    Moving Average 2-Line Cross trading strategy.
    
    Based on the Pine Script strategy:
    - Uses two Simple Moving Averages (fast and slow)
    - Buy when fast MA crosses above slow MA (ta.crossover)
    - Sell when fast MA crosses below slow MA (ta.crossunder)
    """
    
    @property
    def name(self) -> str:
        return "MovingAvg2Line"

    def suggest_parameters(self) -> Dict[str, Any]:
        """Suggest parameters for optimization trials"""
        # Get base parameters from config
        fast_min, fast_max = MOVINGAVG2LINE_PARAMS['fast_length_range']
        slow_min, slow_max = MOVINGAVG2LINE_PARAMS['slow_length_range']
        
        # Generate parameters ensuring fast < slow
        fast_length = random.randint(fast_min, fast_max)
        # Ensure slow_length is always greater than fast_length
        slow_length = random.randint(max(slow_min, fast_length + 1), slow_max)
        
        return {
            'fast_length': fast_length,
            'slow_length': slow_length
        }

    def prepare(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Prepare the dataframe with dual Moving Average indicators"""
        import talib

        fast_length = int(params.get('fast_length', 9))
        slow_length = int(params.get('slow_length', 18))
        
        prices = df['close']

        # Calculate Simple Moving Averages using TA-Lib
        # Pine Script: mafast = ta.sma(price, fastLength)
        # Pine Script: maslow = ta.sma(price, slowLength)
        fast_ma = talib.SMA(prices, timeperiod=fast_length)
        slow_ma = talib.SMA(prices, timeperiod=slow_length)

        # Add MAs to dataframe
        out = df.copy()
        out['ma_fast'] = fast_ma
        out['ma_slow'] = slow_ma

        return out

    def warmup_period(self, params: Dict[str, Any]) -> int:
        """Return the warmup period needed for dual SMA calculation"""
        slow_length = int(params.get('slow_length', 18))
        # Need enough data for the slower MA calculation
        return slow_length

    def decide_position(self, df: pd.DataFrame, i: int, prev_position: int, params: Dict[str, Any]) -> int:
        """
        Decide position based on Moving Average crossovers.
        
        Pine Script Logic:
        - if (ta.crossover(mafast, maslow)): strategy.entry("MA2CrossLE", strategy.long)
        - if (ta.crossunder(mafast, maslow)): strategy.entry("MA2CrossSE", strategy.short)
        
        ta.crossover(a, b) = a > b and a[1] <= b[1]
        ta.crossunder(a, b) = a < b and a[1] >= b[1]
        """
        if i < 1:  # Need at least 2 data points for crossover detection
            return 0
            
        # Get current and previous MA values
        current_fast = df['ma_fast'].iloc[i]
        current_slow = df['ma_slow'].iloc[i]
        prev_fast = df['ma_fast'].iloc[i-1]
        prev_slow = df['ma_slow'].iloc[i-1]
        
        # Check if we have valid values
        if pd.isna(current_fast) or pd.isna(current_slow) or pd.isna(prev_fast) or pd.isna(prev_slow):
            return prev_position

        # Pine Script: if (ta.crossover(mafast, maslow))
        # ta.crossover(mafast, maslow) = mafast > maslow and mafast[1] <= maslow[1]
        crossover = (current_fast > current_slow) and (prev_fast <= prev_slow)
        if crossover:
            return 1  # Enter long position
        
        # Pine Script: if (ta.crossunder(mafast, maslow))
        # ta.crossunder(mafast, maslow) = mafast < maslow and mafast[1] >= maslow[1]  
        crossunder = (current_fast < current_slow) and (prev_fast >= prev_slow)
        if crossunder:
            return -1  # Enter short position
        
        # No crossover signal - maintain current position
        return prev_position
