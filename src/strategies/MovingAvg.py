from typing import Dict, Any
import pandas as pd
import numpy as np
import random
from .base import Strategy

MOVINGAVG_PARAMS = {
    'length_range': (5, 100),
    'confirm_bars_range': (1, 100),
}


class MovingAvgStrategy(Strategy):
    """
    Moving Average Cross trading strategy.
    
    Based on the Pine Script strategy:
    - Uses Simple Moving Average (SMA) with confirmation bars
    - Buy when price stays above SMA for specified confirmation bars
    - Sell when price stays below SMA for specified confirmation bars
    """
    
    @property
    def name(self) -> str:
        return "MovingAvg"

    def suggest_parameters(self) -> Dict[str, Any]:
        """Suggest parameters for optimization trials"""
        # Get base parameters from config
        length = random.randint(*MOVINGAVG_PARAMS['length_range'])
        confirm_bars = random.randint(*MOVINGAVG_PARAMS['confirm_bars_range'])
        
        return {
            'length': length,
            'confirm_bars': confirm_bars
        }

    def prepare(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Prepare the dataframe with Moving Average indicators"""
        import talib

        length = int(params.get('length', 9))
        confirm_bars = int(params.get('confirm_bars', 1))
        
        prices = df['close']

        # Calculate Simple Moving Average using TA-Lib
        sma = talib.SMA(prices, timeperiod=length)

        # Add SMA to dataframe
        out = df.copy()
        out['sma'] = sma
        
        # Calculate conditions for bullish and bearish signals
        out['price_above_ma'] = out['close'] > out['sma']
        out['price_below_ma'] = out['close'] < out['sma']
        
        # Initialize counters
        out['bull_count'] = 0
        out['bear_count'] = 0
        
        # Calculate consecutive bars above/below MA (matching Pine Script logic exactly)
        # Pine Script: bcount := bcond ? nz(bcount[1]) + 1 : 0
        # Pine Script: scount := scond ? nz(scount[1]) + 1 : 0
        for i in range(1, len(out)):
            if pd.notna(out['sma'].iloc[i]) and pd.notna(out['close'].iloc[i]):
                # Bull count: increment if price > ma, reset to 0 otherwise
                if out['price_above_ma'].iloc[i]:
                    out.loc[out.index[i], 'bull_count'] = out['bull_count'].iloc[i-1] + 1
                else:
                    out.loc[out.index[i], 'bull_count'] = 0
                    
                # Bear count: increment if price < ma, reset to 0 otherwise  
                if out['price_below_ma'].iloc[i]:
                    out.loc[out.index[i], 'bear_count'] = out['bear_count'].iloc[i-1] + 1
                else:
                    out.loc[out.index[i], 'bear_count'] = 0

        return out

    def warmup_period(self, params: Dict[str, Any]) -> int:
        """Return the warmup period needed for SMA calculation"""
        length = int(params.get('length', 9))
        confirm_bars = int(params.get('confirm_bars', 1))
        # Need enough data for the SMA plus confirmation bars
        return length + confirm_bars

    def decide_position(self, df: pd.DataFrame, i: int, prev_position: int, params: Dict[str, Any]) -> int:
        """Decide position based on Moving Average crossovers with confirmation"""
        if i < 1:  # Need at least 2 data points
            return 0
            
        confirm_bars = int(params.get('confirm_bars', 1))
        
        current_bull_count = df['bull_count'].iloc[i]
        current_bear_count = df['bear_count'].iloc[i]
        
        # Check if we have valid values
        if pd.isna(current_bull_count) or pd.isna(current_bear_count):
            return prev_position

        # Long Entry: price has been above MA for exactly confirm_bars consecutive bars
        # Pine Script: if (bcount == confirmBars) -> strategy.entry("MACrossLE", strategy.long)
        # This automatically closes any short position and opens long
        if current_bull_count == confirm_bars:
            return 1
        
        # Short Entry: price has been below MA for exactly confirm_bars consecutive bars
        # Pine Script: if (scount == confirmBars) -> strategy.entry("MACrossSE", strategy.short)  
        # This automatically closes any long position and opens short
        if current_bear_count == confirm_bars:
            return -1
        
        # Hold current position if no clear signal
        # Pine Script doesn't have explicit exit conditions other than opposite entry
        return prev_position
