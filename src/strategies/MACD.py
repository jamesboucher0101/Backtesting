from typing import Dict, Any
import pandas as pd
import numpy as np
import random
from .base import Strategy

MACD_PARAMS = {
    'fast_length_range': (5, 50),     
    'slow_length_range': (20, 100),    
    'signal_length_range': (5, 50),   
}


class MACDStrategy(Strategy):
    """
    MACD (Moving Average Convergence Divergence) trading strategy.
    
    Based on the Pine Script strategy:
    - Uses MACD histogram (delta) crossovers for signals
    - Buy when MACD histogram crosses above zero
    - Sell when MACD histogram crosses below zero
    """
    
    @property
    def name(self) -> str:
        return "MACD"

    def suggest_parameters(self) -> Dict[str, Any]:
        """Suggest parameters for optimization trials"""
        # Get base parameters from config
        fast_length = random.randint(*MACD_PARAMS['fast_length_range'])
        slow_length = random.randint(*MACD_PARAMS['slow_length_range'])
        signal_length = random.randint(*MACD_PARAMS['signal_length_range'])
        
        # Ensure fast_length is always less than slow_length
        if fast_length >= slow_length:
            fast_length, slow_length = slow_length - 1, fast_length + 1
        
        return {
            'fast_length': fast_length,
            'slow_length': slow_length,
            'signal_length': signal_length
        }

    def prepare(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Prepare the dataframe with MACD indicators"""
        import talib

        fast_length = int(params.get('fast_length', 12))
        slow_length = int(params.get('slow_length', 26))
        signal_length = int(params.get('signal_length', 9))
        
        prices = df['close']

        # Calculate MACD using TA-Lib
        macd_line, signal_line, histogram = talib.MACD(
            prices, 
            fastperiod=fast_length, 
            slowperiod=slow_length, 
            signalperiod=signal_length
        )

        # Add MACD components to dataframe
        out = df.copy()
        out['macd'] = macd_line
        out['macd_signal'] = signal_line
        out['macd_histogram'] = histogram
        
        # Calculate the delta (histogram) as in Pine Script
        # In Pine Script: delta = MACD - aMACD
        # This is equivalent to the histogram from TA-Lib
        out['delta'] = histogram

        return out

    def warmup_period(self, params: Dict[str, Any]) -> int:
        """Return the warmup period needed for MACD calculation"""
        slow_length = int(params.get('slow_length', 26))
        signal_length = int(params.get('signal_length', 9))
        # Need enough data for the slow EMA plus signal EMA
        return slow_length + signal_length

    def decide_position(self, df: pd.DataFrame, i: int, prev_position: int, params: Dict[str, Any]) -> int:
        """Decide position based on MACD histogram crossovers"""
        if i < 1:  # Need at least 2 data points for crossover detection
            return 0
            
        current_delta = df['delta'].iloc[i]
        prev_delta = df['delta'].iloc[i - 1]
        
        # Check if we have valid delta values
        if pd.isna(current_delta) or pd.isna(prev_delta):
            return prev_position

        # Buy signal: MACD histogram crosses above zero (ta.crossover(delta, 0))
        # This means prev_delta <= 0 and current_delta > 0
        if prev_delta <= 0 and current_delta > 0:
            return 1
        
        # Sell signal: MACD histogram crosses below zero (ta.crossunder(delta, 0))
        # This means prev_delta >= 0 and current_delta < 0
        if prev_delta >= 0 and current_delta < 0:
            return -1
        
        # Hold current position if no clear signal
        return prev_position
