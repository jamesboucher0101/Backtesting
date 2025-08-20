from typing import Dict, Any, List
import pandas as pd
import numpy as np
import itertools
from .base import Strategy

MACD_PARAMS = {
    'fast_length_range': (5, 50),     
    'fast_length_step': 2,
    'slow_length_range': (20, 100),    
    'slow_length_step': 3,
    'signal_length_range': (5, 50),
    'signal_length_step': 2,
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

    def suggest_parameters(self, trial=None) -> Dict[str, Any]:
        """
        Suggest parameters for optimization trials.
        
        Note: This method is now mainly for compatibility. Parameter combinations 
        are generated externally in parameter_combinations.py
        
        Args:
            trial: Optional trial parameter for compatibility with optimization frameworks.
        """
        # Return default parameters for compatibility
        return {
            'fast_length': 12,
            'slow_length': 26,
            'signal_length': 9
        }

    @classmethod
    def get_parameter_combinations(cls) -> List[Dict[str, Any]]:
        """
        Generate all parameter combinations for MACD strategy.
        
        Returns:
            List of parameter dictionaries for MACD strategy
        """
        # Use MACD_PARAMS configuration
        fast_min, fast_max = MACD_PARAMS['fast_length_range']
        fast_step = MACD_PARAMS['fast_length_step']
        slow_min, slow_max = MACD_PARAMS['slow_length_range']
        slow_step = MACD_PARAMS['slow_length_step']
        signal_min, signal_max = MACD_PARAMS['signal_length_range']
        signal_step = MACD_PARAMS['signal_length_step']
        
        # Generate ranges with configurable steps
        fast_length_range = range(fast_min, fast_max + 1, fast_step)
        slow_length_range = range(slow_min, slow_max + 1, slow_step)
        signal_length_range = range(signal_min, signal_max + 1, signal_step)
        
        combinations = []
        for fast_length, slow_length, signal_length in itertools.product(
            fast_length_range, slow_length_range, signal_length_range
        ):
            # Ensure fast_length < slow_length
            if fast_length < slow_length:
                combinations.append({
                    'fast_length': fast_length,
                    'slow_length': slow_length,
                    'signal_length': signal_length
                })
        
        return combinations

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
