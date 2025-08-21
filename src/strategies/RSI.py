from typing import Dict, Any, List
import pandas as pd
import itertools
from .base import Strategy

RSI_PARAMS = {
    'length_range': (5, 30), 
    'length_step': 2,
    'oversold_range': (20, 40),    
    'oversold_step': 3,
    'overbought_range': (60, 80),
    'overbought_step': 3,
}

class RSIStrategy(Strategy):
    """
    Simple RSI (Relative Strength Index) trading strategy.
    
    Buy when RSI crosses above the oversold threshold; sell when RSI crosses below
    the overbought threshold.
    """
    
    @property
    def name(self) -> str:
        return "RSI"

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
            'length': 14,
            'oversold': 30,
            'overbought': 70
        }

    @classmethod
    def get_parameter_combinations(cls) -> List[Dict[str, Any]]:
        """
        Generate all parameter combinations for RSI strategy.
        
        Returns:
            List of parameter dictionaries for RSI strategy
        """
        # Use RSI_PARAMS configuration
        length_min, length_max = RSI_PARAMS['length_range']
        length_step = RSI_PARAMS['length_step']
        oversold_min, oversold_max = RSI_PARAMS['oversold_range']
        oversold_step = RSI_PARAMS['oversold_step']
        overbought_min, overbought_max = RSI_PARAMS['overbought_range']
        overbought_step = RSI_PARAMS['overbought_step']
        
        # Generate ranges with configurable steps
        length_range = range(length_min, length_max + 1, length_step)
        oversold_range = range(oversold_min, oversold_max + 1, oversold_step)
        overbought_range = range(overbought_min, overbought_max + 1, overbought_step)
        
        combinations = []
        for length, oversold, overbought in itertools.product(length_range, oversold_range, overbought_range):
            combinations.append({
                'length': length,
                'oversold': oversold,
                'overbought': overbought
            })
        
        return combinations

    def prepare(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Prepare the dataframe with RSI indicator"""
        import talib

        length = int(params.get('length', 14))
        prices = df['close']

        # Use TA-Lib's RSI function
        rsi = talib.RSI(prices, timeperiod=length)

        # Add RSI to dataframe
        out = df.copy()
        out['rsi'] = rsi

        return out

    def warmup_period(self, params: Dict[str, Any]) -> int:
        # The RSI indicator requires 'length' periods to produce its first value.
        return int(params.get('length', 14))

    def decide_position(self, df: pd.DataFrame, i: int, prev_position: int, params: Dict[str, Any]) -> int:
        """Decide position based on basic RSI cross signals"""
        if i < 1:  # Need at least 2 data points
            return 0
            
        oversold = params.get('oversold', 30)
        overbought = params.get('overbought', 70)
        
        current_rsi = df['rsi'].iloc[i]
        prev_rsi = df['rsi'].iloc[i - 1]
        
        # Check if we have valid RSI values
        if pd.isna(current_rsi) or pd.isna(prev_rsi):
            return prev_position

        # Buy signal: RSI crosses above oversold threshold
        if (prev_rsi < oversold and current_rsi >= oversold and 
            prev_position <= 0):
            return 1
        
        # Sell signal: RSI crosses below overbought threshold
        if (prev_rsi > overbought and current_rsi <= overbought and 
            prev_position >= 0):
            return -1
        
        # Hold current position
        return prev_position
    