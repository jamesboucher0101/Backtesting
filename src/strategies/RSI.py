from typing import Dict, Any
import pandas as pd
import random
from .base import Strategy

RSI_PARAMS = {
    'length_range': (5, 15), 
    'oversold_range': (10, 40),    
    'overbought_range': (60, 90),  
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

    def suggest_parameters(self) -> Dict[str, Any]:
        """Suggest parameters for optimization trials"""
        # Get base parameters
        length = random.randint(*RSI_PARAMS['length_range'])
        oversold = random.randint(*RSI_PARAMS['oversold_range'])
        overbought = random.randint(*RSI_PARAMS['overbought_range'])
        
        return {
            'length': length,
            'oversold': oversold,
            'overbought': overbought
        }

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
        return 0

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
    