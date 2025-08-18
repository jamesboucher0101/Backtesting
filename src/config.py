# Optimization Settings
OPTIMIZATION_CONFIG = {
    'n_trials_per_pair': 50,          # More trials than 10 to better explore RSI params
    'max_drawdown_limit': 0.20,       # 20% maximum drawdown limit
    'min_trades': 20,                 # Increase for stronger statistical confidence
    'min_profit_factor': 1.3,         # Slightly higher than 1.2 to improve robustness
    'min_sharpe_ratio': 0.6,          # Higher than 0.5; filters more risky results
    'max_volatility': 0.05,           # Keep daily volatility ceiling at 5%
    'min_calmar_ratio': 0.4           # A touch higher than 0.3 for better DD control
}

