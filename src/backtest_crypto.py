import os
import json
import csv
import datetime
import pandas as pd
import threading

# Global lock for thread-safe file writing
file_write_lock = threading.Lock()

# Global cache for loaded data to avoid redundant reads
data_cache = {}
data_cache_lock = threading.Lock()

def timeframe_to_pandas_rule(tf):
    if tf.endswith('m'):
        return f"{int(tf[:-1])}min"
    elif tf.endswith('h'):
        return f"{int(tf[:-1])}h"  # Fixed: changed 'H' to 'h' to avoid deprecation warning
    elif tf.endswith('d'):
        return f"{int(tf[:-1])}D"
    else:
        raise ValueError(f"Unsupported timeframe: {tf}")

def load_data_crypto(exchange, trading_pair, timeframe):
    # Create a cache key
    cache_key = f"{exchange}_{trading_pair}_{timeframe}"
    
    # Check if data is already cached
    with data_cache_lock:
        if cache_key in data_cache:
            return data_cache[cache_key].copy()  # Return a copy to avoid modifications
    
    # Load data if not cached
    csv_path = f'data/crypto/{exchange}_{trading_pair.replace("/", "_")}.csv'.lower()
    df = pd.read_csv(csv_path)

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        df = df.set_index('timestamp')
    else:
        raise ValueError("CSV must contain a 'timestamp' column.")

    resample_rule = timeframe_to_pandas_rule(timeframe)

    ohlc_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    resampled_df = df.resample(resample_rule).agg(ohlc_dict).dropna()
    
    # Cache the data
    with data_cache_lock:
        data_cache[cache_key] = resampled_df.copy()
    
    return resampled_df

def simulate_backtest_crypto(strategy, exchange, trading_pair, timeframe, single_params=None, initial_capital=10000, commission_rate=0.0006, order_size=1.0, tick_slippage=2):
    """
    Run backtest for a strategy with a single parameter set.
    
    Args:
        strategy: Strategy instance
        exchange: Exchange name
        trading_pair: Trading pair
        timeframe: Timeframe
        single_params: Single parameter dictionary to test. If None, uses strategy.suggest_parameters()
        initial_capital: Initial capital
        commission_rate: Commission rate
        order_size: Fixed order size (quantity)
        tick_slippage: Number of ticks of slippage to apply (default 2 ticks)
    """
    df = load_data_crypto(exchange, trading_pair, timeframe)
    
    # If no parameters provided, use default parameters (fallback)
    if single_params is None:
        single_params = strategy.suggest_parameters()
    
    print ("-" * 50)
    print (f"Backtesting {strategy.name} on {exchange} {trading_pair} {timeframe}")
    print (f"Parameters: {single_params}")
    
    # Run single backtest with these parameters
    run_single_backtest_crypto(strategy, df, single_params, exchange, trading_pair, timeframe, 
                    initial_capital, commission_rate, order_size, tick_slippage)

def run_single_backtest_crypto(strategy, df, params, exchange, trading_pair, timeframe, 
                       initial_capital=10000, commission_rate=0.0006, order_size=1.0, tick_slippage=2):
    """
    Run a single backtest with specific parameters.
    
    This function contains the core backtesting logic.
    """
    
    def calculate_tick_size(price):
        """Calculate tick size based on price level (common crypto convention)"""
        if price < 1:
            return 0.0001  # 0.01 cent tick for sub-dollar prices
        elif price < 10:
            return 0.001   # 0.1 cent tick
        elif price < 100:
            return 0.01    # 1 cent tick
        else:
            return 0.1     # 10 cent tick
    
    def apply_tick_slippage(base_price, ticks, direction):
        """Apply tick-based slippage
        Args:
            base_price: Base execution price
            ticks: Number of ticks to apply
            direction: 'unfavorable_buy' (add ticks) or 'unfavorable_sell' (subtract ticks)
        """
        tick_size = calculate_tick_size(base_price)
        if direction == 'unfavorable_buy':
            return base_price + (ticks * tick_size)
        else:  # unfavorable_sell
            return base_price - (ticks * tick_size)
    # Prepare the dataframe with strategy indicators
    df = strategy.prepare(df.copy(), params)
    df['position'] = 0
    df['equity'] = float(initial_capital)  # Ensure float dtype to avoid pandas warning
    df['trades'] = 0
    
    # Initialize first equity value properly
    df.loc[df.index[0], 'equity'] = float(initial_capital)
    
    position = 0
    entry_price = 0
    entry_execution_price = 0  # Store original execution price for commission calculation
    position_qty = 0.0
    trades = []
    length = len(df)
    
    # Get warmup period to avoid trading during indicator stabilization
    warmup_period = strategy.warmup_period(params)

    try:
        for i in range(warmup_period, length - 1):
            # Use current bar's close for decision and next bar's open for execution (realistic timing)
            decision_price = df['close'].iloc[i]  # Price available for decision
            base_execution_price = df['open'].iloc[i+1]  # Base execution price (next bar open)
            
            # Get desired position from strategy
            desired_position = strategy.decide_position(df, i, position, params)
            
            equity_prev = df['equity'].iloc[i] if i >= 0 else float(initial_capital)
            equity_curr = equity_prev

            # If changing position, close existing one first
            if position > 0 and desired_position <= 0:
                # Apply tick slippage for long exit (unfavorable = lower price)
                exit_price = apply_tick_slippage(base_execution_price, tick_slippage, 'unfavorable_sell')
                qty = position_qty
                # Calculate commission on execution price
                commission_entry = entry_execution_price * qty * commission_rate  # Use stored execution price
                commission_exit = exit_price * qty * commission_rate  # Use actual exit price with slippage
                pnl_currency = (exit_price - entry_price) * qty - commission_entry - commission_exit
                # Calculate P&L% based on position value (TradingView standard)
                position_value = entry_price * qty
                pnl_pct = (pnl_currency / position_value) * 100 if position_value != 0 else 0
                trades.append({
                    'type': 'long_exit',
                    'entry': entry_price,
                    'exit': exit_price,
                    'qty': qty,
                    'pnl_currency': pnl_currency,
                    'pnl_pct': pnl_pct,
                    'commission_entry': commission_entry,
                    'commission_exit': commission_exit,
                    'order_size': order_size,
                    'timestamp': df.index[i]
                })
                equity_curr = equity_prev + pnl_currency
                entry_price = 0
                entry_execution_price = 0
                position_qty = 0.0
            elif position < 0 and desired_position >= 0:
                # Apply tick slippage for short exit (unfavorable = higher price)
                exit_price = apply_tick_slippage(base_execution_price, tick_slippage, 'unfavorable_buy')
                qty = position_qty
                # Calculate commission on execution price
                commission_entry = entry_execution_price * qty * commission_rate  # Use stored execution price
                commission_exit = exit_price * qty * commission_rate  # Use actual exit price with slippage
                pnl_currency = (entry_price - exit_price) * qty - commission_entry - commission_exit
                # Calculate P&L% based on position value (TradingView standard)
                position_value = entry_price * qty
                pnl_pct = (pnl_currency / position_value) * 100 if position_value != 0 else 0
                trades.append({
                    'type': 'short_exit',
                    'entry': entry_price,
                    'exit': exit_price,
                    'qty': qty,
                    'pnl_currency': pnl_currency,
                    'pnl_pct': pnl_pct,
                    'commission_entry': commission_entry,
                    'commission_exit': commission_exit,
                    'order_size': order_size,
                    'timestamp': df.index[i]
                })
                equity_curr = equity_prev + pnl_currency
                entry_price = 0
                entry_execution_price = 0
                position_qty = 0.0

            if desired_position != position:
                if desired_position != 0:
                    # Apply tick slippage for entry based on position direction
                    if desired_position > 0:  # Long entry
                        execution_price = apply_tick_slippage(base_execution_price, tick_slippage, 'unfavorable_buy')
                    else:  # Short entry
                        execution_price = apply_tick_slippage(base_execution_price, tick_slippage, 'unfavorable_sell')
                    
                    # Use fixed order size (quantity)
                    position_qty = order_size
                    
                    # Store execution prices for later use
                    entry_execution_price = execution_price
                    entry_price = execution_price
                    
                    # Check if we have enough capital for this trade
                    total_cost = position_qty * execution_price * (1 + commission_rate)
                    if total_cost > equity_curr:  # Not enough capital
                        position_qty = 0.0
                        desired_position = 0
                    else:
                        position_qty = round(position_qty, 8)  # Round to 8 decimal places
                else:
                    entry_price = 0
                    position_qty = 0.0
                
                position = desired_position

            df.loc[df.index[i+1], 'equity'] = equity_curr
            df.loc[df.index[i+1], 'position'] = position
            df.loc[df.index[i+1], 'trades'] = len(trades)
    except Exception as e:
        print ("E", e)

    # Handle any remaining open position at the end
    if position != 0 and position_qty > 0:
        final_price = df['close'].iloc[-1]
        if position > 0:  # Close long position
            exit_price = final_price
            commission_entry = entry_execution_price * position_qty * commission_rate  # Use stored execution price
            commission_exit = final_price * position_qty * commission_rate  # Use base final price
            final_pnl = (exit_price - entry_price) * position_qty - commission_entry - commission_exit
        else:  # Close short position
            exit_price = final_price
            commission_entry = entry_execution_price * position_qty * commission_rate  # Use stored execution price
            commission_exit = final_price * position_qty * commission_rate  # Use base final price
            final_pnl = (entry_price - exit_price) * position_qty - commission_entry - commission_exit
        
        # Calculate P&L% based on position value (TradingView standard)
        position_value = entry_price * position_qty
        final_pnl_pct = (final_pnl / position_value) * 100 if position_value != 0 else 0
        
        # Add final trade to trades list
        trades.append({
            'type': 'final_exit',
            'entry': entry_price,
            'exit': exit_price,
            'qty': position_qty,
            'pnl_currency': final_pnl,
            'pnl_pct': final_pnl_pct,
            'commission_entry': commission_entry,
            'commission_exit': commission_exit,
            'order_size': order_size,
            'timestamp': df.index[-1]
        })
        
        # Update final equity
        df.loc[df.index[-1], 'equity'] = df.loc[df.index[-1], 'equity'] + final_pnl
    
    final_equity = df['equity'].iloc[-1] if 'equity' in df.columns and not df.empty else initial_capital

    # Calculate required metrics
    total_days = (df.index[-1] - df.index[0]).days
    weeks_tested = total_days / 7.0
    avg_trades_per_week = len(trades) / weeks_tested if weeks_tested > 0 else 0

    # Profit calculation
    profit = final_equity - initial_capital
    profit_pct = (profit / initial_capital) * 100
    
    # Expected Payoff (average profit per trade)
    expected_payoff = profit / len(trades) if len(trades) > 0 else 0
    
    # Profit Factor (gross profit / gross loss)
    winning_trades = [t for t in trades if t['pnl_currency'] > 0]
    losing_trades = [t for t in trades if t['pnl_currency'] < 0]
    gross_profit = sum(t['pnl_currency'] for t in winning_trades)
    gross_loss = abs(sum(t['pnl_currency'] for t in losing_trades))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0
    
    # Calculate max equity drawdown in currency terms first (needed for Recovery Factor)
    if 'equity' in df.columns and not df.empty:
        equity_curve = df['equity'].values
        running_max = pd.Series(equity_curve).cummax().values
        drawdowns_currency = running_max - equity_curve  # Absolute drawdown in currency
        max_equity_drawdown = drawdowns_currency.max()  # Maximum drawdown in currency
        equity_dd_pct = (max_equity_drawdown / initial_capital) * 100  # As percentage of initial capital
    else:
        max_equity_drawdown = 0.0
        equity_dd_pct = 0.0
    
    # Recovery Factor (TradingView methodology: Net Profit / Maximum Drawdown)
    recovery_factor = profit / max_equity_drawdown if max_equity_drawdown > 0 else (float('inf') if profit > 0 else 0)
    
    # Sharpe Ratio calculation (TradingView methodology)
    # Formula: SR = (MR - RFR) / SD
    # Where MR = average return for monthly trading period, RFR = risk-free rate (2% annually), SD = standard deviation
    if 'equity' in df.columns and len(df) > 1:
        # Resample equity to monthly periods (end of month values)
        monthly_equity = df['equity'].resample('ME').last().dropna()  # Fixed: changed 'M' to 'ME' to avoid deprecation warning
        
        if len(monthly_equity) >= 2:
            # Calculate monthly returns
            monthly_returns = monthly_equity.pct_change().dropna()
            
            if len(monthly_returns) > 0 and monthly_returns.std() > 0:
                # TradingView default: 2% annual risk-free rate
                annual_risk_free_rate = 0.02
                monthly_risk_free_rate = annual_risk_free_rate / 12  # Convert to monthly
                
                # Calculate Sharpe ratio components
                average_monthly_return = monthly_returns.mean()  # MR
                monthly_std_deviation = monthly_returns.std()    # SD
                
                # Sharpe Ratio = (MR - RFR) / SD
                sharpe_ratio = (average_monthly_return - monthly_risk_free_rate) / monthly_std_deviation
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0
    else:
        sharpe_ratio = 0

    # Number of trades
    num_trades = len(trades)

    print (f"Backtest result:\n Final Equity: {final_equity} \n Max Drawdown: {equity_dd_pct:.2f}% \n Num trades: {num_trades}")
    print (f"Profit: {profit} ({profit_pct}%)")
    print (f"Weeks Tested: {weeks_tested}, Avg Trades/Week: {avg_trades_per_week}")
    print (f"Expected Payoff: {expected_payoff}, Profit Factor: {profit_factor}")
    print (f"Recovery Factor: {recovery_factor}, Sharpe Ratio: {sharpe_ratio}")
    
    if trades:
        print (trades[0])
    
    if equity_dd_pct > (0.18 * 100):
        return
    if final_equity <= initial_capital:
        return

    result_dir = "result/crypto"
    os.makedirs(result_dir, exist_ok=True)
    
    # Create params_str for filename and CSV content
    params_str = '_'.join(f"{k}_{v}" for k, v in params.items())
    result_filename = os.path.join(result_dir, f"{strategy.name}_results.csv".lower())

    # Thread-safe CSV writing
    with file_write_lock:
        # Check if file exists to determine if we need to write headers
        file_exists = os.path.exists(result_filename)
        
        with open(result_filename, mode='a', newline='') as csvfile:
            # Write headers if file doesn't exist
            if not file_exists:
                headers = ['Exchange', 'Trading_Pair', 'Timeframe'] + [
                    'Weeks_Tested', 'Avg_Trades_Per_Week', 'Profit', 
                    'Expected_Payoff', 'Profit_Factor', 'Recovery_Factor', 
                    'Sharpe_Ratio', 'Max Equity_DD_%', 'Trades'
                ] + list(params.keys())
                csvfile.write(','.join(headers) + '\n')
            
            # Write data row
            csv_content = f'{exchange},{trading_pair},{timeframe}'
            csv_content += f',{weeks_tested:.2f},{avg_trades_per_week:.2f},{profit:.2f}'
            csv_content += f',{expected_payoff:.2f},{profit_factor:.2f},{recovery_factor:.2f},{sharpe_ratio:.2f}'
            csv_content += f',{equity_dd_pct:.2f},{num_trades}'
            csv_content += ',' + ','.join(f"{v}" for k, v in params.items())
            csvfile.write(csv_content + "\n")
