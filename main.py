import os
import csv
import datetime
import pandas as pd
import json
from src.strategies.RSI import RSIStrategy
from src.strategies.MACD import MACDStrategy

def load_data(exchange, trading_pair, timeframe):
    csv_path = f'data/{exchange}_{trading_pair.replace("/", "_")}.csv'.lower()
    df = pd.read_csv(csv_path)

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        df = df.set_index('timestamp')
    else:
        raise ValueError("CSV must contain a 'timestamp' column.")

    def timeframe_to_pandas_rule(tf):
        if tf.endswith('m'):
            return f"{int(tf[:-1])}T"
        elif tf.endswith('h'):
            return f"{int(tf[:-1])}H"
        elif tf.endswith('d'):
            return f"{int(tf[:-1])}D"
        else:
            raise ValueError(f"Unsupported timeframe: {tf}")

    resample_rule = timeframe_to_pandas_rule(timeframe)

    ohlc_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    resampled_df = df.resample(resample_rule).agg(ohlc_dict).dropna()    
    return resampled_df

def simulate_backtest(strategy, exchange, trading_pair, timeframe, initial_capital=10000, commission_rate=0.0006, position_fraction=0.10):
    df = load_data(exchange, trading_pair, timeframe)
    
    params = strategy.suggest_parameters()
    print (params)

    df = strategy.prepare(df.copy(), params)
    df['position'] = 0
    df['equity'] = initial_capital
    df['trades'] = 0
    
    position = 0
    entry_price = 0
    position_qty = 0.0
    trades = []
    length = len(df)

    for i in range(0, length - 1):
        current_price = df['open'].iloc[i+1]
        desired_position = strategy.decide_position(df, i, position, params)
        equity_prev = df['equity'].iloc[i-1] if i > 0 else df['equity'].iloc[0]
        equity_curr = equity_prev

        # If changing position, close existing one first
        if position > 0 and desired_position <= 0:
            exit_price = current_price
            qty = position_qty
            commission_entry = entry_price * qty * commission_rate
            commission_exit = exit_price * qty * commission_rate
            pnl_currency = (exit_price - entry_price) * qty - commission_entry - commission_exit
            pnl_pct = (pnl_currency / equity_prev) * 100 if equity_prev != 0 else 0
            trades.append({
                'type': 'long_exit',
                'entry': entry_price,
                'exit': exit_price,
                'qty': qty,
                'pnl_currency': pnl_currency,
                'pnl_pct': pnl_pct,
                'commission_entry': commission_entry,
                'commission_exit': commission_exit,
                'size_fraction': position_fraction,
                'timestamp': df.index[i]
            })
            equity_curr = equity_prev + pnl_currency
            entry_price = 0
            position_qty = 0.0
        elif position < 0 and desired_position >= 0:
            exit_price = current_price
            qty = position_qty
            commission_entry = entry_price * qty * commission_rate
            commission_exit = exit_price * qty * commission_rate
            pnl_currency = (entry_price - exit_price) * qty - commission_entry - commission_exit
            pnl_pct = (pnl_currency / equity_prev) * 100 if equity_prev != 0 else 0
            trades.append({
                'type': 'short_exit',
                'entry': entry_price,
                'exit': exit_price,
                'qty': qty,
                'pnl_currency': pnl_currency,
                'pnl_pct': pnl_pct,
                'commission_entry': commission_entry,
                'commission_exit': commission_exit,
                'size_fraction': position_fraction,
                'timestamp': df.index[i]
            })
            equity_curr = equity_prev + pnl_currency
            entry_price = 0
            position_qty = 0.0

        if desired_position != position:
            if desired_position != 0:
                entry_price = current_price
                position_qty = (equity_curr * position_fraction * (1 - commission_rate)) / entry_price if entry_price != 0 else 0.0
            position = desired_position

        df.loc[df.index[i], 'equity'] = equity_curr
        df.loc[df.index[i], 'position'] = position
        df.loc[df.index[i], 'trades'] = len(trades)

    initial_capital = 10000
    final_equity = df['equity'].iloc[-2] if 'equity' in df.columns and not df.empty else initial_capital
    
    # Prepare result directory and filename
    result_dir = "result"
    os.makedirs(result_dir, exist_ok=True)
    result_filename = os.path.join(result_dir, f"blofin_{trading_pair.replace('/', '_')}_{timeframe}_params_{strategy.name()}.csv".lower())

    # Calculate max equity drawdown
    # Max equity drawdown: greatest % loss from any peak to subsequent trough
    if 'equity' in df.columns and not df.empty:
        equity_curve = df['equity'].values
        running_max = pd.Series(equity_curve).cummax().values
        drawdowns = (equity_curve - running_max) / running_max
        max_drawdown = drawdowns.min()  # This will be negative; greatest loss
    else:
        max_drawdown = 0.0

    if (abs(max_drawdown) > 0.18):
        return

    # Write trades to CSV in append mode
    with open(result_filename, mode='a', newline='') as csvfile:
        params_str = ','.join(f"{v}" for k, v in params.items())
        params_str += ',' + str(final_equity) + "," + str(len(trades)) + "," + str(max_drawdown)
        csvfile.write(params_str + "\n")


def main():
    exchange = 'blofin'
    trading_pair = 'FARTCOIN/USDT'
    timeframes = ['5m', '15m']

    # You can choose which strategy to use:
    # strategy = RSIStrategy()  # Original RSI strategy
    strategy = MACDStrategy()   # New MACD strategy

    while True:
        for timeframe in timeframes:
            try:
                simulate_backtest(strategy, exchange, trading_pair, timeframe)
            except Exception as e:
                print(e)
                time.sleep(1)
    
main()