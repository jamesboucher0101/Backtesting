from src.strategies.RSI import RSIStrategy
from src.strategies.MACD import MACDStrategy
from src.strategies.MovingAvg import MovingAvgStrategy
from src.strategies.MovingAvg2Line import MovingAvg2LineStrategy
from src.backtest import simulate_backtest
import multiprocessing as mp
import itertools
import os


def run_backtest_worker(args):
    """Worker function to run a single backtest"""
    strategy, exchange, trading_pair, timeframe = args
    try:
        simulate_backtest(strategy, exchange, trading_pair, timeframe)
        return f"Success: {strategy.name} on {exchange} {trading_pair} {timeframe}"
    except Exception as e:
        return f"Error: {strategy.name} on {exchange} {trading_pair} {timeframe} - {str(e)}"


def main():
    exchanges = [
        'blofin',
        # 'bitget'
    ]
    trading_pairs = [
        'FARTCOIN/USDT',
        'AERO/USDT',
        'PEPE/USDT',
        'POPCAT/USDT',
        'BTC/USDT',
        'ETH/USDT',
        'SOL/USDT',
        'DOGE/USDT',
        'GOAT/USDT',
        'SUI/USDT'
    ]
    timeframes = [
        '5m',
        # '10m',
        # '15m',
        # '30m',
        # '1h',
        # '2h'
    ]

    strategies = [
        RSIStrategy(),
        # MACDStrategy(),
        # MovingAvgStrategy(),
        # MovingAvg2LineStrategy()
    ]

    # Get number of CPU cores
    num_cores = mp.cpu_count()
    print(f"Using {num_cores} CPU cores for parallel processing")

    while True:
        # Generate all combinations of parameters
        combinations = list(itertools.product(strategies, exchanges, trading_pairs, timeframes))
        
        print(f"Running {len(combinations)} backtests in parallel...")
        
        # Create a process pool and run backtests in parallel
        with mp.Pool(processes=num_cores) as pool:
            results = pool.map(run_backtest_worker, combinations)
        
        # Print results
        for result in results:
            print(result)
        
        print("Completed all backtests. Starting next iteration...")
    
if __name__ == '__main__':
    main()