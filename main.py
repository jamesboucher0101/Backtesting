from src.strategies.RSI import RSIStrategy
from src.strategies.MACD import MACDStrategy
from src.strategies.MovingAvg import MovingAvgStrategy
from src.strategies.MovingAvg2Line import MovingAvg2LineStrategy
from src.backtest import simulate_backtest
import multiprocessing as mp
import itertools
import os


def run_backtest_worker(args):
    """Worker function to run a single backtest with single parameter set"""
    strategy, exchange, trading_pair, timeframe, single_params = args
    try:
        simulate_backtest(strategy, exchange, trading_pair, timeframe, single_params)
        return f"Success: {strategy.name} on {exchange} {trading_pair} {timeframe} - params: {single_params}"
    except Exception as e:
        return f"Error: {strategy.name} on {exchange} {trading_pair} {timeframe} - params: {single_params} - {str(e)}"


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
        '10m',
        '15m',
        '30m',
        '1h',
        '2h'
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
        # Generate individual jobs for each parameter combination
        print("Generating individual backtest jobs...")
        
        combinations = []
        total_param_combinations = 0
        
        for strategy in strategies:
            param_combos = strategy.get_parameter_combinations()
            total_param_combinations += len(param_combos)
            print(f"{strategy.name}: {len(param_combos)} parameter combinations")
            
            # Create individual job for each parameter combination
            for single_params in param_combos:
                for exchange, trading_pair, timeframe in itertools.product(exchanges, trading_pairs, timeframes):
                    combinations.append((strategy, exchange, trading_pair, timeframe, single_params))
        
        print(f"Generated {len(combinations)} individual backtest jobs from {total_param_combinations} parameter combinations")
        print(f"Each job will run a single backtest with one parameter set")
        
        # Create a process pool and run backtests in parallel
        with mp.Pool(processes=num_cores) as pool:
            results = pool.map(run_backtest_worker, combinations)
        
        # Print results
        for result in results:
            print(result)
        
        print("Completed all backtests. Starting next iteration...")
    
if __name__ == '__main__':
    main()