from src.strategies.RSI import RSIStrategy
from src.strategies.MACD import MACDStrategy
from src.strategies.MovingAvg import MovingAvgStrategy
from src.strategies.MovingAvg2Line import MovingAvg2LineStrategy
from src.strategies.KeltnerChannels import KeltnerChannelsStrategy
from src.backtest_crypto import simulate_backtest_crypto
import multiprocessing as mp
import itertools
import os

# Global variables for progress tracking
simulation_counter = None
total_simulations = None
counter_lock = None


def run_backtest_worker(args):
    """Worker function to run a single backtest with single parameter set"""
    global simulation_counter, total_simulations, counter_lock
    strategy, exchange, trading_pair, timeframe, single_params = args
    try:
        # You can now choose position sizing method:
        # simulate_backtest(strategy, exchange, trading_pair, timeframe, single_params, 
        #                  position_sizing_method='risk_based', risk_percentage=0.02)
        simulate_backtest_crypto(strategy, exchange, trading_pair, timeframe, single_params)
        
        # Update counter and display progress
        with counter_lock:
            simulation_counter.value += 1
            current_count = simulation_counter.value
            total_count = total_simulations.value
            progress_pct = (current_count / total_count) * 100 if total_count > 0 else 0
            print(f"Simulation {current_count}/{total_count} ({progress_pct:.1f}%) - {strategy.name} on {exchange} {trading_pair} {timeframe}")
        
        return f"Success: {strategy.name} on {exchange} {trading_pair} {timeframe} - params: {single_params}"
    except Exception as e:
        # Still update counter even on error
        with counter_lock:
            simulation_counter.value += 1
            current_count = simulation_counter.value
            total_count = total_simulations.value
            progress_pct = (current_count / total_count) * 100 if total_count > 0 else 0
            print(f"Simulation {current_count}/{total_count} ({progress_pct:.1f}%) - ERROR: {strategy.name} on {exchange} {trading_pair} {timeframe}")
        
        return f"Error: {strategy.name} on {exchange} {trading_pair} {timeframe} - params: {single_params} - {str(e)}"


def main():
    global simulation_counter, total_simulations, counter_lock
    
    exchanges = [
        # 'blofin',
        # 'bitget'
    ]
    trading_pairs = [
        # 'FARTCOIN/USDT',
        # 'AERO/USDT',
        # 'PEPE/USDT',
        # 'POPCAT/USDT',
        # 'BTC/USDT',
        # 'ETH/USDT',
        # 'SOL/USDT',
        # 'DOGE/USDT',
        # 'GOAT/USDT',
        # 'SUI/USDT'
    ]
    timeframes = [
        # '5m',
        # '10m',
        '15m',
        # '30m',
        # '1h',
        # '2h'
    ]

    strategies = [
        RSIStrategy(),
        # MACDStrategy(),
        # KeltnerChannelsStrategy(),
        # MovingAvgStrategy(),
        # MovingAvg2LineStrategy()
    ]

    # Get number of CPU cores
    num_cores = mp.cpu_count()
    print(f"Using {num_cores} CPU cores for parallel processing")

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
    
    # Initialize shared counter for progress tracking
    simulation_counter = mp.Value('i', 0)  # 'i' for integer
    total_simulations = mp.Value('i', len(combinations))
    counter_lock = mp.Lock()
    
    print(f"Starting {len(combinations)} simulations...")
    
    # Create a process pool and run backtests in parallel
    with mp.Pool(processes=num_cores) as pool:
        results = pool.map(run_backtest_worker, combinations)
    
    # Print results summary
    success_count = sum(1 for result in results if result.startswith("Success"))
    error_count = len(results) - success_count
    print(f"\n=== BATCH COMPLETED ===")
    print(f"Total simulations: {len(results)}")
    print(f"Successful: {success_count}")
    print(f"Errors: {error_count}")
    print("Starting next iteration...\n")
    
if __name__ == '__main__':
    main()