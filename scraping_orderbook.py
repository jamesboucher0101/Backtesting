import ccxt
import json
import os
from datetime import datetime

def load_orderbook_data():
    trading_pairs = [
        'AERO/USDT:USDT',
        'BTC/USDT:USDT',
        'FARTCOIN/USDT:USDT',
        'PEPE/USDT:USDT',
        'POPCAT/USDT:USDT',
        'ETH/USDT:USDT', 
        'SOL/USDT:USDT',
        'DOGE/USDT:USDT',
        'GOAT/USDT:USDT',
        'SUI/USDT:USDT'
    ]

    orderbooks = {}
    bitget = ccxt.bitget()
    blofin = ccxt.blofin()
    for symbol in trading_pairs:
        try:
            bitget_ob = bitget.fetch_order_book(symbol)
        except Exception as e:
            bitget_ob = None
        try:
            blofin_ob = blofin.fetch_order_book(symbol)
        except Exception as e:
            blofin_ob = None
        orderbooks[symbol] = {
            'bitget': bitget_ob,
            'blofin': blofin_ob
        }
    
    return orderbooks

def save_individual_orderbooks(orderbooks, base_dir='orderbook_data', max_levels=10):
    """Save each exchange-symbol combination to individual JSON files"""
    try:
        # Create directory if it doesn't exist
        os.makedirs(base_dir, exist_ok=True)
        
        timestamp = datetime.now().isoformat()
        saved_files = []
        
        for symbol, exchanges in orderbooks.items():
            # Clean symbol name for filename (replace / with _)
            clean_symbol = symbol.replace('/', '_').replace(":USDT", "").lower()
            
            for exchange_name, orderbook_data in exchanges.items():
                if orderbook_data:  # Only save if data exists
                    # Create filename: orderbook_exchange_symbol.json
                    filename = f"orderbook_{exchange_name}_{clean_symbol}.json"
                    filepath = os.path.join(base_dir, filename)
                    
                    # Extract only bids and asks data (limit to max_levels)
                    bids = orderbook_data.get('bids', [])[:max_levels]
                    asks = orderbook_data.get('asks', [])[:max_levels]
                    
                    # Try to read existing data
                    existing_data = {'bids': [], 'asks': []}
                    try:
                        with open(filepath, 'r') as f:
                            existing_data = json.load(f)
                            # Ensure it has the right structure
                            if not isinstance(existing_data, dict) or 'bids' not in existing_data:
                                existing_data = {'bids': [], 'asks': []}
                    except (FileNotFoundError, json.JSONDecodeError):
                        # File doesn't exist or is empty/invalid, start with empty structure
                        existing_data = {'bids': [], 'asks': []}
                    
                    
                    # INSERT_YOUR_CODE
                    # Remove overlap/duplication between new data (bids/asks) and existing data (bids/asks)
                    # Only append new bids/asks that are not already at the start of existing_data
                    def deduplicate_front(new_list, existing_list):
                        # Remove from new_list any items that are identical (in order) to the start of existing_list
                        overlap = 0
                        max_overlap = min(len(new_list), len(existing_list))
                        for i in range(max_overlap, 0, -1):
                            if new_list[-i:] == existing_list[:i]:
                                overlap = i
                                break
                        if overlap > 0:
                            return new_list[:-overlap]
                        else:
                            return new_list

                    deduplicated_bids = deduplicate_front(bids, existing_data['bids'])
                    deduplicated_asks = deduplicate_front(asks, existing_data['asks'])

                    # Check if there's any new data after deduplication
                    if not deduplicated_bids and not deduplicated_asks:
                        print(f"Skipping - all data is duplicate for {filename}")
                        continue
                    
                    # Extend (unpack/spread) deduplicated bids and asks arrays
                    existing_data['bids'] = deduplicated_bids + existing_data['bids']
                    existing_data['asks'] = deduplicated_asks + existing_data['asks']
                    
                    # Save back to file (compact format without indentation)
                    with open(filepath, 'w') as f:
                        json.dump(existing_data, f, separators=(',', ':'), default=str)
                    
                    saved_files.append(filename)
                    print(f"Data appended to {filename} (bids entries: {len(existing_data['bids'])}, asks entries: {len(existing_data['asks'])})")
        
        print(f"\nSaved {len(saved_files)} files to {base_dir}/ directory")
        return True
        
    except Exception as e:
        print(f"Error saving individual orderbooks: {e}")
        return False

if __name__ == "__main__":
    print("Loading orderbook data...")
    while True:
        data = load_orderbook_data()
        save_individual_orderbooks(data, max_levels=10)
