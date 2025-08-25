#!/usr/bin/env python3
"""
Historical Candlestick Data Scraper for Bitget and Blofin Exchanges
Downloads 1-minute candlestick data for specified trading pairs
"""

import ccxt
import pandas as pd
import time
import os
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scraping.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ExchangeDataScraper:
    def __init__(self, exchange_name: str, config: Dict):
        """Initialize exchange connection"""
        self.exchange_name = exchange_name
        self.config = config
        
        # Initialize exchange
        exchange_class = getattr(ccxt, exchange_name)
        self.exchange = exchange_class({
            'apiKey': config.get('api_key'),
            'secret': config.get('secret'),
            'sandbox': config.get('sandbox', False),
            'enableRateLimit': True,
            'rateLimit': 1000,  # 1 second between requests
        })
        
        logger.info(f"Initialized {exchange_name} exchange")
    
    def get_exchange_info(self) -> Dict:
        """Get exchange information and supported symbols"""
        try:
            markets = self.exchange.load_markets()
            logger.info(f"Loaded {len(markets)} markets from {self.exchange_name}")
            return markets
        except Exception as e:
            logger.error(f"Error loading markets from {self.exchange_name}: {e}")
            return {}
    
    def fetch_ohlcv(self, symbol: str, timeframe: str = '1m', until: Optional[int] = None, limit: int = 1000) -> List:
        """Fetch OHLCV data for a symbol"""
        try:
            # Convert symbol format if needed
            ohlcv = self.exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                # since=since,
                limit=limit,
                params={
                    'until': until
                }
            )
            return ohlcv
        except Exception as e:
            logger.error(f"Error fetching OHLCV for {symbol} on {self.exchange_name}: {e}")
            return []
    
    def download_historical_data(self, symbol: str, start_date: datetime, end_date: datetime, 
                               timeframe: str = '1m') -> pd.DataFrame:
        """Download full historical data for a symbol"""
        logger.info(f"Starting download for {symbol} on {self.exchange_name} from {end_date} to {start_date}")
        
        # Convert dates to timestamps
        start_timestamp = int(start_date.timestamp() * 1000)
        since = int(end_date.timestamp() * 1000)
        
        all_data = []
        current_since = since
        
        while current_since > start_timestamp:
            try:
                print (datetime.fromtimestamp(current_since/1000))
                # Fetch data
                ohlcv_data = self.fetch_ohlcv(symbol, timeframe, current_since, 1000)
                
                if not ohlcv_data:
                    logger.warning(f"No data received for {symbol} at {current_since}")
                    break
                
                all_data.extend(ohlcv_data)
                
                # Update timestamp for next request (going backward)
                if len(ohlcv_data) > 0:
                    current_since = ohlcv_data[0][0] - 1  # Previous timestamp
                else:
                    break
                
                # Rate limiting
                time.sleep(0.1)
                
                logger.info(f"Downloaded {len(all_data)} candles for {symbol} on {self.exchange_name}")
                
            except Exception as e:
                logger.error(f"Error during download for {symbol}: {e}")
                break
        
        if not all_data:
            logger.warning(f"No data downloaded for {symbol} on {self.exchange_name}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['symbol'] = symbol
        
        # Remove duplicates and sort (maintain chronological order)
        df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
        
        # print (df, start_date, end_date)
        # Filter to requested date range
        df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
        
        logger.info(f"Successfully downloaded {len(df)} candles for {symbol} on {self.exchange_name}")
        return df
    
    def save_to_csv(self, df: pd.DataFrame, symbol: str, output_dir: str = 'data'):
        """Append DataFrame to CSV file (or create if not exists)"""
        if df.empty:
            logger.warning(f"No data to save for {symbol}")
            return

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Format filename
        clean_symbol = symbol.replace("/", "_").replace(":USDT", "").lower()
        filename = f"{self.exchange_name}_{clean_symbol}.csv"
        filepath = os.path.join(output_dir, filename)

        # Append to CSV (or create if not exists)
        file_exists = os.path.isfile(filepath)
        df.to_csv(filepath, mode='a', header=not file_exists, index=False)
        logger.info(f"Appended {len(df)} records to {filepath}")

        return filepath

def main():
    """Main function to run the scraper"""
    parser = argparse.ArgumentParser(description='Download historical candlestick data from exchanges')
    parser.add_argument('--start-date', type=str, default='2025-08-01', 
                       help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end-date', type=str, default='2025-08-18', 
                       help='End date in YYYY-MM-DD format')
    parser.add_argument('--timeframe', type=str, default='1m', 
                       help='Timeframe for data (1m, 5m, 1h, etc.)')
    parser.add_argument('--output-dir', type=str, default='data', 
                       help='Output directory for CSV files')
    parser.add_argument('--exchanges', nargs='+', default=['bitget', 'blofin'], 
                       help='Exchanges to scrape from')
    
    args = parser.parse_args()
    
    # Parse dates
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    
    # Trading pairs to download (based on config.py)
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
    
    # Exchange configurations
    exchange_configs = {
        'bitget': {
            'name': 'bitget',
            'api_key': None,
            'secret': None,
            'sandbox': False,
            'enabled': True,
        },
        'blofin': {
            'name': 'blofin',
            'api_key': None,
            'secret': None,
            'sandbox': False,
            'enabled': True,
        }
    }
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each exchange
    for exchange_name in args.exchanges:
        if exchange_name not in exchange_configs:
            logger.warning(f"Exchange {exchange_name} not configured, skipping...")
            continue
        
        config = exchange_configs[exchange_name]
        if not config.get('enabled', True):
            logger.info(f"Exchange {exchange_name} is disabled, skipping...")
            continue
        
        try:
            # Initialize scraper
            scraper = ExchangeDataScraper(exchange_name, config)
            
            # Test connection
            markets = scraper.get_exchange_info()
            if not markets:
                logger.error(f"Could not connect to {exchange_name}, skipping...")
                continue
            
            # Download data for each trading pair
            for symbol in trading_pairs:
                try:
                    # Check if symbol exists on this exchange
                    if symbol not in markets.keys():
                        logger.warning(f"Symbol {symbol} not available on {exchange_name}, skipping...")
                        continue

                    df = scraper.download_historical_data(
                        symbol=symbol,
                        start_date=start_date,
                        end_date=end_date,
                        timeframe=args.timeframe
                    )
                    
                    # Save to CSV
                    if not df.empty:
                        scraper.save_to_csv(df, symbol, args.output_dir)
                    
                    # Rate limiting between symbols
                    time.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Error processing {symbol} on {exchange_name}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error initializing {exchange_name}: {e}")
            continue
    
    logger.info("Data scraping completed!")

if __name__ == "__main__":
    main()
