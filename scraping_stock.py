#!/usr/bin/env python3
"""
Script to split combined stock data CSV into separate symbol-named CSV files.
Only saves timestamp, open, high, low, close, volume fields with ISO-format timestamps.
"""

import pandas as pd
import os
from datetime import datetime
import sys

def split_stock_data(input_file, output_dir):
    """
    Split combined stock data CSV into separate files by symbol.
    
    Args:
        input_file (str): Path to the combined CSV file
        output_dir (str): Directory to save the split CSV files
    """
    
    print(f"Reading data from {input_file}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Check which files already exist to avoid reprocessing
    existing_files = set()
    if os.path.exists(output_dir):
        existing_files = {f.replace('.csv', '') for f in os.listdir(output_dir) if f.endswith('.csv')}
        if existing_files:
            print(f"Found existing files for symbols: {existing_files}")
    
    # Read the CSV file in chunks to handle large files efficiently
    chunk_size = 50000  # Large chunk size for better performance
    symbol_writers = {}
    processed_symbols = set()
    
    try:
        for chunk_num, chunk in enumerate(pd.read_csv(input_file, chunksize=chunk_size)):
            print(f"Processing chunk {chunk_num + 1}...")
            
            # Filter to only the required columns
            # ts_event is already in ISO format, just need to rename it to timestamp
            filtered_chunk = chunk[['ts_event', 'open', 'high', 'low', 'close', 'volume', 'symbol']].copy()
            filtered_chunk.rename(columns={'ts_event': 'timestamp'}, inplace=True)
            
            # Process each symbol in this chunk
            for symbol, group in filtered_chunk.groupby('symbol'):
                # Skip if we already processed this symbol
                if symbol in existing_files:
                    continue
                    
                # Clean symbol name for filename (replace problematic characters)
                clean_symbol = symbol.replace('/', '_').replace('\\', '_').replace(':', '_')
                output_file = os.path.join(output_dir, f"{clean_symbol}.csv")
                
                # Initialize writer for new symbols
                if symbol not in symbol_writers:
                    symbol_writers[symbol] = open(output_file, 'w')
                    # Write header
                    symbol_writers[symbol].write("timestamp,open,high,low,close,volume\n")
                    processed_symbols.add(symbol)
                    print(f"Started processing symbol: {symbol}")
                
                # Write data for this symbol
                group_data = group.drop('symbol', axis=1)
                group_data.to_csv(symbol_writers[symbol], index=False, header=False)
        
        # Close all file writers and sort data
        for symbol, writer in symbol_writers.items():
            writer.close()
            clean_symbol = symbol.replace('/', '_').replace('\\', '_').replace(':', '_')
            output_file = os.path.join(output_dir, f"{clean_symbol}.csv")
            
            # Sort the file by timestamp
            df = pd.read_csv(output_file)
            df.sort_values('timestamp', inplace=True)
            df.to_csv(output_file, index=False)
            
            print(f"Completed processing {symbol}: {len(df)} records saved to {output_file}")
        
        print(f"\nSplit complete! Processed {len(processed_symbols)} new symbols.")
        print(f"All files saved to {output_dir}")
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        # Close any open file handles
        for writer in symbol_writers.values():
            try:
                writer.close()
            except:
                pass
        sys.exit(1)

def get_symbol_count(input_file):
    """
    Get a count of unique symbols in the file for preview.
    
    Args:
        input_file (str): Path to the combined CSV file
    
    Returns:
        int: Number of unique symbols
    """
    try:
        # Read just a sample to get symbol list
        sample_size = 100000
        df_sample = pd.read_csv(input_file, nrows=sample_size)
        unique_symbols = df_sample['symbol'].nunique()
        print(f"Sample analysis: Found at least {unique_symbols} unique symbols in first {sample_size} rows")
        return unique_symbols
    except Exception as e:
        print(f"Could not analyze symbols: {str(e)}")
        return 0

def verify_split_files(output_dir):
    """
    Verify the split files and show summary statistics.
    
    Args:
        output_dir (str): Directory containing the split CSV files
    """
    if not os.path.exists(output_dir):
        print(f"Output directory {output_dir} does not exist.")
        return
    
    csv_files = [f for f in os.listdir(output_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in {output_dir}")
        return
    
    print(f"\nFound {len(csv_files)} split files:")
    print("-" * 50)
    
    total_records = 0
    for csv_file in sorted(csv_files):
        file_path = os.path.join(output_dir, csv_file)
        try:
            df = pd.read_csv(file_path)
            records = len(df)
            total_records += records
            symbol = csv_file.replace('.csv', '')
            print(f"{symbol:<15} : {records:>8,} records")
        except Exception as e:
            print(f"{csv_file:<15} : Error reading file - {str(e)}")
    
    print("-" * 50)
    print(f"{'Total':<15} : {total_records:>8,} records")

def main():
    # Configuration
    input_file = "/home/ubuntu/project/stock_data/xnas-itch-20220101-20250819.ohlcv-1m.csv"
    output_dir = "/home/ubuntu/project/stock_data/split"
    
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found!")
        sys.exit(1)
    
    print("Stock Data Splitter")
    print("=" * 60)
    print(f"Input file: {input_file}")
    print(f"Output directory: {output_dir}")
    print("Columns to save: timestamp, open, high, low, close, volume")
    print("=" * 60)
    
    # Check if split files already exist
    if os.path.exists(output_dir):
        csv_files = [f for f in os.listdir(output_dir) if f.endswith('.csv')]
        if csv_files:
            print(f"Found {len(csv_files)} existing split files.")
            print("Choose an option:")
            print("1. Skip splitting and show verification summary")
            print("2. Re-split (will skip existing files)")
            print("3. Force re-split (will overwrite existing files)")
            
            try:
                choice = input("Enter choice (1-3): ").strip()
                
                if choice == "1":
                    verify_split_files(output_dir)
                    return
                elif choice == "3":
                    # Remove existing files for fresh split
                    for csv_file in csv_files:
                        os.remove(os.path.join(output_dir, csv_file))
                    print("Removed existing files for fresh split.")
                # choice == "2" or any other input continues with normal processing
                    
            except KeyboardInterrupt:
                print("\nOperation cancelled.")
                return
    
    # Preview symbol count
    get_symbol_count(input_file)
    print("=" * 60)
    
    # Perform the split
    split_stock_data(input_file, output_dir)
    
    # Show verification summary
    print("\nVerification Summary:")
    verify_split_files(output_dir)

if __name__ == "__main__":
    main()
