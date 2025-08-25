import pandas as pd
import datetime
from pathlib import Path
from typing import List, Dict

# CME month codes
MONTH_CODES = {
    3: "H",   # March
    6: "M",   # June
    9: "U",   # September
    12: "Z",  # December
}

def get_symbol(contract_prefix: str, date: datetime.date) -> str:
    """
    Given a date and contract prefix, return the active futures contract symbol
    (approximation of TradingView's continuous contract logic).
    
    Args:
        contract_prefix: The contract prefix (MES, MGC, MNQ, MYM)
        date: The date to get the symbol for
    """
    year = date.year % 10  # last digit for symbol
    month = date.month

    # Determine which contract month we're in
    if month <= 3:
        contract_month = 3
    elif month <= 6:
        contract_month = 6
    elif month <= 9:
        contract_month = 9
    else:
        contract_month = 12

    # Expiration date = 3rd Friday of contract_month
    first_day = datetime.date(date.year, contract_month, 1)
    fridays = [first_day + datetime.timedelta(days=i) 
               for i in range(31) if (first_day + datetime.timedelta(days=i)).weekday() == 4 and (first_day + datetime.timedelta(days=i)).month == contract_month]
    expiration = fridays[2]  # 3rd Friday

    # Approx rollover = 1 week before expiration
    rollover = expiration - datetime.timedelta(days=7)

    # If date >= rollover, we should move to next contract
    if date >= rollover:
        if contract_month == 12:
            contract_month = 3
            year = (date.year + 1) % 10
        else:
            contract_month += 3
    
    return f"{contract_prefix}{MONTH_CODES[contract_month]}{year}"

def load_and_filter_csv_files(data_directory: str = "future_data/original") -> Dict[str, pd.DataFrame]:
    """
    Load CSV files (MES.csv, MGC.csv, MNQ.csv, MYM.csv) and filter records
    where the symbol column matches the get_symbol function result for the timestamp.
    
    Args:
        data_directory: Directory containing the CSV files
        
    Returns:
        Dictionary with filename as key and filtered DataFrame as value
    """
    csv_files = ["MES.csv", "MGC.csv", "MNQ.csv", "MYM.csv"]
    data_dir = Path(data_directory)
    filtered_data = {}
    
    for csv_file in csv_files:
        file_path = data_dir / csv_file
        
        if not file_path.exists():
            print(f"Warning: {csv_file} not found in {data_directory}")
            continue
            
        print(f"Loading {csv_file}...")
        
        try:
            # Load CSV file
            df = pd.read_csv(file_path)
            
            # Convert timestamp to datetime
            df['ts_event'] = pd.to_datetime(df['ts_event'])
            df['date'] = df['ts_event'].dt.date
            
            # Extract contract prefix from filename (MES, MGC, MNQ, MYM)
            contract_prefix = csv_file.replace(".csv", "")
            
            # Apply get_symbol function to each date with the contract prefix
            df['expected_symbol'] = df['date'].apply(lambda date: get_symbol(contract_prefix, date))
            # Filter to only records where the year is 2023 or later
            df = df[df['ts_event'].dt.year >= 2023]
            # Filter records where symbol matches expected symbol
            filtered_df = df[df['symbol'] == df['expected_symbol']].copy()
            
            # Drop helper columns
            filtered_df = filtered_df.drop(['date', 'expected_symbol'], axis=1)
            
            print(f"  Original records: {len(df):,}")
            print(f"  Filtered records: {len(filtered_df):,}")
            print(f"  Filtered percentage: {len(filtered_df)/len(df)*100:.2f}%")
            
            filtered_data[csv_file] = filtered_df
            
        except Exception as e:
            print(f"Error processing {csv_file}: {str(e)}")
            continue
    
    return filtered_data

def save_filtered_data(filtered_data: Dict[str, pd.DataFrame], output_directory: str = "future_data"):
    """
    Save filtered data to CSV files in the output directory.
    
    Args:
        filtered_data: Dictionary with filename as key and filtered DataFrame as value
        output_directory: Directory to save filtered CSV files
    """
    output_dir = Path(output_directory)
    output_dir.mkdir(exist_ok=True)
    
    for filename, df in filtered_data.items():
        output_path = output_dir / f"{filename}"
        df.to_csv(output_path, index=False)
        print(f"Saved filtered data to {output_path}")

if __name__ == "__main__":
    # Load and filter CSV files
    filtered_data = load_and_filter_csv_files()
    
    # Save filtered data
    save_filtered_data(filtered_data)
    
    print("\nFiltering completed!")
