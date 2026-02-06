import pandas as pd
import os
import yfinance as yf
import config


def download_stock_data(ticker=None, period=None, interval=None):
    ticker = ticker or config.TICKER
    period = period or config.PERIOD
    interval = interval or config.INTERVAL

    print(f"Fetching data for {ticker}...")
    try:
        data = yf.download(ticker, period=period, interval=interval)
        
        if data.empty:
            print(f"Error: No data found for {ticker}.")
            return None
            
        # Initial reset index to make 'Date' a column
        data.reset_index(inplace=True)
        return data
        
    except Exception as e:
        print(f"An error occurred during download: {e}")
        return None


def save_data(df, path=None):
    if path is None:
        path = config.FULL_DATA_PATH
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Data saved successfully to {path}")


def load_local_data(path=None):
    """
    Loads data from a local CSV file.
    """
    if path is None:
        path = config.FULL_DATA_PATH

    if os.path.exists(path):
        print(f"Loading data from {path}")
        return pd.read_csv(path)
    else:
        print(f"File not found: {path}")
        return None
