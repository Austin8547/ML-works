import pandas as pd
import config

def clean_data(df):
    # Assignment is necessary for drop
    df = df.drop(index=[0, 1], errors='ignore')
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Standardize Date
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    
    # Convert numeric columns
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Rounding and dropping NaNs
    df = df.dropna().reset_index(drop=True)
    df.loc[:, df.columns != 'Date'] = df.loc[:, df.columns != 'Date'].round(2)
    return df

def create_features(df, lags=config.LAG_DAYS):
    df = df.copy()
    target = config.TARGET_COL # 'Close'

    # 1. Lags
    for lag in lags:
        df[f'Close_lag{lag}'] = df[target].shift(lag)
    
    # 2. Moving Averages (The missing ones!)
    df['SMA_5'] = df[target].rolling(window=5).mean()
    df['EMA_20'] = df[target].ewm(span=20, adjust=False).mean()
    
    # 3. MACD
    df['EMA_12'] = df[target].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df[target].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # Drop rows with NaNs from calculations
    return df.dropna().reset_index(drop=True)

def process_data(df):
    df = clean_data(df)
    df = create_features(df)
    return df