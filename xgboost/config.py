import os


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE_DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_PATH = os.path.join(BASE_DATA_DIR, "stock_price")
GRAPH_DIR = os.path.join(BASE_DATA_DIR, "graphs")
EDA_GRAPH_DIR = os.path.join(GRAPH_DIR, "eda_graphs")
VAL_GRAPH_DIR = os.path.join(GRAPH_DIR, "validation_graphs")
FORECAST_GRAPH_DIR = os.path.join(GRAPH_DIR, "final_forecast")
MODEL_DIR = os.path.join(BASE_DATA_DIR, "saved_models")
REPORT_DIR = os.path.join(BASE_DATA_DIR, "reports")


TICKER = "RELIANCE.NS"
PERIOD = "5y"
INTERVAL = "1d"
FILE_NAME = f"{TICKER}_data.csv"
FULL_DATA_PATH = os.path.join(RAW_DATA_PATH, FILE_NAME)


TARGET_COL = "Close"
LAG_DAYS = [1, 2, 3, 5]


XGB_PARAMS = {
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'eval_metric': 'rmse',
    'early_stopping_rounds': 50,
    'random_state': 42
}


TSC_SPLITS = 5
FORECAST_DAYS = 60

# --- Dynamic Configuration ---

NIFTY50_TICKERS = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
    "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BEL", "BHARTIARTL",
    "COALINDIA", "DRREDDY", "EICHERMOT", "HCLTECH", "HDFCBANK",
    "HDFCLIFE", "HINDUNILVR", "ICICIBANK", "INFY", "ITC",
    "JSWSTEEL", "KOTAKBANK", "LT", "M&M", "MARUTI",
    "NTPC", "ONGC", "POWERGRID", "RELIANCE", "SBIN",
    "SBILIFE", "SHRIRAMFIN", "SUNPHARMA", "TCS", "TATACONSUM",
    "TATAMOTORS", "TATASTEEL", "TECHM", "TITAN", "TRENT",
    "ULTRACEMCO", "WIPRO"
]

def update_config(ticker_symbol):
    """
    Updates the global configuration variables for a new ticker.
    """
    global TICKER, FILE_NAME, FULL_DATA_PATH
    
    # Ensure .NS suffix for NSE stocks
    if not ticker_symbol.endswith(".NS") and not ticker_symbol.endswith(".BO"):
         ticker_symbol += ".NS"
         
    TICKER = ticker_symbol.upper()
    FILE_NAME = f"{TICKER}_data.csv"
    FULL_DATA_PATH = os.path.join(RAW_DATA_PATH, FILE_NAME)
    print(f"Configuration updated for: {TICKER}")