import os


BASE_DATA_DIR = "xgboost/data"
RAW_DATA_PATH = os.path.join(BASE_DATA_DIR, "stock_price")
GRAPH_DIR = os.path.join(BASE_DATA_DIR, "graphs")
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