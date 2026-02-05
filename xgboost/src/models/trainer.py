import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import joblib
import os
import config

def train_model(df):
    """
    Pure training function: Scales data, fits XGBoost on the full 
    dataset, and saves the artifacts.
    """
    # 1. Feature Selection
    features = [
        'Open', 'High', 'Low', 'Volume', 
        'Close_lag1', 'Close_lag2', 'Close_lag3', 'Close_lag5',
        'SMA_5', 'EMA_20', 'MACD', 'MACD_Signal', 'MACD_Hist'
    ]
    
    X = df[features]
    y = df[config.TARGET_COL]

    # 2. Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save the scaler for future use in forecasting
    scaler_path = os.path.join(config.MODEL_DIR, "scaler.pkl")
    joblib.dump(scaler, scaler_path)

    # 3. Model Training
    # We train on the entire processed dataset here
    model = xgb.XGBRegressor(
        n_estimators=config.XGB_PARAMS.get('n_estimators', 100),
        learning_rate=config.XGB_PARAMS.get('learning_rate', 0.1),
        max_depth=config.XGB_PARAMS.get('max_depth', 6),
        random_state=42
    )
    
    print(f"Fitting XGBoost model on {len(X)} rows...")
    model.fit(X_scaled, y)

    # 4. Save the Model
    model_path = os.path.join(config.MODEL_DIR, "xgboost_stock_model.pkl")
    joblib.dump(model, model_path)
    
    print(f"Training Complete. Model and Scaler saved to {config.MODEL_DIR}")
    
    return model, scaler, features