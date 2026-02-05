import pandas as pd
import numpy as np
import joblib
import os
from datetime import timedelta
import config

def generate_forecast(df, features):
    """
    Performs a 60-day recursive forecast.
    """
    # 1. Load Artifacts
    model_path = os.path.join(config.MODEL_DIR, "xgboost_stock_model.pkl")
    scaler_path = os.path.join(config.MODEL_DIR, "scaler.pkl")
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # 2. Setup initial state (The last available window of data)
    # We need enough history to calculate indicators for the next 60 days
    current_data = df.copy().tail(100) 
    forecasts = []
    
    last_date = pd.to_datetime(df['Date'].max())
    
    print(f"\n--- Starting 60-Day Recursive Forecast for {config.TICKER} ---")

    for i in range(config.FORECAST_DAYS):
        # Extract the current features for the most recent date
        # We only need the very last row to predict the next day
        X_latest = current_data[features].tail(1)
        
        # Scale
        X_scaled = scaler.transform(X_latest)
        
        # Predict
        pred_price = model.predict(X_scaled)[0]
        
        # Determine next date (skipping weekends is optional, but let's keep it simple)
        next_date = last_date + timedelta(days=1)
        if next_date.weekday() >= 5:  # Skip Saturday/Sunday
             next_date += timedelta(days=2 if next_date.weekday() == 5 else 1)
        
        # 3. Update the dataframe with the new prediction
        # We create a new row with the prediction as the 'Close' price
        new_row = {
            'Date': next_date,
            'Close': pred_price,
            'Open': pred_price, # We assume open is roughly last close for the loop
            'High': pred_price,
            'Low': pred_price,
            'Volume': current_data['Volume'].mean() # Keep volume neutral
        }
        
        current_data = pd.concat([current_data, pd.DataFrame([new_row])], ignore_index=True)
        
        # 4. RECALCULATE INDICATORS (Crucial Step)
        # The model needs updated SMA, MACD, and Lags based on our NEW predicted price
        current_data['Close_lag1'] = current_data['Close'].shift(1)
        current_data['Close_lag2'] = current_data['Close'].shift(2)
        current_data['Close_lag3'] = current_data['Close'].shift(3)
        current_data['Close_lag5'] = current_data['Close'].shift(5)
        
        current_data['SMA_5'] = current_data['Close'].rolling(window=5).mean()
        # Note: For speed in the loop, we use simple SMA for EMA/MACD proxies or 
        # you can call your full processor function here if it's fast.
        
        # Record result
        forecasts.append({'Date': next_date, 'Predicted_Close': pred_price})
        last_date = next_date

    forecast_df = pd.DataFrame(forecasts)
    return forecast_df

def plot_forecast(historical_df, forecast_df):
    """
    Visualizes the historical trend plus the 60-day prediction.
    """
    import matplotlib.pyplot as plt
    plt.style.use('dark_background')
    plt.figure(figsize=(12, 6))
    
    # Plot last 90 days of history
    hist = historical_df.tail(90)
    plt.plot(pd.to_datetime(hist['Date']), hist['Close'], label='Historical Price', color='#00FFD1')
    
    # Plot Forecast
    plt.plot(pd.to_datetime(forecast_df['Date']), forecast_df['Predicted_Close'], 
             label='60-Day Forecast', color='#FF007A', linestyle='--')
    
    plt.title(f"{config.TICKER} - 60 Day Price Prediction", fontsize=16)
    plt.xlabel("Date")
    plt.ylabel("Price (INR)")
    plt.legend()
    plt.grid(True, alpha=0.1)
    
    save_path = os.path.join(config.GRAPH_DIR, "9_final_forecast.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Forecast chart saved to {save_path}")