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
        
        # 4. RECALCULATE INDICATORS (Crucial Step: Incremental Updates)
        # 4.1 Lags
        current_data.loc[current_data.index[-1], 'Close_lag1'] = current_data['Close'].iloc[-2]
        current_data.loc[current_data.index[-1], 'Close_lag2'] = current_data['Close'].iloc[-3]
        current_data.loc[current_data.index[-1], 'Close_lag3'] = current_data['Close'].iloc[-4]
        current_data.loc[current_data.index[-1], 'Close_lag5'] = current_data['Close'].iloc[-6]
        
        # 4.2 SMA (Simple Moving Average) - Re-calculate on the window
        # Efficiently calculate only the last value
        current_data.loc[current_data.index[-1], 'SMA_5'] = current_data['Close'].tail(5).mean()

        # 4.3 EMA (Exponential Moving Average) - Incremental Calculation
        # Formula: EMA_today = (Value_today * alpha) + (EMA_yesterday * (1 - alpha))
        # alpha = 2 / (span + 1)
        # Note: We use the *predicted* close for the calculation
        
        def calculate_next_ema(series_name, span, current_price):
            alpha = 2 / (span + 1)
            prev_ema = current_data[series_name].iloc[-2] # The value before the new row
            new_ema = (current_price * alpha) + (prev_ema * (1 - alpha))
            return new_ema

        current_data.loc[current_data.index[-1], 'EMA_20'] = calculate_next_ema('EMA_20', 20, pred_price)
        
        # 4.4 MACD Components
        # We need to maintain EMA_12 and EMA_26 to calculate MACD
        # These columns were added to processor.py to ensure they exist
        new_ema_12 = calculate_next_ema('EMA_12', 12, pred_price)
        new_ema_26 = calculate_next_ema('EMA_26', 26, pred_price)
        
        current_data.loc[current_data.index[-1], 'EMA_12'] = new_ema_12
        current_data.loc[current_data.index[-1], 'EMA_26'] = new_ema_26
        
        new_macd = new_ema_12 - new_ema_26
        current_data.loc[current_data.index[-1], 'MACD'] = new_macd
        
        # 4.5 MACD Signal (EMA of MACD)
        # This is an EMA of the MACD line itself
        prev_signal = current_data['MACD_Signal'].iloc[-2]
        alpha_signal = 2 / (9 + 1)
        new_signal = (new_macd * alpha_signal) + (prev_signal * (1 - alpha_signal))
        
        current_data.loc[current_data.index[-1], 'MACD_Signal'] = new_signal
        current_data.loc[current_data.index[-1], 'MACD_Hist'] = new_macd - new_signal
        
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
    
    plt.grid(True, alpha=0.1)
    
    save_path = os.path.join(config.FORECAST_GRAPH_DIR, "9_final_forecast.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Forecast chart saved to {save_path}")

def save_forecast_csv(forecast_df):
    """
    Saves the forecast data to a CSV file in the reports directory.
    """
    os.makedirs(config.REPORT_DIR, exist_ok=True)
    report_path = os.path.join(config.REPORT_DIR, f"{config.TICKER}_60_day_forecast.csv")
    forecast_df.to_csv(report_path, index=False)
    print(f"Forecast results saved to {report_path}")