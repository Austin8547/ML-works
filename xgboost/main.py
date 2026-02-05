import config
import os
from src.data_loader import loader
from src.data_processing import processor
from src.visualization import plotter
from src.models import trainer
from src.models import evaluate
from src.models import forecaster # Ensure this is imported

def run_visualization(df):
    print("\n--- Generating Modernized Seaborn Charts ---")
    try:
        plotter.plot_all_charts(df)
        print(f"Charts successfully saved to: {config.GRAPH_DIR}")
    except Exception as e:
        print(f"An error occurred during visualization: {e}")

def main():
    # 0. Infrastructure Check
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.GRAPH_DIR, exist_ok=True)
    
    print(f"--- Starting Pipeline for {config.TICKER} ---")
    
    # 1. Acquire Data
    df_raw = loader.download_stock_data()
    
    if df_raw is not None and not df_raw.empty:
        loader.save_data(df_raw)
        
        # 3. Process Data
        print("\n--- Starting Data Processing ---")
        df_processed = processor.process_data(df_raw)
        
        if not df_processed.empty:
            print(f"Processing Successful! Rows: {len(df_processed)}")
            
            # 4. EDA Visuals
            run_visualization(df_processed)
            
            # 5. Define ML Features
            features = [
                'Open', 'High', 'Low', 'Volume', 
                'Close_lag1', 'Close_lag2', 'Close_lag3', 'Close_lag5',
                'SMA_5', 'EMA_20', 'MACD', 'MACD_Signal', 'MACD_Hist'
            ]
            
            # 6. EVALUATION (Validation)
            evaluate.validate_performance(df_processed, features)
            
            # 7. TRAINING (Production Fit)
            model, scaler, feature_list = trainer.train_model(df_processed)
            
            # 8. POST-TRAIN VISUALS
            evaluate.plot_feature_importance(model, feature_list)
            
            # --- NEW: 9. FORECASTING (Predicting the Future) ---
            print("\n--- Generating 60-Day Prediction ---")
            forecast_df = forecaster.generate_forecast(df_processed, features)
            
            # 10. PLOT FINAL FORECAST
            forecaster.plot_forecast(df_processed, forecast_df)
            
            print("\n--- Pipeline Complete ---")
            print(f"All reports and the 60-day trend chart are in {config.GRAPH_DIR}")
            print(f"Predicted Price for 60 days from now: ₹{forecast_df['Predicted_Close'].iloc[-1]:.2f}")
            
        else:
            print("Processing failed: Resulting dataframe is empty.")
    else:
        print("Failed to download data.")

if __name__ == "__main__":
    main()