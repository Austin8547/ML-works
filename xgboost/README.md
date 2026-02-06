# 📈 StockSense AI

**StockSense AI** is a powerful stock prediction and analysis tool built with **XGBoost** and **Streamlit**. It provides advanced technical insights, 60-day price forecasts, and interactive visualizations for all **NIFTY 50** stocks.

## 🚀 Features

*   **Dynamic Stock Selection**: Choose any stock from the NIFTY 50 index (e.g., RELIANCE, TCS, HDFCBANK).
*   **AI Forecasting**: Uses a recursive XGBoost model to predict stock prices for the next 60 days.
*   **Smart Analyst**: Automated "Buy/Sell/Hold" signals based on Trend (SMA/EMA) and Momentum (MACD) analysis.
*   **Interactive Dashboard**: A modern, dark-themed UI built with Streamlit.
    *   **Forecast Tab**: Visualize the future price trajectory.
    *   **EDA Tab**: Deep dive into historical trends, volume, and indicators.
    *   **Validation Tab**: Transparent model performance metrics (RMSE, Learning Curves).
    *   **Raw Data**: Downloadable CSV reports of the forecasts.

## 🛠️ Installation

1.  **Clone the repository** (or navigate to the project folder):
    ```bash
    cd xgboost
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 🖥️ Usage

Run the Streamlit application:

```bash
streamlit run streamlit_app.py
```

1.  Select a stock from the sidebar.
2.  Click **"Run AI Analysis"**.
3.  Explore the generated reports and charts in the tabs.

## 📂 Project Structure

*   `streamlit_app.py`: The frontend dashboard application.
*   `main.py`: The core pipeline orchestrator.
*   `config.py`: Configuration settings (paths, model parameters).
*   `src/`:
    *   `analyst.py`: Logic for generating trading signals and insights.
    *   `models/`: XGBoost training (`trainer.py`), forecasting (`forecaster.py`), and evaluation (`evaluate.py`).
    *   `data_loader/`: Handles data fetching from Yahoo Finance.
    *   `data_processing/`: Feature engineering (Lags, MACD, etc.).
    *   `visualization/`: Generates static plots for the dashboard.
*   `data/`: Stores downloaded stock data, trained models, and generated graphs.

## 🧠 How It Works

1.  **Data**: Fetches 5 years of daily OHLCV data via `yfinance`.
2.  **Processing**: Calculates technical indicators (SMA_5, EMA_20, MACD, Lags).
3.  **Training**: Trains an XGBoost regressor to predict the *next day's* closing price.
4.  **Forecasting**: Uses a recursive loop to predict 60 days into the future, dynamically updating indicators at each step.
5.  **Analysis**: Evaluates the latest market state to provide actionable trading signals.

---
*Built with ❤️ for ML Enthusiasts & Traders.*
