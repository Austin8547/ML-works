# Reliance Stock Price Prediction using XGBoost

This project uses an **XGBoost Regressor** model to forecast the stock price of **Reliance Industries (RELIANCE.NS)** for the next 21 trading days (approximately one month).

The script fetches the last 5 years of historical stock data, preprocesses it, performs feature engineering, trains a machine learning model using robust time-series cross-validation, and then predicts future prices on an iterative, day-by-day basis.

## Features 

* **Data Acquisition**: Downloads 5 years of daily stock data for Reliance Industries using the `yfinance` library.
* **Data Visualization**: Generates plots for historical prices (Open, High, Low, Close) and trading volume to provide an initial overview.
* **Feature Engineering**: Creates lagged features from the closing price (`Close_lag1`, `Close_lag2`, etc.) to provide the model with historical context.
* **Time-Series Cross-Validation**: Employs `TimeSeriesSplit` from Scikit-learn to evaluate the model in a way that respects the temporal order of the data, preventing lookahead bias.
* **Model Training**: Trains a powerful and efficient XGBoost Regressor on the prepared dataset.
* **Iterative Forecasting**: Predicts future prices one day at a time, using the prediction of the current day to help generate features for the next day.
* **Prediction Visualization**: Plots the final predictions alongside the most recent historical data for a clear visual comparison.
* **Performance Summary**: Outputs a summary of the prediction, including the initial price, the final predicted price, the percentage change, and a bullish/bearish outlook.

***

## Technologies Used 

* **Python 3.x**
* **Pandas**: For data manipulation and analysis.
* **NumPy**: For numerical operations.
* **yfinance**: To download market data from Yahoo! Finance.
* **Scikit-learn**: For data preprocessing (`StandardScaler`) and model validation (`TimeSeriesSplit`).
* **XGBoost**: For building the gradient boosting regression model.
* **Matplotlib** & **Seaborn**: For data visualization.

***

## Setup and Usage 

Follow these steps to get the project running on your local machine.

### 1. Prerequisites

Make sure you have **Python 3.7 or higher** installed.

### 2. Clone the Repository

```bash
git clone [https://github.com/Austin8547/ML-works/xgboost.git]