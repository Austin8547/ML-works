import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import os
import config

def validate_performance(df, features):
    """
    Performs Time Series CV and plots Training vs Validation Learning Curves.
    """
    X = df[features]
    y = df[config.TARGET_COL]
    
    tscv = TimeSeriesSplit(n_splits=config.TSC_SPLITS)
    rmse_scores = []
    
    print(f"\n--- Validating Model via {config.TSC_SPLITS}-Fold TimeSeriesSplit ---")
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        temp_model = xgb.XGBRegressor(**config.XGB_PARAMS)
        
        # Fit and capture the evaluation results
        temp_model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False
        )
        
        # 1. Capture Learning Curve Data
        results = temp_model.evals_result()
        plot_learning_curve(results, fold + 1)
        
        # 2. Final Fold Evaluation
        preds = temp_model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        rmse_scores.append(rmse)
        print(f"Fold {fold + 1} RMSE: {rmse:.2f}")

    # 3. Plot final Bar Chart of all Fold RMSEs
    plot_validation_summary(rmse_scores)
    print(f"\nAverage Validation RMSE: {np.mean(rmse_scores):.2f}")

def plot_learning_curve(results, fold_num):
    """Plots Train vs Validation RMSE to check for overfitting."""
    plt.style.use('dark_background')
    plt.figure(figsize=(10, 5))
    
    epochs = len(results['validation_0']['rmse'])
    x_axis = range(0, epochs)
    
    plt.plot(x_axis, results['validation_0']['rmse'], label='Train RMSE', color='#00FFD1')
    plt.plot(x_axis, results['validation_1']['rmse'], label='Val RMSE', color='#FF007A')
    
    plt.title(f"Learning Curve - Fold {fold_num}")
    plt.xlabel("Boosting Rounds")
    plt.ylabel("RMSE")
    plt.legend()
    plt.grid(True, alpha=0.1)
    
    plt.savefig(os.path.join(config.GRAPH_DIR, f"learning_curve_fold_{fold_num}.png"))
    plt.close()

def plot_validation_summary(rmse_scores):
    """Saves a bar chart of final RMSE scores across all folds."""
    plt.style.use('dark_background')
    plt.figure(figsize=(10, 6))
    folds = [f"Fold {i+1}" for i in range(len(rmse_scores))]
    
    # Fixed the palette warning here
    sns.barplot(x=folds, y=rmse_scores, hue=folds, palette="magma", legend=False)
    
    plt.axhline(np.mean(rmse_scores), ls='--', color='#00FFD1', label=f'Avg: {np.mean(rmse_scores):.2f}')
    plt.title("Cross-Validation RMSE Summary")
    plt.ylabel("RMSE Score")
    plt.legend()
    plt.savefig(os.path.join(config.GRAPH_DIR, "8_validation_rmse_summary.png"))
    plt.close()

def plot_feature_importance(model, features):
    """Visualizes which indicators are driving the model's decisions."""
    plt.style.use('dark_background')
    importances = model.feature_importances_
    feat_imp = pd.Series(importances, index=features).sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=feat_imp.values, y=feat_imp.index, hue=feat_imp.index, palette="viridis", legend=False)
    plt.title("Feature Importance: Indicators & Lags", color='#00FFD1')
    
    save_path = os.path.join(config.GRAPH_DIR, "7_feature_importance.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Importance chart saved to {save_path}")