import pandas as pd
import numpy as np
import config

def generate_insights(df):
    """
    Analyzes the latest data point to generate trading signals 
    and detailed technical insights.
    """
    # Get the latest row of data (the state 'now')
    # Note: df usually ends at 'today'. The forecast is future.
    # We analyze the "Current State" based on the last known actual close.
    latest = df.iloc[-1]
    
    # 1. Extract Key Metrics
    close = latest['Close']
    sma_5 = latest['SMA_5']
    ema_20 = latest['EMA_20']
    macd = latest['MACD']
    signal_line = latest['MACD_Signal']
    
    # 2. Determine Trend
    # Short-term trend is Bullish if Fast MA > Slow MA
    is_trend_bullish = sma_5 > ema_20
    trend_status = "BULLISH" if is_trend_bullish else "BEARISH"
    
    # 3. Determine Momentum
    # Momentum is positive if MACD > Signal Line
    is_momentum_bullish = macd > signal_line
    momentum_status = "POSITIVE" if is_momentum_bullish else "NEGATIVE"
    
    # 4. Generate Core Signal
    signal = "HOLD"
    color = "gray" # For UI
    reasoning = ""
    
    if is_trend_bullish and is_momentum_bullish:
        signal = "STRONG BUY"
        color = "green"
        reasoning = "Both trend and momentum are positive. The price is likely in an uptrend with strength."
    elif not is_trend_bullish and not is_momentum_bullish:
        signal = "STRONG SELL"
        color = "red"
        reasoning = "Both trend and momentum are negative. The price is likely in a downtrend."
    elif is_trend_bullish and not is_momentum_bullish:
        signal = "BUY / ACCUMULATE"
        color = "lightgreen"
        reasoning = "Trend is up, but momentum is fading. Could be a temporary pullback or consolidation."
    elif not is_trend_bullish and is_momentum_bullish:
        signal = "SELL / REDUCE"
        color = "orange"
        reasoning = "Trend is down, but momentum is rising. This might be a dead cat bounce or early reversal."
        
    # 5. Compile Report
    report = {
        "ticker": config.TICKER,
        "current_price": close,
        "signal": signal,
        "signal_color": color,
        "summary": reasoning,
        "metrics": {
            "Trend": trend_status,
            "Momentum": momentum_status,
            "SMA_5": sma_5,
            "EMA_20": ema_20,
            "MACD": macd,
            "MACD_Signal": signal_line
        }
    }
    
    return report
