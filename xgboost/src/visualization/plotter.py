import matplotlib.pyplot as plt
import seaborn as sns
import os
import config

# --- Professional Styling ---
# Using a clean dark background for a "Terminal/Trading View" feel
plt.style.use('dark_background') 
sns.set_context("talk") # Makes labels and lines thicker/readable

def apply_custom_style(ax, title):
    """Helper to apply a consistent clean look to each plot."""
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20, color='#E0E0E0')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', alpha=0.2)

def plot_all_charts(df):
    """
    Generates and saves all requested technical and statistical charts
    with a modern dark-neon theme.
    """
    
    # 1. Daily Close and Date Chart
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=df, x='Date', y='Close', color='#00FFD1', linewidth=2, ax=ax)
    apply_custom_style(ax, f"{config.TICKER} - Performance")
    plt.tight_layout()
    plt.savefig(os.path.join(config.GRAPH_DIR, "1_daily_close.png"))
    plt.close()

    # 2. Volume Chart
    fig, ax = plt.subplots(figsize=(12, 4))
    # Using a gradient-like color for volume
    sns.barplot(data=df, x='Date', y='Volume', color='#FF007A', alpha=0.8, ax=ax)
    ax.xaxis.set_major_locator(plt.MaxNLocator(10)) 
    apply_custom_style(ax, "Market Liquidity (Volume)")
    plt.tight_layout()
    plt.savefig(os.path.join(config.GRAPH_DIR, "2_volume_chart.png"))
    plt.close()

    # 3. EMA and SMA with Daily Price
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=df, x='Date', y='Close', label='Price', color='#FFFFFF', alpha=0.3, ax=ax)
    sns.lineplot(data=df, x='Date', y='SMA_5', label='Fast SMA (5)', color='#39FF14', linewidth=1.5, ax=ax)
    sns.lineplot(data=df, x='Date', y='EMA_20', label='Slow EMA (20)', color='#FF00FF', linewidth=1.5, ax=ax)
    apply_custom_style(ax, "Trend Analysis: SMA vs EMA")
    plt.legend(facecolor='black', edgecolor='none')
    plt.tight_layout()
    plt.savefig(os.path.join(config.GRAPH_DIR, "3_ma_trends.png"))
    plt.close()

    # 4. MACD and Daily Price (Subplots)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    sns.lineplot(data=df, x='Date', y='Close', ax=ax1, color='#00FFD1')
    apply_custom_style(ax1, "Price Momentum (MACD)")
    
    # MACD Plot with Neon Fill
    ax2.plot(df['Date'], df['MACD'], label='MACD', color='#39FF14', linewidth=1)
    ax2.plot(df['Date'], df['MACD_Signal'], label='Signal', color='#FF00FF', linewidth=1)
    ax2.fill_between(df['Date'], df['MACD_Hist'], 0, where=(df['MACD_Hist'] >= 0), color='#39FF14', alpha=0.4)
    ax2.fill_between(df['Date'], df['MACD_Hist'], 0, where=(df['MACD_Hist'] < 0), color='#FF007A', alpha=0.4)
    apply_custom_style(ax2, "") # Subtitle handled by ax1
    plt.tight_layout()
    plt.savefig(os.path.join(config.GRAPH_DIR, "4_macd_analysis.png"))
    plt.close()

    # 5 & 6. Distributions (Combined into one figure for cleaner reports)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    sns.histplot(df['Volume'], kde=True, color='#FF007A', ax=ax1, element="step")
    apply_custom_style(ax1, "Volume Distribution")
    
    sns.histplot(df['Close'], kde=True, color='#00FFD1', ax=ax2, element="step")
    apply_custom_style(ax2, "Price Density")
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.GRAPH_DIR, "5_6_distributions.png"))
    plt.close()

    print(f"Modernized charts saved to {config.GRAPH_DIR}")