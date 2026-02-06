import streamlit as st
import pandas as pd
import os
import glob
import config
import main

# --- Page Configuration ---
st.set_page_config(
    page_title="StockSense AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS Styling ---
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    h1, h2, h3 {
        color: #00FFD1 !important;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .stButton>button {
        color: #ffffff;
        background-color: #FF007A;
        border-radius: 10px;
        border: none;
        padding: 10px 24px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #D60066;
    }
    .metric-card {
        background-color: #1e2130;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #333;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.title("🚀 Control Panel")
st.sidebar.markdown("---")

selected_ticker = st.sidebar.selectbox(
    "Select Stock Ticker",
    options=config.NIFTY50_TICKERS,
    index=config.NIFTY50_TICKERS.index(config.TICKER) if config.TICKER in config.NIFTY50_TICKERS else 0
)

# Optional: Text input for custom ticker
custom_ticker = st.sidebar.text_input("Or enter custom (e.g., ZOMATO.NS)")
if custom_ticker:
    selected_ticker = custom_ticker.upper()

st.sidebar.markdown("---")
analyze_btn = st.sidebar.button("Run AI Analysis")

# --- Main Page ---
st.title("📈 StockSense AI: Prediction Dashboard")
st.markdown(f"### Analyzing: **{selected_ticker}**")

if analyze_btn:
    with st.spinner(f"Running XGBoost Pipeline for {selected_ticker}... This may take a minute."):
        try:
            # RUN THE PIPELINE
            # Now returns a tuple (forecast_df, insights)
            forecast_df, insights = main.run_pipeline(selected_ticker)
            
            if forecast_df is not None:
                st.session_state['forecast_df'] = forecast_df
                st.session_state['insights'] = insights
                st.session_state['ticker'] = selected_ticker
                st.success("Analysis Complete!")
            else:
                st.error("Analysis Failed. Please check the logs or try another stock.")
                
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.code(e)

# --- Results Display ---
if 'forecast_df' in st.session_state and st.session_state.get('ticker') == selected_ticker:
    df = st.session_state['forecast_df']
    insights = st.session_state.get('insights', None)
    
    # KPIs
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    last_pred = df['Predicted_Close'].iloc[-1]
    first_pred = df['Predicted_Close'].iloc[0]
    growth = ((last_pred - first_pred) / first_pred) * 100
    
    with col1:
        st.metric("Current Prediction", f"₹{first_pred:.2f}")
    with col2:
        st.metric("Future Prediction (60 Days)", f"₹{last_pred:.2f}")
    with col3:
        st.metric("Expected Growth", f"{growth:.2f}%", delta_color="normal")
    with col4:
        if insights:
            st.metric("Analyst Signal", insights['signal'], delta=insights['metrics']['Trend'], delta_color="off")
        
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["💡 Analyst Report", "🔮 Forecast", "📊 Deep Dive (EDA)", "🛠️ Model Health", "📄 Raw Data"])
    
    # --- TAB 1: ANALYST REPORT ---
    with tab1:
        if insights:
            st.subheader(f"Analyst Verdict: {insights['signal']}")
            
            # Colored Banner
            color_map = {
                "green": "success",
                "lightgreen": "success",
                "orange": "warning",
                "red": "error",
                "gray": "info"
            }
            banner_fn = getattr(st, color_map.get(insights['signal_color'], "info"))
            banner_fn(f"**Action**: {insights['summary']}")
            
            st.markdown("### Key Technicals")
            m = insights['metrics']
            c1, c2, c3 = st.columns(3)
            with c1:
                st.info(f"**Trend**: {m['Trend']}")
                st.write(f"SMA (5): {m['SMA_5']:.2f}")
                st.write(f"EMA (20): {m['EMA_20']:.2f}")
            with c2:
                st.info(f"**Momentum**: {m['Momentum']}")
                st.write(f"MACD: {m['MACD']:.2f}")
                st.write(f"Signal: {m['MACD_Signal']:.2f}")
            with c3:
                st.write(f"**Current Price**: ₹{insights['current_price']:.2f}")
        else:
            st.warning("No insights available.")

    with tab2:
        st.subheader("60-Day Price Forecast")
        # Load the generated forecast image
        forecast_img = os.path.join(config.FORECAST_GRAPH_DIR, "9_final_forecast.png")
        if os.path.exists(forecast_img):
            st.image(forecast_img, use_container_width=True)
        else:
            st.warning("Forecast chart not found.")
            
    with tab2:
        st.subheader("Exploratory Data Analysis")
        # Load all images from EDA folder
        eda_images = sorted(glob.glob(os.path.join(config.EDA_GRAPH_DIR, "*.png")))
        if eda_images:
            cols = st.columns(2)
            for i, img in enumerate(eda_images):
                with cols[i % 2]:
                    st.image(img, use_container_width=True, caption=os.path.basename(img))
        else:
            st.info("No EDA charts found.")

    with tab3:
        st.subheader("Model Validation & Training")
        # Load validation images
        val_images = sorted(glob.glob(os.path.join(config.VAL_GRAPH_DIR, "*.png")))
        if val_images:
             cols = st.columns(2)
             for i, img in enumerate(val_images):
                with cols[i % 2]:
                    st.image(img, use_container_width=True, caption=os.path.basename(img))
        else:
            st.info("No validation charts found.")
            
    with tab4:
        st.subheader("Detailed Forecast Data")
        st.dataframe(df)
        
        csv_path = os.path.join(config.REPORT_DIR, f"{selected_ticker}_60_day_forecast.csv")
        if os.path.exists(csv_path):
            with open(csv_path, "rb") as f:
                st.download_button(
                    label="Download Report CSV",
                    data=f,
                    file_name=os.path.basename(csv_path),
                    mime="text/csv"
                )
else:
    st.info("👈 Select a stock from the sidebar and click **Run AI Analysis** to start.")
