import streamlit as st
from datetime import date, timedelta
import pandas as pd
import numpy as np

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
from plotly.subplots import make_subplots

# Set date range
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# Set page configuration
st.set_page_config(
    page_title='Varun\'s Stock Predictor',
    layout='wide',
    initial_sidebar_state='expanded',
    menu_items={
        'About': "### Varun's Stock Predictor\n"
                 "An interactive application to analyze and forecast stock prices using historical data and Prophet."
    }
)

# Custom CSS styles for better aesthetics
st.markdown(
    """
    <style>
    .main-title {
        font-size: 48px;
        font-weight: bold;
        color: #2E86C1;
        padding: 20px;
        text-align: center;
    }
    .section-title {
        font-size: 28px;
        font-weight: bold;
        color: #2C3E50;
        padding: 10px 0px;
        border-bottom: 2px solid #E0E0E0;
    }
    .data-table {
        margin-top: 20px;
        font-size: 14px;
    }
    .data-table table {
        border-collapse: collapse;
        width: 100%;
    }
    .data-table th, .data-table td {
        padding: 8px;
        text-align: left;
        border-bottom: 1px solid #ddd;
    }
    .metric {
        text-align: center;
        padding: 10px;
        background-color: #f8f9fa;
        border-radius: 5px;
        margin: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Main title
st.markdown("<h1 class='main-title'>Varun's Stock Predictor</h1>", unsafe_allow_html=True)

# Sidebar for user inputs
st.sidebar.header("User Input Parameters")

def user_input_features():
    ticker = st.sidebar.selectbox(
        'Select Stock',
        ('AAPL', 'GOOG', 'MSFT', 'GME', 'AMZN', 'TSLA', 'FB', 'NFLX')
    )
    start_date = st.sidebar.date_input(
        "Start Date",
        value=pd.to_datetime("2015-01-01")
    )
    end_date = st.sidebar.date_input(
        "End Date",
        value=pd.to_datetime(TODAY)
    )
    if start_date > end_date:
        st.sidebar.error("Error: Start date must be before end date.")
    n_years = st.sidebar.slider(
        'Forecast Horizon (in years)',
        1, 10, 5
    )
    return ticker, start_date, end_date, n_years

selected_stock, start_date, end_date, n_years = user_input_features()
period = n_years * 365

# Load data with caching to improve performance
@st.cache_data
def load_data(ticker, start, end):
    data = yf.download(ticker, start, end)
    data.reset_index(inplace=True)
    data['Date'] = pd.to_datetime(data['Date'])
    return data

data_load_state = st.text('Loading data...')
data = load_data(selected_stock, start_date, end_date)
data_load_state.text('Loading data...done!')

# Display raw data
st.markdown("<h2 class='section-title'>Raw Data</h2>", unsafe_allow_html=True)
st.dataframe(data.tail(), height=300)

# Add technical indicators
def add_technical_indicators(df):
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['RSI'] = compute_RSI(df['Close'], 14)
    return df

def compute_RSI(series, period):
    delta = series.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

data = add_technical_indicators(data)

# Plot stock performance with technical indicators
st.markdown("<h2 class='section-title'>Stock Performance</h2>", unsafe_allow_html=True)
fig_stock = make_subplots(rows=2, cols=1, shared_xaxes=True,
                          vertical_spacing=0.03,
                          subplot_titles=('Price', 'Volume'),
                          row_width=[0.2, 0.7])

# Price subplot
fig_stock.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='Open', line=dict(color='#FF5733')), row=1, col=1)
fig_stock.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Close', line=dict(color='#17BECF')), row=1, col=1)
fig_stock.add_trace(go.Scatter(x=data['Date'], y=data['MA50'], name='MA50', line=dict(color='orange', dash='dash')), row=1, col=1)
fig_stock.add_trace(go.Scatter(x=data['Date'], y=data['MA200'], name='MA200', line=dict(color='green', dash='dash')), row=1, col=1)

# Volume subplot
fig_stock.add_trace(go.Bar(x=data['Date'], y=data['Volume'], name='Volume', marker_color='#2E86C1'), row=2, col=1)

fig_stock.update_layout(height=600, width=1200, showlegend=True, title=f"{selected_stock} Stock Performance")
st.plotly_chart(fig_stock)

# Plot technical indicators
st.markdown("<h2 class='section-title'>Technical Indicators</h2>", unsafe_allow_html=True)

# MACD
fig_macd = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         vertical_spacing=0.03,
                         subplot_titles=('MACD', 'Signal Line'))

fig_macd.add_trace(go.Scatter(x=data['Date'], y=data['MACD'], name='MACD', line=dict(color='purple')), row=1, col=1)
fig_macd.add_trace(go.Scatter(x=data['Date'], y=data['Signal_Line'], name='Signal Line', line=dict(color='orange')), row=1, col=1)
fig_macd.update_layout(height=400, width=1200, showlegend=True, title='MACD Indicator')
st.plotly_chart(fig_macd)

# RSI
fig_rsi = go.Figure()
fig_rsi.add_trace(go.Scatter(x=data['Date'], y=data['RSI'], name='RSI', line=dict(color='magenta')))
fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
fig_rsi.update_layout(height=300, width=1200, title='Relative Strength Index (RSI)')
st.plotly_chart(fig_rsi)

# Forecasting with Prophet
st.markdown("<h2 class='section-title'>Forecasting</h2>", unsafe_allow_html=True)

# Prepare data for Prophet
df_train = data[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})

# Initialize and fit Prophet model
m = Prophet(daily_seasonality=True, yearly_seasonality=True, weekly_seasonality=True)
m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
m.fit(df_train)

# Create future dataframe
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Display forecast data
st.markdown("<h3>Forecast Data</h3>", unsafe_allow_html=True)
st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(), height=300)

# Plot forecast
st.markdown("<h3>Stock Price Forecast</h3>", unsafe_allow_html=True)
fig_forecast = plot_plotly(m, forecast)
fig_forecast.update_layout(height=600, width=1200, title=f'{selected_stock} Price Forecast')
st.plotly_chart(fig_forecast)

# Display forecast components
st.markdown("<h3>Forecast Components</h3>", unsafe_allow_html=True)
fig_components = m.plot_components(forecast)
st.write(fig_components)

# Analyze forecast accuracy
st.markdown("<h2 class='section-title'>Forecast Accuracy</h2>", unsafe_allow_html=True)
actual = df_train.copy()
actual['ds'] = actual['ds'].dt.date
forecast['ds'] = forecast['ds'].dt.date
merged_df = forecast.merge(actual, on='ds', how='inner')
merged_df['error'] = merged_df['y'] - merged_df['yhat']
mean_abs_error = np.mean(np.abs(merged_df['error']))
st.metric(label="Mean Absolute Error (MAE)", value=f"${mean_abs_error:.2f}")

# Show error distribution
st.markdown("<h3>Error Distribution</h3>", unsafe_allow_html=True)
fig_error = go.Figure()
fig_error.add_trace(go.Histogram(x=merged_df['error'], nbinsx=30, marker_color='#FF5733'))
fig_error.update_layout(
    title='Error Distribution',
    xaxis_title='Error',
    yaxis_title='Count',
    bargap=0.1,
    height=400,
    width=1200
)
st.plotly_chart(fig_error)

# Show prediction confidence intervals
st.markdown("<h3>Prediction Confidence Intervals</h3>", unsafe_allow_html=True)
fig_confidence = make_subplots(rows=1, cols=1)

fig_confidence.add_trace(go.Scatter(
    x=merged_df['ds'],
    y=merged_df['y'],
    mode='lines',
    name='Actual',
    line=dict(color='#17BECF')
))
fig_confidence.add_trace(go.Scatter(
    x=merged_df['ds'],
    y=merged_df['yhat_upper'],
    mode='lines',
    name='Upper Bound',
    line=dict(color='#FF5733', dash='dash')
))
fig_confidence.add_trace(go.Scatter(
    x=merged_df['ds'],
    y=merged_df['yhat_lower'],
    mode='lines',
    name='Lower Bound',
    line=dict(color='#FF5733', dash='dash')
))
fig_confidence.update_layout(
    title='Prediction Confidence Intervals',
    xaxis_title='Date',
    yaxis_title='Price',
    showlegend=True,
    height=600,
    width=1200
)
st.plotly_chart(fig_confidence)

# Additional Analysis: Correlation with Market Indices
st.markdown("<h2 class='section-title'>Market Correlation</h2>", unsafe_allow_html=True)
index_options = ('^GSPC', '^DJI', '^IXIC')  # S&P 500, Dow Jones, NASDAQ
selected_index = st.selectbox('Select Market Index for Correlation', index_options)

# Load index data
index_data = load_data(selected_index, start_date, end_date)
index_data = index_data[['Date', 'Close']].rename(columns={"Close": "Index_Close"})
merged_data = data.merge(index_data, on='Date', how='inner')

# Calculate correlation
correlation = merged_data['Close'].corr(merged_data['Index_Close'])
st.write(f"Correlation between {selected_stock} and {selected_index}: {correlation:.2f}")

# Plot correlation
fig_corr = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         vertical_spacing=0.03,
                         subplot_titles=(f'{selected_stock} Close Price', f'{selected_index} Close Price'))

fig_corr.add_trace(go.Scatter(x=merged_data['Date'], y=merged_data['Close'], name=selected_stock, line=dict(color='#2E86C1')), row=1, col=1)
fig_corr.add_trace(go.Scatter(x=merged_data['Date'], y=merged_data['Index_Close'], name=selected_index, line=dict(color='#FF5733')), row=2, col=1)

fig_corr.update_layout(height=600, width=1200, showlegend=True, title=f"{selected_stock} vs {selected_index} Close Prices")
st.plotly_chart(fig_corr)

# Download data as CSV
st.markdown("<h2 class='section-title'>Download Data</h2>", unsafe_allow_html=True)
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

csv_raw = convert_df(data)
csv_forecast = convert_df(forecast)

st.download_button(
    label="Download Raw Data as CSV",
    data=csv_raw,
    file_name=f'{selected_stock}_raw_data.csv',
    mime='text/csv',
)

st.download_button(
    label="Download Forecast Data as CSV",
    data=csv_forecast,
    file_name=f'{selected_stock}_forecast_data.csv',
    mime='text/csv',
)

# Footer
st.markdown(
    """
    <hr>
    <p style='text-align: center; color: #7f8c8d;'>
        © 2024 Varun\'s Stock Predictor. All rights reserved.
    </p>
    """,
    unsafe_allow_html=True
)
