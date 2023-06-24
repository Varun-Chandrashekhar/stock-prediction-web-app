import streamlit as st
from datetime import date
import pandas as pd
import numpy as np

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

# Set date range
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# Set page title and layout
st.set_page_config(page_title='Stock Predictor', layout='wide')

# Custom CSS styles
st.markdown(
    """
    <style>
    .main-title {
        font-size: 36px;
        font-weight: bold;
        color: #FF5733;
        padding: 20px;
        text-align: center;
    }
    .section-title {
        font-size: 24px;
        font-weight: bold;
        color: #333333;
        padding: 10px 0px;
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
    </style>
    """,
    unsafe_allow_html=True
)

# Main title
st.markdown("<h1 class='main-title'>Stock Predictor</h1>", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("<h3>Select Stock</h3>", unsafe_allow_html=True)
selected_stock = st.sidebar.selectbox('', ('AAPL', 'GOOG', 'MSFT', 'GME'))
n_years = st.sidebar.slider('Select Time Horizon (in years)', 1, 10, 5)
period = n_years * 365

# Load data
@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data = load_data(selected_stock)

# Display raw data
st.markdown("<h2 class='section-title'>Raw Data</h2>", unsafe_allow_html=True)
st.dataframe(data.tail(), height=250)

# Plot raw data
st.markdown("<h3 class='chart-title'>Stock Performance</h3>", unsafe_allow_html=True)
fig = go.Figure()
fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="Open", line=dict(color='#FF5733')))
fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Close", line=dict(color='#17BECF')))
fig.update_layout(
    title='Stock Price',
    xaxis_title='Date',
    yaxis_title='Price',
    showlegend=True,
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=0.8)
)
st.plotly_chart(fig)

# Forecasting
df_train = data[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})
m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Display forecast data
st.markdown("<h2 class='section-title'>Forecast Data</h2>", unsafe_allow_html=True)
st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(), height=250)

# Plot forecast
st.markdown("<h3 class='chart-title'>Stock Price Forecast</h3>", unsafe_allow_html=True)
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

# Display forecast components
st.markdown("<h2 class='section-title'>Forecast Components</h2>", unsafe_allow_html=True)
fig2 = m.plot_components(forecast)
st.write(fig2)

# Analyze forecast accuracy
st.markdown("<h2 class='section-title'>Forecast Accuracy</h2>", unsafe_allow_html=True)
actual = data[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})
actual['ds'] = pd.to_datetime(actual['ds']).dt.date.astype('datetime64[ns]')
forecast['ds'] = forecast['ds'].astype('datetime64[ns]')
merged_df = forecast.merge(actual, on='ds', how='inner')
merged_df['error'] = merged_df['y'] - merged_df['yhat']
mean_abs_error = np.mean(np.abs(merged_df['error']))
st.write(f"Mean Absolute Error: {mean_abs_error:.2f}")

# Show error distribution
st.markdown("<h3 class='chart-title'>Error Distribution</h3>", unsafe_allow_html=True)
fig3 = go.Figure()
fig3.add_trace(go.Histogram(x=merged_df['error'], nbinsx=30))
fig3.update_layout(
    title='Error Distribution',
    xaxis_title='Error',
    yaxis_title='Count',
    showlegend=False
)
st.plotly_chart(fig3)

# Show prediction confidence intervals
st.markdown("<h3 class='chart-title'>Prediction Confidence Intervals</h3>", unsafe_allow_html=True)
fig4 = go.Figure()
fig4.add_trace(go.Scatter(
    x=merged_df['ds'],
    y=merged_df['y'],
    mode='lines',
    name='Actual',
    line=dict(color='#17BECF')
))
fig4.add_trace(go.Scatter(
    x=merged_df['ds'],
    y=merged_df['yhat_upper'],
    mode='lines',
    name='Upper Bound',
    line=dict(color='#FF5733', dash='dash')
))
fig4.add_trace(go.Scatter(
    x=merged_df['ds'],
    y=merged_df['yhat_lower'],
    mode='lines',
    name='Lower Bound',
    line=dict(color='#FF5733', dash='dash')
))
fig4.update_layout(
    title='Prediction Confidence Intervals',
    xaxis_title='Date',
    yaxis_title='Price',
    showlegend=True,
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=0.8)
)
st.plotly_chart(fig4)
