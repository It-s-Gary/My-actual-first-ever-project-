import streamlit as st
from datetime import date
from PIL import Image
import yfinance as yf
from prophet import Prophet 
from prophet.plot import plot_plotly
import plotly.express as px
import pandas as pd 
import numpy as np
st.write("Please don't use this as real financial advice; this was just a beginner summer project  of mine (:")
START = "2020-08-25"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Predictor")
stocks = ("AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA",
    "JPM", "BAC", "C", "GS", "MS",
    "WMT", "HD", "NKE", "PG", "KO", "PEP",
    "JNJ", "PFE", "MRK", "MRNA",
    "XOM", "CVX", "BA", "CAT",
    "DIS", "SBUX", "ADBE", "CRM", "RBLX"
 )
selected_stocks = st.selectbox("Select Dataset for prediction", stocks)
n_years = st.slider("Years of Prediction", 1, 5)
period = n_years * 365
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Loading Data...")
data = load_data(selected_stocks)
data_load_state.text("Loading Data... Done!")



df = pd.DataFrame(data)

df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df.columns]

close_col = f'Close_{selected_stocks}'
open_col = f'Open_{selected_stocks}'
date_col = 'Date_' 
fig = px.line(df, x=date_col, y=close_col, title=f'{selected_stocks} Closing Price Over Time')
st.plotly_chart(fig)
fig_2 = px.line(df, x=date_col, y=open_col, title=f'{selected_stocks} Closing Price Over Time')
st.plotly_chart(fig_2)
data.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in data.columns]
df_train = data[['Date_', close_col]].copy()
df_train = df_train.rename(columns={
    "Date_" : "ds",
    close_col : "y"})

df_train['ds'] = pd.to_datetime(df_train['ds'], errors='coerce') 
df_train['y'] = pd.to_numeric(df_train['y'], errors='coerce')

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)
st.subheader("Forecast data")
st.write(forecast.tail())
