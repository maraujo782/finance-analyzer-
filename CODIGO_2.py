import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import datetime

# Ensure accessibility via URL
st.set_page_config(page_title="Finance Analyzer", layout="wide")

def calculate_sma(data, period):
    return data['Close'].rolling(window=period).mean()

def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data['Close'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal

def calculate_bollinger_bands(data, window=20):
    sma = data['Close'].rolling(window=window).mean()
    std = data['Close'].rolling(window=window).std()
    upper_band = sma + (std * 2)
    lower_band = sma - (std * 2)
    return upper_band, lower_band

def plot_stock_data(stock_data, symbol, current_price):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=stock_data.index,
                                 open=stock_data['Open'],
                                 high=stock_data['High'],
                                 low=stock_data['Low'],
                                 close=stock_data['Close'],
                                 name='Candlestick'))
    # Add SMA indicators
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['SMA_50'], mode='lines', name='SMA 50'))
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['SMA_200'], mode='lines', name='SMA 200'))

    # Add MACD indicator
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['MACD'], mode='lines', name='MACD'))
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Signal'], mode='lines', name='MACD Signal'))

    # Add Bollinger Bands
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Upper_Band'], mode='lines', name='Upper Band'))
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Lower_Band'], mode='lines', name='Lower Band'))

    # Add current price marker
    if current_price is not None:
        fig.add_trace(go.Scatter(x=[stock_data.index[-1]], y=[current_price],
                                 mode='markers+text',
                                 name='Current Price',
                                 text=[f'Current: {current_price:.2f}'],
                                 textposition='top center',
                                 marker=dict(color='red', size=10)))

    fig.update_layout(title=f'{symbol} Stock Price with Indicators',
                      xaxis_title='Date',
                      yaxis_title='Price',
                      template='plotly_dark')
    return fig

def preprocess_data(data):
    data['SMA_50'] = calculate_sma(data, 50)
    data['SMA_200'] = calculate_sma(data, 200)
    data['MACD'], data['Signal'] = calculate_macd(data)
    data['Upper_Band'], data['Lower_Band'] = calculate_bollinger_bands(data)
    return data

def train_lstm_model(data, forecast_days):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[['Close']].values)

    X, y = [], []
    for i in range(forecast_days, len(scaled_data)):
        X.append(scaled_data[i-forecast_days:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(X, y, epochs=10, batch_size=32)

    return model, scaler

def predict_future(data, model, scaler, forecast_days):
    last_data = data[-forecast_days:]['Close'].values.reshape(-1, 1)
    last_data_scaled = scaler.transform(last_data)

    X_test = np.reshape(last_data_scaled, (1, forecast_days, 1))
    prediction_scaled = model.predict(X_test)
    prediction = scaler.inverse_transform(prediction_scaled)

    return prediction[0][0]

st.title('Finance Analyzer')
st.sidebar.title('Options')

symbol = st.sidebar.text_input('Stock Symbol', '')
start_date = st.sidebar.date_input('Start Date', datetime.date(2010, 1, 1))
end_date = st.sidebar.date_input('End Date', datetime.date.today())
forecast_days = st.sidebar.slider('Days to Predict', 1, 30, 7)

try:
    if symbol:
        stock_data = yf.download(symbol, start=start_date, end=end_date)

        if stock_data.empty:
            st.error(f"No data found for {symbol}. Check the symbol and date range.")
        else:
            stock_data = preprocess_data(stock_data)
            current_price = stock_data['Close'].iloc[-1] if not stock_data['Close'].empty else None

            if current_price is None:
                st.error("Unable to determine the current price.")
            else:
                st.plotly_chart(plot_stock_data(stock_data, symbol, float(current_price)), use_container_width=True)

                lstm_model, scaler = train_lstm_model(stock_data, forecast_days)
                future_price = predict_future(stock_data, lstm_model, scaler, forecast_days)

                st.subheader('Future Prediction')
                st.write(f'Predicted price for {symbol} after {forecast_days} days: ${float(future_price):.2f}')
                prediction_date = (pd.to_datetime(end_date) + pd.Timedelta(days=forecast_days)).strftime("%Y-%m-%d")
                st.write(f'Prediction Date: {prediction_date}')

                # Display past projections (value change over time)
                st.line_chart(stock_data['Close'], use_container_width=True)

except Exception as e:
    st.error(f"Error loading data or making predictions: {str(e)}")
