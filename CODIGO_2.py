import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import datetime


def calculate_sma(data, period):
    return data['Close'].rolling(window=period).mean()


def calculate_rsi(data, period=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def plot_stock_data(stock_data, symbol):
    fig = go.Figure()

    # Add candlestick
    fig.add_trace(go.Candlestick(
        x=stock_data.index,
        open=stock_data['Open'],
        high=stock_data['High'],
        low=stock_data['Low'],
        close=stock_data['Close'],
        name='Candlestick'
    ))

    # Add SMA and RSI
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['SMA_50'], mode='lines', name='SMA 50'))
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['RSI'], mode='lines', name='RSI'))

    # Current price and date
    current_price = float(stock_data['Close'].iloc[-1])  # Latest price as float
    current_date = stock_data.index[-1]  # Latest date

    # Add current price marker
    fig.add_trace(go.Scatter(
        x=[current_date],
        y=[current_price],
        mode='markers+text',
        marker=dict(size=12, color='blue', symbol='circle'),
        text=[f'${current_price:.2f}'],
        textposition='top center',
        name='Current Price'
    ))

    # Layout configuration
    fig.update_layout(
        title=f'{symbol} Stock Price with Indicators',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_dark',
        showlegend=True
    )

    return fig


def preprocess_data(data):
    data['SMA_50'] = calculate_sma(data, 50)
    data['RSI'] = calculate_rsi(data, 14)
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
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(X, y, epochs=5, batch_size=32)

    return model, scaler


def predict_future(data, model, scaler, forecast_days):
    last_data = data[-forecast_days:]['Close'].values.reshape(-1, 1)
    last_data_scaled = scaler.transform(last_data)

    X_test = np.reshape(last_data_scaled, (1, forecast_days, 1))
    prediction_scaled = model.predict(X_test)
    prediction = scaler.inverse_transform(prediction_scaled)

    return prediction[0][0]


# Streamlit UI
st.title('Finance Analyzer')
st.sidebar.title('Options')

symbol = st.sidebar.text_input('Stock Symbol', '')
start_date = st.sidebar.date_input('Start Date', datetime.date(2010, 1, 1))
end_date = st.sidebar.date_input('End Date', datetime.date.today())
forecast_days = st.sidebar.slider('Days to Predict', 1, 30, 7)

try:
    stock_data = yf.download(symbol, start=start_date, end=end_date)

    if stock_data.empty:
        st.error(f"No data found for {symbol}. Check the symbol and date range.")
    else:
        stock_data = preprocess_data(stock_data)

        # Add detailed information to the sidebar
        current_close = float(stock_data['Close'].iloc[-1])
        current_open = float(stock_data['Open'].iloc[-1])
        current_high = float(stock_data['High'].iloc[-1])
        current_low = float(stock_data['Low'].iloc[-1])
        current_volume = int(stock_data['Volume'].iloc[-1])

        st.subheader(f'{symbol} Current Data')
        st.write(f"**Current Price:** ${current_close:.2f}")
        st.write(f"**Open:** ${current_open:.2f}")
        st.write(f"**High:** ${current_high:.2f}")
        st.write(f"**Low:** ${current_low:.2f}")
        st.write(f"**Volume:** {current_volume:,}")

        # Display the stock chart
        st.plotly_chart(plot_stock_data(stock_data, symbol), use_container_width=True)

        # Train the model and predict future prices
        lstm_model, scaler = train_lstm_model(stock_data, forecast_days)
        future_price = predict_future(stock_data, lstm_model, scaler, forecast_days)

        # Display prediction
        st.subheader('Future Prediction')
        st.write(f'Predicted price for {symbol} after {forecast_days} days: ${future_price:.2f}')

except Exception as e:
    st.error(f"Error loading data or making predictions: {str(e)}")
