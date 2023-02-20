import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima_model import ARIMA
from sklearn.svm import SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingRegressor
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

# Define the function to get the historical price data
@st.cache
def get_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# Define the function to train the regression model
@st.cache
def train_model(data, model_name):
    if model_name == 'Linear Regression':
        model = LinearRegression()
    elif model_name == 'Decision Tree Regressor':
        model = DecisionTreeRegressor()
    elif model_name == 'Random Forest Regressor':
        model = RandomForestRegressor()
    elif model_name == 'Gradient Boosting Regressor':
        model = GradientBoostingRegressor()
    elif model_name == 'ARIMA':
        model = ARIMA(data['Close'], order=(5, 1, 0))
        model = model.fit(disp=0)
    elif model_name == 'Support Vector Machines':
        model = SVR(kernel='rbf')
        X = np.array(range(len(data)))
        X = X.reshape(-1, 1)
        y = data['Close']
        model.fit(X, y)
    elif model_name == 'Naive Bayes':
        model = GaussianNB()
        X = np.array(range(len(data)))
        X = X.reshape(-1, 1)
        y = np.zeros(len(data))
        y[data['Close'].diff().shift(-1) > 0] = 1
        model.fit(X, y)
    elif model_name == 'LSTM':
        X_train = np.array(data['Close'][0:-30]).reshape(-1,1)
        y_train = np.array(data['Close'][30:]).reshape(-1,1)
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(LSTM(units=50))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        model.fit(X_train, y_train, epochs=100, batch_size=32)

    else:
        raise ValueError('Invalid model name')

    return model

# Define the function to get the prediction
def predict_price(model, X):
    if model.__class__.__name__ == 'LSTM':
        # Scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_scaled = scaler.fit_transform(X)

        # Reshape the data
        X_scaled = np.reshape(X_scaled, (X_scaled.shape[0], X_scaled.shape[1], 1))

        # Make the prediction
        y_pred_scaled = model.predict(X_scaled)

        # Inverse the scaling
        y_pred = scaler.inverse_transform(y_pred_scaled)

    else:
        y_pred = model.predict(X)

    return y_pred

# Define the Streamlit web app
