import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import yfinance as yf
import ta  # Technical Analysis library

# Load historical stock price data using yfinance
ticker = 'AAPL'  # Example: Apple Inc.
stock_data = yf.download(ticker, start='1980-01-01', end='2023-01-01')
stock_data['Date'] = stock_data.index

# Feature Engineering
stock_data['Year'] = stock_data['Date'].dt.year
stock_data['Month'] = stock_data['Date'].dt.month
stock_data['Day'] = stock_data['Date'].dt.day
stock_data['DayOfWeek'] = stock_data['Date'].dt.dayofweek

# Adding moving average feature with increased window size
window_size = 45
stock_data[f'{window_size}d_MA'] = stock_data['Close'].rolling(window=window_size).mean()

# Adding RSI and MACD features
stock_data['RSI'] = ta.momentum.RSIIndicator(close=stock_data['Close'], window=14).rsi()
stock_data['MACD'] = ta.trend.MACD(close=stock_data['Close']).macd()

# Creating lag features
for lag in range(1, 6):
    stock_data[f'Close_lag_{lag}'] = stock_data['Close'].shift(lag)

# Drop rows with NaN values created by lag features and moving average
stock_data.dropna(inplace=True)

# Define features (X) and target (y)
features = ['Open', 'High', 'Low', 'Volume', 'Year', 'Month', 'Day', 'DayOfWeek', f'{window_size}d_MA', 'RSI', 'MACD'] + [f'Close_lag_{lag}' for lag in range(1, 6)]
X = stock_data[features]
y = stock_data['Close']

# Normalize the features and target variable
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, shuffle=False)

# Create a Linear Regression model
lr_model = LinearRegression()

# Fit the model
lr_model.fit(X_train, y_train)

# Make Predictions and Evaluate the Model
y_pred_train_lr = lr_model.predict(X_train)
y_pred_test_lr = lr_model.predict(X_test)

# Inverse transform the predictions to original scale
y_pred_train_lr_orig = scaler_y.inverse_transform(y_pred_train_lr.reshape(-1, 1)).flatten()
y_pred_test_lr_orig = scaler_y.inverse_transform(y_pred_test_lr.reshape(-1, 1)).flatten()
y_train_orig = scaler_y.inverse_transform(y_train.reshape(-1, 1)).flatten()
y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

mse_train_lr = mean_squared_error(y_train_orig, y_pred_train_lr_orig)
mse_test_lr = mean_squared_error(y_test_orig, y_pred_test_lr_orig)

print("Mean Squared Error (Train - LR):", mse_train_lr)
print("Mean Squared Error (Test - LR):", mse_test_lr)

# Plot Results
plt.figure(figsize=(14, 7))

# Plot training data
plt.plot(stock_data.index[:len(y_train)], y_train_orig, label='Actual Train', color='blue')
plt.plot(stock_data.index[:len(y_train)], y_pred_train_lr_orig, label=f'Predicted Train (LR)', color='cyan')

# Plot test data
plt.plot(stock_data.index[len(y_train):], y_test_orig, label='Actual Test', color='orange')
plt.plot(stock_data.index[len(y_train):], y_pred_test_lr_orig, label=f'Predicted Test (LR)', color='red')

plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title(f'{ticker} Stock Price Prediction with Linear Regression')
plt.legend()
plt.show()
