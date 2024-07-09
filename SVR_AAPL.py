import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import yfinance as yf
import ta  # Technical Analysis library

# Load historical stock price data using yfinance
ticker = 'AAPL'  # Example: Apple Inc.
stock_data = yf.download(ticker, start='2000-01-01', end='2023-01-01')
stock_data['Date'] = stock_data.index

# Feature Engineering
stock_data['Year'] = stock_data['Date'].dt.year
stock_data['Month'] = stock_data['Date'].dt.month
stock_data['Day'] = stock_data['Date'].dt.day
stock_data['DayOfWeek'] = stock_data['Date'].dt.dayofweek

# Adding moving average feature with variable window size
ma_window = 45
stock_data[f'{ma_window}d_MA'] = stock_data['Close'].rolling(window=ma_window).mean()

# Adding RSI and MACD features
stock_data['RSI'] = ta.momentum.RSIIndicator(close=stock_data['Close'], window=14).rsi()
stock_data['MACD'] = ta.trend.MACD(close=stock_data['Close']).macd()

# Creating lag features
for lag in range(1, 6):
    stock_data[f'Close_lag_{lag}'] = stock_data['Close'].shift(lag)

# Drop rows with NaN values created by lag features and moving average
stock_data.dropna(inplace=True)

# Define features (X) and target (y)
features = ['Open', 'High', 'Low', 'Volume', 'Year', 'Month', 'Day', 'DayOfWeek', f'{ma_window}d_MA', 'RSI', 'MACD'] + [f'Close_lag_{lag}' for lag in range(1, 6)]
X = stock_data[features]
y = stock_data['Close']

# Normalize the features and target variable
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, shuffle=False)

# Create a Support Vector Regression model
svr_model = SVR(kernel='rbf')

# Set up the parameter grid for GridSearchCV
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': [0.001, 0.01, 0.1],
    'epsilon': [0.01, 0.1, 1]
}

# Perform GridSearchCV to find the best parameters
grid_search = GridSearchCV(svr_model, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2)
grid_search.fit(X_train, y_train)

# Best estimator from grid search
best_svr_model = grid_search.best_estimator_

# Make Predictions and Evaluate the Model
y_pred_train_svr = best_svr_model.predict(X_train)
y_pred_test_svr = best_svr_model.predict(X_test)

# Inverse transform the predictions to original scale
y_pred_train_svr_orig = scaler_y.inverse_transform(y_pred_train_svr.reshape(-1, 1)).flatten()
y_pred_test_svr_orig = scaler_y.inverse_transform(y_pred_test_svr.reshape(-1, 1)).flatten()
y_train_orig = scaler_y.inverse_transform(y_train.reshape(-1, 1)).flatten()
y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

mse_train_svr = mean_squared_error(y_train_orig, y_pred_train_svr_orig)
mse_test_svr = mean_squared_error(y_test_orig, y_pred_test_svr_orig)

print("Best Parameters for SVR:", grid_search.best_params_)
print("Mean Squared Error (Train - SVR):", mse_train_svr)
print("Mean Squared Error (Test - SVR):", mse_test_svr)

# Plot Results
plt.figure(figsize=(14, 7))

# Plot training data
plt.plot(stock_data.index[:len(y_train)], y_train_orig, label='Actual Train', color='blue')
plt.plot(stock_data.index[:len(y_train)], y_pred_train_svr_orig, label='Predicted Train (SVR)', color='cyan')

# Plot test data
plt.plot(stock_data.index[len(y_train):], y_test_orig, label='Actual Test', color='orange')
plt.plot(stock_data.index[len(y_train):], y_pred_test_svr_orig, label='Predicted Test (SVR)', color='red')

plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title(f'{ticker} Stock Price Prediction')
plt.legend()
plt.show()
