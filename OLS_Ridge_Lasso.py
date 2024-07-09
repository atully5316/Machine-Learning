import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
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

# Adding moving average feature
stock_data['20d_MA'] = stock_data['Close'].rolling(window=20).mean()

# Adding RSI and MACD features
stock_data['RSI'] = ta.momentum.RSIIndicator(close=stock_data['Close'], window=14).rsi()
stock_data['MACD'] = ta.trend.MACD(close=stock_data['Close']).macd()

# Creating lag features
for lag in range(1, 6):
    stock_data[f'Close_lag_{lag}'] = stock_data['Close'].shift(lag)

# Drop rows with NaN values created by lag features and moving average
stock_data.dropna(inplace=True)

# Define features (X) and target (y)
features = ['Open', 'High', 'Low', 'Volume', 'Year', 'Month', 'Day', 'DayOfWeek', '20d_MA', 'RSI', 'MACD'] + [f'Close_lag_{lag}' for lag in range(1, 6)]
X = stock_data[features]
y = stock_data['Close']

# Normalize the features and target variable
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, shuffle=False)

# Create a Linear Regression model (OLS)
lr_model = LinearRegression()

# Create Ridge and Lasso models
ridge_model = Ridge()
lasso_model = Lasso()

# Fit the models
lr_model.fit(X_train, y_train)
ridge_model.fit(X_train, y_train)
lasso_model.fit(X_train, y_train)

# Make Predictions and Evaluate the Models
y_pred_train_lr = lr_model.predict(X_train)
y_pred_test_lr = lr_model.predict(X_test)

y_pred_train_ridge = ridge_model.predict(X_train)
y_pred_test_ridge = ridge_model.predict(X_test)

y_pred_train_lasso = lasso_model.predict(X_train)
y_pred_test_lasso = lasso_model.predict(X_test)

# Inverse transform the predictions to original scale
y_pred_train_lr_orig = scaler_y.inverse_transform(y_pred_train_lr.reshape(-1, 1)).flatten()
y_pred_test_lr_orig = scaler_y.inverse_transform(y_pred_test_lr.reshape(-1, 1)).flatten()

y_pred_train_ridge_orig = scaler_y.inverse_transform(y_pred_train_ridge.reshape(-1, 1)).flatten()
y_pred_test_ridge_orig = scaler_y.inverse_transform(y_pred_test_ridge.reshape(-1, 1)).flatten()

y_pred_train_lasso_orig = scaler_y.inverse_transform(y_pred_train_lasso.reshape(-1, 1)).flatten()
y_pred_test_lasso_orig = scaler_y.inverse_transform(y_pred_test_lasso.reshape(-1, 1)).flatten()

y_train_orig = scaler_y.inverse_transform(y_train.reshape(-1, 1)).flatten()
y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

mse_train_lr = mean_squared_error(y_train_orig, y_pred_train_lr_orig)
mse_test_lr = mean_squared_error(y_test_orig, y_pred_test_lr_orig)

mse_train_ridge = mean_squared_error(y_train_orig, y_pred_train_ridge_orig)
mse_test_ridge = mean_squared_error(y_test_orig, y_pred_test_ridge_orig)

mse_train_lasso = mean_squared_error(y_train_orig, y_pred_train_lasso_orig)
mse_test_lasso = mean_squared_error(y_test_orig, y_pred_test_lasso_orig)

print("Mean Squared Error (Train - LR):", mse_train_lr)
print("Mean Squared Error (Test - LR):", mse_test_lr)

print("Mean Squared Error (Train - Ridge):", mse_train_ridge)
print("Mean Squared Error (Test - Ridge):", mse_test_ridge)

print("Mean Squared Error (Train - Lasso):", mse_train_lasso)
print("Mean Squared Error (Test - Lasso):", mse_test_lasso)

# Plot Results
plt.figure(figsize=(14, 7))

# Plot training data
plt.plot(stock_data.index[:len(y_train)], y_train_orig, label='Actual Train', color='blue')
plt.plot(stock_data.index[:len(y_train)], y_pred_train_lr_orig, label='Predicted Train (LR)', color='cyan')
plt.plot(stock_data.index[:len(y_train)], y_pred_train_ridge_orig, label='Predicted Train (Ridge)', color='green')
plt.plot(stock_data.index[:len(y_train)], y_pred_train_lasso_orig, label='Predicted Train (Lasso)', color='purple')

# Plot test data
plt.plot(stock_data.index[len(y_train):], y_test_orig, label='Actual Test', color='orange')
plt.plot(stock_data.index[len(y_train):], y_pred_test_lr_orig, label='Predicted Test (LR)', color='red')
plt.plot(stock_data.index[len(y_train):], y_pred_test_ridge_orig, label='Predicted Test (Ridge)', color='brown')
plt.plot(stock_data.index[len(y_train):], y_pred_test_lasso_orig, label='Predicted Test (Lasso)', color='pink')

plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title(f'{ticker} Stock Price Prediction with Linear Regression, Ridge, and Lasso (2000-2023)')
plt.legend()
plt.show()
