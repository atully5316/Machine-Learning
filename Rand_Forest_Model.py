import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
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

# Hyperparameter Tuning using GridSearchCV with Random Forest
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

rf_model = RandomForestRegressor(random_state=42)
grid_search_rf = GridSearchCV(estimator=rf_model, param_grid=param_grid_rf, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search_rf.fit(X_train, y_train)

# Best model from GridSearchCV
best_rf_model = grid_search_rf.best_estimator_

# Make Predictions and Evaluate the Model
y_pred_train_rf = best_rf_model.predict(X_train)
y_pred_test_rf = best_rf_model.predict(X_test)

# Inverse transform the predictions to original scale
y_pred_train_rf_orig = scaler_y.inverse_transform(y_pred_train_rf.reshape(-1, 1)).flatten()
y_pred_test_rf_orig = scaler_y.inverse_transform(y_pred_test_rf.reshape(-1, 1)).flatten()
y_train_orig = scaler_y.inverse_transform(y_train.reshape(-1, 1)).flatten()
y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

mse_train_rf = mean_squared_error(y_train_orig, y_pred_train_rf_orig)
mse_test_rf = mean_squared_error(y_test_orig, y_pred_test_rf_orig)

print("Best Parameters (RF):", grid_search_rf.best_params_)
print("Mean Squared Error (Train - RF):", mse_train_rf)
print("Mean Squared Error (Test - RF):", mse_test_rf)

# Plot Results
plt.figure(figsize=(14, 7))

# Plot training data
plt.plot(stock_data.index[:len(y_train)], y_train_orig, label='Actual Train', color='blue')
plt.plot(stock_data.index[:len(y_train)], y_pred_train_rf_orig, label='Predicted Train (RF)', color='cyan')

# Plot test data
plt.plot(stock_data.index[len(y_train):], y_test_orig, label='Actual Test', color='orange')
plt.plot(stock_data.index[len(y_train):], y_pred_test_rf_orig, label='Predicted Test (RF)', color='red')

plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title(f'{ticker} Stock Price Prediction with Random Forest (2000-2023)')
plt.legend()
plt.show()
