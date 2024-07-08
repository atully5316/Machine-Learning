import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the Boston Housing dataset
boston = load_boston()
data = pd.DataFrame(boston.data, columns=boston.feature_names)
data['PRICE'] = boston.target

# Display the first few rows of the dataset
print(data.head())

# Check for missing values
print(data.isnull().sum())

# Split the data into features (X) and target (y)
X = data.drop('PRICE', axis=1)
y = data['PRICE']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Visualize the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.show()

# Plotting a pairplot to see relationships
sns.pairplot(data, x_vars=['RM', 'LSTAT', 'PTRATIO'], y_vars='PRICE', height=5, aspect=0.7, kind='scatter')
plt.show()

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Print the model coefficients
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Plot actual vs predicted values
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.show()

# Example: Predicting the price for a new set of features
new_data = np.array([[0.1, 20, 3, 0, 0.5, 6, 65, 4, 2, 300, 15, 400, 5]])
predicted_price = model.predict(new_data)
print("Predicted Price:", predicted_price)
