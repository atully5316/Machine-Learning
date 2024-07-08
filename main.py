import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the California Housing dataset
california = fetch_california_housing()
data = pd.DataFrame(california.data, columns=california.feature_names)
data['MedHouseVal'] = california.target

# Display the first few rows of the dataset
print(data.head())

# Check for missing values
print(data.isnull().sum())

# Split the data into features (X) and target (y)
X = data.drop('MedHouseVal', axis=1)
y = data['MedHouseVal']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Visualize the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.show()

# Plotting a pairplot to see relationships
sns.pairplot(data, x_vars=['MedInc', 'AveRooms', 'AveOccup'], y_vars='MedHouseVal', height=5, aspect=0.7, kind='scatter')
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
new_data = np.array([[8.3252, 41.0, 6.98412698, 1.02380952, 322.0, 2.55555556, 37.88, -122.23]])
predicted_price = model.predict(new_data)
print("Predicted Price:", predicted_price)
