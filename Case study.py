# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import joblib

# Step 2: Data Collection
# Load the dataset
url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
data = pd.read_csv(url)

# Display the first few rows of the dataset
print(data.head())

# Step 3: Data Cleaning
# Check for missing values
print("Missing values:\n", data.isnull().sum())

# Check for duplicates
data = data.drop_duplicates()

# Step 4: Exploratory Data Analysis (EDA)
# Plot pairplot for initial EDA
sns.pairplot(data, diag_kind='kde')
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Step 5: Feature Engineering
# Drop the target variable 'medv' from features
X = data.drop(columns=['medv'])
y = data['medv']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Model Building
# Initialize the Random Forest Regressor
model = RandomForestRegressor(random_state=42)

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model from GridSearchCV
best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

# Step 7: Model Evaluation
# Predictions on test set
y_pred = best_model.predict(X_test)

# Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Plot Actual vs Predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.show()

# Step 8: Model Deployment
# Save the model using joblib
joblib.dump(best_model, 'house_price_model.pkl')
print("Model saved as 'house_price_model.pkl'")

# Example of loading and using the saved model for predictions
loaded_model = joblib.load('house_price_model.pkl')
example_data = X_test.iloc[0:1]  # Predicting for the first sample
predicted_price = loaded_model.predict(example_data)
print(f"Predicted price for example data: {predicted_price[0]}")
