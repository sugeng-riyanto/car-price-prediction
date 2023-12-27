import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Generating random data for demonstration purposes
np.random.seed(42)

# Random features: Year, Mileage, Horsepower, Condition
# Generating 1000 samples
num_samples = 1000

years = np.random.randint(2000, 2023, num_samples)  # Random year between 2000 and 2022
mileage = np.random.randint(10000, 100000, num_samples)  # Random mileage between 10000 and 100000
horsepower = np.random.randint(100, 500, num_samples)  # Random horsepower between 100 and 500
condition = np.random.randint(1, 6, num_samples)  # Random condition between 1 and 5

# Generating target variable: Price (assuming a simple relationship)
prices = 15000 + (years - 2000) * 1000 - (mileage / 10000) - (horsepower / 100) + (condition * 500)

# Creating a feature matrix
X = np.column_stack((years, mileage, horsepower, condition))

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, prices, test_size=0.2, random_state=42)

# Training a Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"Train R^2 score: {train_score:.2f}")
print(f"Test R^2 score: {test_score:.2f}")

# Save the trained model
joblib.dump(model, 'car_price_prediction_model.pkl')
