'''
Predict the price of the Uber ride from a given pickup point to the agreed drop-off
location. Perform following tasks:
1. Pre-process the dataset.
2. Identify outliers.
3. Check the correlation.
4. Implement linear regression and ridge, Lasso regression models.
5. Evaluate the models and compare their respective scores like R2, RMSE, etc.
Dataset link: https://www.kaggle.com/datasets/yasserh/uber-fares-dataset


'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv('uber.csv') 

# Convert datetime
data['pickup_datetime'] = pd.to_datetime(data['pickup_datetime'])

# Drop unnecessary columns
data.drop(columns=['key'], inplace=True)

# Function to detect outliers using IQR
def detect_outliers(df):
    Q1 = df['fare_amount'].quantile(0.25)   # first quartile
    Q3 = df['fare_amount'].quantile(0.75)   # third quartile
    IQR = Q3 - Q1   # Inter quartile range
    # Lower and upper quartile
    return df[(df['fare_amount'] < (Q1 - 1.5 * IQR)) | (df['fare_amount'] > (Q3 + 1.5 * IQR))]

# Identify outliers
outliers = detect_outliers(data)
print("Outliers:\n", outliers)

# Check the correlation between features and target variable
correlation_matrix = data.corr()
print("Correlation with fare_amount:\n", correlation_matrix['fare_amount'])

# Prepare features and target variable
X = data[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count']]
y = data['fare_amount']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    "Linear Regression": LinearRegression(),    # Basic Linear Regression
    "Ridge Regression": Ridge(alpha=1),         # Regression with L2 regularization
    "Lasso Regression": Lasso(alpha=1)          # Regression with L1 regularization
}

# Train models and evaluate
# Iterate through each model
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # RMSE - how far off the relations are from actual values
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    # R2 - how well the model explains variability
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {'RMSE': rmse, 'R^2': r2}

# Print model evaluation results
print("\nModel Evaluation Results:")
for model_name, metrics in results.items():
    print(f"{model_name}: RMSE = {metrics['RMSE']:.2f}, R^2 = {metrics['R^2']:.2f}")
