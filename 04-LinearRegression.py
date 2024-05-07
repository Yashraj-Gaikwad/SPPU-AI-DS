'''
Viva Questions:-

Linear regression is a fundamental statistical technique used for modeling the relationship between a dependent variable (often denoted as YY) and one or more independent variables (often denoted as XX).
 It assumes that there is a linear relationship between the independent variables and the dependent variable. 
 In essence, linear regression attempts to fit a straight line to the data that best represents the relationship between the variables.

the mean squared error (MSE) between the actual target values and the predicted target values.
 MSE is a measure of the average squared difference between the observed values and the predicted values in a regression problem. 
It is commonly used as a metric to evaluate the performance of regression models.

Dataset columns

1) CRIM - Per capita crime rate by town
... 

'''


# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the Boston Housing dataset
boston = pd.read_csv("HousingData.csv")

# Create a DataFrame with the feature variables
df = pd.DataFrame(data, columns=boston.feature_names)

# Add the target variable 'PRICE' to the DataFrame
df['PRICE'] = boston.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.drop('PRICE', axis=1), df['PRICE'], test_size=0.2, random_state=42)

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate Mean Squared Error (MSE) to evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)
