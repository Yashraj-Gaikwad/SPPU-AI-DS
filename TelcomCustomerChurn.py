'''
Data Cleaning and Preparation

Problem Statement: Analyzing Customer Churn in a Telecommunications Company

Dataset: "Telecom_Customer_Churn.csv"

Description: The dataset contains information about customers of a telecommunications
company and whether they have churned (i.e., discontinued their services). The dataset
includes various attributes of the customers, such as their demographics, usage patterns, and
account information. The goal is to perform data cleaning and preparation to gain insights
into the factors that contribute to customer churn.

Tasks to Perform:

1. Import the "Telecom_Customer_Churn.csv" dataset.

2. Explore the dataset to understand its structure and content.

3. Handle missing values in the dataset, deciding on an appropriate strategy.

4. Remove any duplicate records from the dataset.

5. Check for inconsistent data, such as inconsistent formatting or spelling variations,
and standardize it.

6. Convert columns to the correct data types as needed.

7. Identify and handle outliers in the data.

8. Perform feature engineering, creating new features that may be relevant to
predicting customer churn.

9. Normalize or scale the data if necessary.

10. Split the dataset into training and testing sets for further analysis.

11. Export the cleaned dataset for future analysis or modeling.



'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Import the Dataset
data = pd.read_csv('Telecom_Customer_Churn.csv')

# 2. Explore the Dataset
print(data.info())
print(data.describe())
print(data.head())

# 3. Handle Missing Values
missing_values = data.isnull().sum()
print("Missing values:\n", missing_values)

# Example: Fill missing values or drop them based on analysis
data.fillna(0, inplace=True)  # Replace NaN with 0 for numerical columns, adjust as necessary

# 4. Remove Duplicate Records
data.drop_duplicates(inplace=True)

# 5. Check for Inconsistent Data
# Standardizing categorical variables (e.g., International plan, Voice mail plan)
data['International plan'] = data['International plan'].str.strip().str.lower()
data['Voice mail plan'] = data['Voice mail plan'].str.strip().str.lower()

# 6. Convert Columns to Correct Data Types
data['Churn'] = data['Churn'].astype('bool')  # Ensure Churn is boolean

# 7. Identify and Handle Outliers
# Example: Visualize outliers in Total day minutes
import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(x=data['Total day minutes'])
plt.show()

# Optionally remove outliers based on domain knowledge or statistical methods
# For example, remove rows where Total day minutes are greater than a certain threshold
threshold = data['Total day minutes'].quantile(0.95)
data = data[data['Total day minutes'] <= threshold]

# 8. Perform Feature Engineering
# Example: Create a new feature for total call minutes
data['Total call minutes'] = data[['Total day minutes', 'Total eve minutes', 'Total night minutes']].sum(axis=1)

# 9. Normalize or Scale the Data if Necessary
scaler = StandardScaler()
numerical_cols = ['Account length', 'Number vmail messages', 'Total day minutes', 
                  'Total day calls', 'Total day charge', 'Total eve minutes', 
                  'Total eve calls', 'Total eve charge', 'Total night minutes', 
                  'Total night calls', 'Total night charge', 'Total intl minutes', 
                  'Total intl calls', 'Total intl charge', 'Customer service calls']

data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

# 10. Split the Dataset into Training and Testing Sets
X = data.drop('Churn', axis=1)  # Features excluding the target variable
y = data['Churn']                # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 11. Export the Cleaned Dataset for Future Analysis or Modeling
data.to_csv('Cleaned_Telecom_Customer_Churn.csv', index=False)

