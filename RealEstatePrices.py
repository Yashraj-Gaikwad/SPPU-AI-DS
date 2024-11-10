'''
Data Wrangling

Problem Statement: Data Wrangling on Real Estate Market

Dataset: "real_estate_prices.csv"

Description: The dataset contains information about housing prices in a specific real estate
market. It includes various attributes such as property characteristics, location, sale prices,
and other relevant features. The goal is to perform data wrangling to gain insights into the
factors influencing housing prices and prepare the dataset for further analysis or modeling.

Tasks to Perform:
1. Import the "RealEstate_Prices.csv" dataset. Clean column names by removing spaces,
special characters, or renaming them for clarity.

2. Handle missing values in the dataset, deciding on an appropriate strategy (e.g.,
imputation or removal).

3. Perform data merging if additional datasets with relevant information are available
(e.g., neighborhood demographics or nearby amenities).

4. Filter and subset the data based on specific criteria, such as a particular time period,
property type, or location.

5. Handle categorical variables by encoding them appropriately (e.g., one-hot encoding or
label encoding) for further analysis.

6. Aggregate the data to calculate summary statistics or derived metrics such as average
sale prices by neighborhood or property type.

7. Identify and handle outliers or extreme values in the data that may affect the analysis
or modeling process.

Reduced dataset

'''

import pandas as pd

# Load the dataset
df = pd.read_csv('real_estate_prices.csv')

# Clean column names
df.columns = df.columns.str.replace(' ', '_').str.replace(r'[^\w]', '')

# Display the cleaned column names
print("Cleaned Column Names:", df.columns.tolist())

# 2. Handling Missing Values
# Check for missing values
missing_values = df.isnull().sum()
print("Missing Values:\n", missing_values)

# Convert total_sqft to numeric, forcing errors to NaN
df['total_sqft'] = pd.to_numeric(df['total_sqft'], errors='coerce')

# Fill missing values in 'society' and 'total_sqft'
df.fillna({'society': 'Unknown', 'total_sqft': df['total_sqft'].mean()}, inplace=True)

# 3. Data Merging (if applicable)
# Uncomment and modify if you have additional datasets to merge
# neighborhood_df = pd.read_csv('neighborhood_data.csv')
# df = df.merge(neighborhood_df, on='location', how='left')

# 4. Filtering and Subsetting Data
# Example: Filter properties available for sale
filtered_df = df[df['availability'] == 'Ready To Move']

# 5. Handling Categorical Variables
# One-hot encoding example for area_type and society
df = pd.get_dummies(df, columns=['area_type', 'society'], drop_first=True)

# 6. Aggregating Data
# Calculate average price by location
average_prices = df.groupby('location')['price'].mean().reset_index()
print("Average Prices by Location:\n", average_prices)

# 7. Identifying and Handling Outliers
# Example using IQR to detect outliers in price
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1

# Identify outliers
outliers = df[(df['price'] < (Q1 - 1.5 * IQR)) | (df['price'] > (Q3 + 1.5 * IQR))]

# Remove outliers from dataset
df = df[~df.index.isin(outliers.index)]

# Display final cleaned dataframe shape and sample data
print("Final DataFrame Shape:", df.shape)
print("Sample Data:\n", df.head())

