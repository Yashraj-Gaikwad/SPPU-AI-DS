'''
Data Aggregation

Problem Statement: Analyzing Sales Performance by Region in a Retail Company

Dataset: "Retail_Sales_Data.csv"

Description: The dataset contains information about sales transactions in a retail company. It
includes attributes such as transaction date, product category, quantity sold, and sales
amount. The goal is to perform data aggregation to analyze the sales performance by region
and identify the top-performing regions.

Tasks to Perform:

1. Import the "Retail_Sales_Data.csv" dataset.

2. Explore the dataset to understand its structure and content.

3. Identify the relevant variables for aggregating sales data, such as region, sales
amount, and product category.

4. Group the sales data by region and calculate the total sales amount for each region.

5. Create bar plots or pie charts to visualize the sales distribution by region.

6. Identify the top-performing regions based on the highest sales amount.

7. Group the sales data by region and product category to calculate the total sales
amount for each combination.

8. Create stacked bar plots or grouped bar plots to compare the sales amounts across
different regions and product categories.

'''

import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'retail_sales_data.csv'  # Adjust the path as necessary
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print("Initial Data:")
print(data.head())

# Check for missing values and data types
print("\nData Info:")
print(data.info())
print("\nMissing Values:")
print(data.isnull().sum())

# Clean column names (if necessary)
data.columns = data.columns.str.strip().str.replace(' ', '_').str.replace(r'[^a-zA-Z0-9_]', '', regex=True)

# Convert 'invoice_date' to datetime format
data['invoice_date'] = pd.to_datetime(data['invoice_date'], format='%d/%m/%Y', errors='coerce')

# Check for duplicates
data.drop_duplicates(inplace=True)

# Calculate total sales amount for each transaction
data['total_sales'] = data['quantity'] * data['price']

# Group by shopping mall and sum total sales amount
sales_by_mall = data.groupby('shopping_mall')['total_sales'].sum().reset_index()
sales_by_mall = sales_by_mall.sort_values(by='total_sales', ascending=False)

# Bar plot for sales distribution by shopping mall
plt.figure(figsize=(10, 6))
plt.bar(sales_by_mall['shopping_mall'], sales_by_mall['total_sales'], color='skyblue')
plt.title('Sales Distribution by Shopping Mall')
plt.xlabel('Shopping Mall')
plt.ylabel('Total Sales Amount')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Identify top-performing shopping malls
top_performing_malls = sales_by_mall.head(5)
print("\nTop Performing Shopping Malls:")
print(top_performing_malls)

# Group by shopping mall and category, summing total sales amounts
sales_by_category_mall = data.groupby(['shopping_mall', 'category'])['total_sales'].sum().unstack(fill_value=0)

# Stacked bar plot for sales by shopping mall and category
sales_by_category_mall.plot(kind='bar', stacked=True, figsize=(12, 7))
plt.title('Sales Amount by Shopping Mall and Product Category')
plt.xlabel('Shopping Mall')
plt.ylabel('Total Sales Amount')
plt.legend(title='Product Category')
plt.tight_layout()
plt.show()

