'''
Data Loading, Storage and File Formats

Problem Statement: Analyzing Sales Data from Multiple File Formats

Dataset: Sales data in multiple file formats (e.g., CSV, Excel, JSON)

Description: The goal is to load and analyze sales data from different file formats, including
CSV, Excel, and JSON, and perform data cleaning, transformation, and analysis on the
dataset.

Tasks to Perform:
Obtain sales data files in various formats, such as CSV, Excel, and JSON.

1. Load the sales data from each file format into the appropriate data structures or
dataframes.

2. Explore the structure and content of the loaded data, identifying any inconsistencies,
missing values, or data quality issues.

3. Perform data cleaning operations, such as handling missing values, removing
duplicates, or correcting inconsistencies.

4. Convert the data into a unified format, such as a common dataframe or data structure,
to enable seamless analysis.

5. Perform data transformation tasks, such as merging multiple datasets, splitting
columns, or deriving new variables.

6. Analyze the sales data by performing descriptive statistics, aggregating data by
specific variables, or calculating metrics such as total sales, average order value, or
product category distribution.

7. Create visualizations, such as bar plots, pie charts, or box plots, to represent the sales
data and gain insights into sales trends, customer behavior, or product performance.

viva questions
1) explain EDA

'''

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

# Step 1: Create sample sales data files (if not already created)
# CSV
csv_data = {
    'OrderID': [1, 2, 3, 4],
    'Product': ['A', 'B', 'C', 'D'],
    'Quantity': [10, 20, np.nan, 15],
    'Price': [100.0, 200.0, 150.0, 300.0],
}
csv_df = pd.DataFrame(csv_data)
csv_df.to_csv('sales_data.csv', index=False)

# Excel
excel_data = {
    'OrderID': [5, 6, 7],
    'Product': ['E', 'F', 'G'],
    'Quantity': [np.nan, 25, 30],
    'Price': [400.0, 500.0, 600.0],
}
excel_df = pd.DataFrame(excel_data)
excel_df.to_excel('sales_data.xlsx', index=False)

# JSON
json_data = [
    {'OrderID': 8, 'Product': 'H', 'Quantity': 5, 'Price': 700.0},
    {'OrderID': 9, 'Product': 'I', 'Quantity': np.nan, 'Price': 800.0},
]
with open('sales_data.json', 'w') as json_file:
    json.dump(json_data, json_file)

# Step 2: Load the sales data from each file format
csv_sales = pd.read_csv('sales_data.csv')
excel_sales = pd.read_excel('sales_data.xlsx')
json_sales = pd.read_json('sales_data.json')

# Step 3: Explore the structure and content of the loaded data
print("CSV Sales Data:")
print(csv_sales)
print("\nExcel Sales Data:")
print(excel_sales)
print("\nJSON Sales Data:")
print(json_sales)

# Check for inconsistencies and missing values
print("\nMissing values in CSV:\n", csv_sales.isnull().sum())
print("Missing values in Excel:\n", excel_sales.isnull().sum())
print("Missing values in JSON:\n", json_sales.isnull().sum())

# Step 4: Perform data cleaning operations
# Fill missing values with a default value (e.g., quantity = 0)
csv_sales['Quantity'] = csv_sales['Quantity'].fillna(0)
excel_sales['Quantity'] = excel_sales['Quantity'].fillna(0)
json_sales['Quantity'] = json_sales['Quantity'].fillna(0)

# Remove duplicates if any (not applicable in this example but good practice)
csv_sales.drop_duplicates(inplace=True)
excel_sales.drop_duplicates(inplace=True)
json_sales.drop_duplicates(inplace=True)

# Step 5: Convert the data into a unified format
unified_sales = pd.concat([csv_sales, excel_sales, json_sales], ignore_index=True)

# Step 6: Perform basic analysis on the cleaned and transformed dataset
unified_sales['TotalPrice'] = unified_sales['Quantity'] * unified_sales['Price']

total_revenue = unified_sales['TotalPrice'].sum()

product_summary = unified_sales.groupby('Product').agg({'Quantity': 'sum', 'TotalPrice': 'sum'})

print("\nUnified Sales Data:")
print(unified_sales)
print(f"\nTotal Revenue: ${total_revenue:.2f}")
print("\nProduct Summary:")
print(product_summary)

# Step 7: Create visualizations
plt.figure(figsize=(10,5))
product_summary['TotalPrice'].plot(kind='bar')
plt.title('Total Sales by Product')
plt.xlabel('Product')
plt.ylabel('Total Sales ($)')
plt.xticks(rotation=360)
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(10,5))
unified_sales['Product'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Product Distribution')
plt.ylabel('')
plt.show()


