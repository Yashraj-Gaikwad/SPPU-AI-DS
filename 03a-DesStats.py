'''
Viva Questions:-



'''

import pandas as pd

# Assuming we have a DataFrame called 'df' with columns 'age_group' and 'income'
# Here's a sample DataFrame for demonstration purposes:
data = {
    'age_group': ['Young', 'Young', 'Middle-aged', 'Middle-aged', 'Old', 'Old'],
    'income': [30000, 35000, 50000, 60000, 45000, 55000]
}

df = pd.DataFrame(data)

# Calculate summary statistics of income grouped by age groups
summary_stats = df.groupby('age_group')['income'].describe()

# Print summary statistics
print(summary_stats)

# Create a list containing a numeric value for each response to the categorical variable
# For simplicity, we'll just calculate the mean income for each age group
numeric_values = df.groupby('age_group')['income'].mean().tolist()

print("Numeric values for each response to the categorical variable:")

print(numeric_values)



