'''
Viva Questions:-

1) what is pandas


'''



import pandas as pd

# The dataset is named tested.csv and is in the current directory
# df - dataframe
df = pd.read_csv("tested.csv")

# Check for missing values
# calulates the sum of missing values from each column
missing_values = df.isnull().sum()
print("Missing values: \n", missing_values)

# Get initial statistics
description = df.describe()
print("\n Initial Statistics: \n", description)

# Variable description and types
variable_description = {
    "PassengerId: Unique",
    "Survived: Survival",
    "Gender: Male or Female",
    "Age: Age",
    "Ticket: Ticket Number",
    "Fare: Price"
}

print("Variable Descriptions: \n", variable_description)

# Dimensions of the dataframe
print("Dimensions of the dataframe", df.shape)

# Summarize types of variables
variable_types = df.dtypes
print("Types of variables: \n", variable_types)
