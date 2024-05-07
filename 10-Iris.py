'''
Viva Questions:-



'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the Iris dataset into a DataFrame
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
iris_df = pd.read_csv(url, names=column_names)

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(iris_df.head())

# List down the features and their types
print("\nFeatures and their types:")
print(iris_df.dtypes)

# Step 2: Create a histogram for each feature
print("\nHistograms for each feature:")
iris_df.hist(figsize=(10, 8))
plt.tight_layout()
plt.show()

# Step 3: Create a boxplot for each feature
print("\nBoxplots for each feature:")
plt.figure(figsize=(10, 8))
sns.boxplot(data=iris_df)
plt.show()

# Step 4: Compare distributions and identify outliers
# We can visually compare the histograms and boxplots to identify outliers or differences in distributions
