'''
Viva Questions:-



'''

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats.mstats import winsorize

# Step 1: Load the dataset
# Assuming the dataset is stored in a CSV file named 'academic_performance.csv'
df = pd.read_csv('AcademicPerformance.csv')

# Step 2: Scan for missing values and inconsistencies
missing_values = df.isnull().sum()
print("Missing Values:\n", missing_values)

# Step 3: Handle missing values
# Example: Replace missing values in numeric columns with mean
#df.fillna(df.mean(), inplace=True)

# Step 4: Scan for outliers in numeric variables
numeric_columns = df.select_dtypes(include=np.number).columns
for column in numeric_columns:
    sns.boxplot(x=df[column])
    plt.title(f'Boxplot of {column}')
    plt.show()

# Step 5: Handle outliers
# Example: Winsorization to replace outliers with the 5th and 95th percentiles

for column in numeric_columns:
    df[column] = winsorize(df[column], limits=[0.05, 0.05])

# Step 6: Apply data transformations
# Example: Log transformation to reduce skewness of a numeric variable
df['exam_scores_log'] = np.log(df['exam_scores'] + 1)

# Visualize the transformed variable
sns.histplot(df['exam_scores_log'], kde=True)
plt.title('Histogram of Transformed Exam Scores')
plt.show()
