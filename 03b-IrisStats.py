'''
Viva Questions:-

The Iris dataset is a classic and widely used dataset in the field of machine learning and statistics.
It was introduced by the British statistician and biologist Ronald Fisher
in his 1936 paper "The Use of Multiple Measurements in Taxonomic Problems" as an example of discriminant analysis.
The dataset consists of 150 samples of iris flowers, each belonging to one of three species: Setosa, Versicolor, and Virginica.
'''


import pandas as pd

# Load Iris dataset
iris_df = pd.read_csv('iris.csv')

print("Size   \n", iris_df.size)
print("Shape  \n",iris_df.shape)
print("Dtypes \n",iris_df.dtypes)
print("Mean   \n",iris_df.mean)
print("Median \n",iris_df.median)


# Filter data for each target
setosa_stats = iris_df[iris_df['target'] == 'Iris-setosa'].describe()
versicolor_stats = iris_df[iris_df['target'] == 'Iris-versicolor'].describe()
virginica_stats = iris_df[iris_df['target'] == 'Iris-virginica'].describe()

# Print statistical details for each target
print("Statistical details for Iris-setosa:")
print(setosa_stats)

print("\nStatistical details for Iris-versicolor:")
print(versicolor_stats)

print("\nStatistical details for Iris-virginica:")
print(virginica_stats)
