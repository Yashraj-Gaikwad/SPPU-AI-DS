'''
Implement Agglomerative hierarchical clustering algorithm using appropriate dataset.


viva questions
1)
'''

# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Load the Iris dataset
iris = load_iris()
X = iris.data  # Features

# Perform Agglomerative Clustering
# use 3 clusters, ward - algorithm to minimize variance within clusters
agg_clustering = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = agg_clustering.fit_predict(X)

# Print cluster labels for each data point
print("Cluster Labels:", labels)

# Create a dendrogram for visualization
# linkage - computes linkage matrix using ward's method
linked = linkage(X, method='ward')

plt.figure(figsize=(10, 7))
dendrogram(linked,
           orientation='top',
           distance_sort='descending',
           show_leaf_counts=True)
plt.title('Dendrogram for Agglomerative Clustering')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()

# Scatter plot of the clusters
plt.figure(figsize=(8, 6))
# sepal length and sepal width, color based on labels, edge color is black, point sizes are 100
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow', edgecolor='k', s=100)
plt.title('Agglomerative Clustering Results')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.show()
