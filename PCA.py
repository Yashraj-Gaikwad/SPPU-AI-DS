'''
To use PCA Algorithm for dimensionality reduction.
You have a dataset that includes measurements for different variables on wine
(alcohol, ash, magnesium, and so on). Apply PCA algorithm & transform this data
so that most variations in the measurements of the variables are captured by a small
number of principal components so that it is easier to distinguish between red and
white wine by inspecting these principal components.
Dataset Link: https://media.geeksforgeeks.org/wp-content/uploads/Wine.csv

Viva Questions
1) Explain PCA?
Principal Component Analysis (PCA) is a statistical technique used for dimensionality reduction and data analysis.
Unsupervised
Based on Variance
Calculate Covariance Matrix
Calculates Eigenvalues and Eigenvectors
'''

# To read dataset
import pandas as pd
# to plot
import matplotlib.pyplot as plt
# Use to standardize features by removing mean and scaling to unit variance
from sklearn.preprocessing import StandardScaler
# PCA module
from sklearn.decomposition import PCA

# Load the dataset
data = pd.read_csv("Wine.csv")

# Display the first few rows of the dataset
print(data.head())

# Preprocess the data
# Separate the features and the target
X = data.drop('Customer_Segment', axis=1) # All features
y = data['Customer_Segment']    # Only customer segment

# Standardize the features
scaler = StandardScaler()   # standardize
X_scaled = scaler.fit_transform(X)  

# Apply PCA
pca = PCA(n_components=2)  # Reducing to 2 components for visualization
X_pca = pca.fit_transform(X_scaled)

# Create a DataFrame with the principal components
pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
pca_df['Target'] = y

# Visualize the principal components
plt.figure(figsize=(10, 7))

for target in pca_df['Target'].unique():
    indices = pca_df['Target'] == target
    plt.scatter(pca_df.loc[indices, 'PC1'], pca_df.loc[indices, 'PC2'], label=target)

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Wine Dataset')
plt.legend()
plt.show()
