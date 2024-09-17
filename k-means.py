import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Generate random data
X = np.random.rand(100, 2)  # 100 points in 2D

# Apply K-Means clustering
kmeans = KMeans(n_clusters=10)  # Specify the number of clusters
kmeans.fit(X)

# Get the cluster labels and centroids
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Visualize the clusters
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='X')
plt.show()
