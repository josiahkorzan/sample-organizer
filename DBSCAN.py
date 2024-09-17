import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# Generate random data
X = np.random.rand(100, 2)  # 100 points in 2D

# Apply DBSCAN clustering
dbscan = DBSCAN(eps=0.1, min_samples=3)  # Adjust eps and min_samples as needed
dbscan.fit(X)

# Get the cluster labels
labels = dbscan.labels_

# Visualize the clusters
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.show()
