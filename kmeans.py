import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data_2d = pd.read_csv("data_2d.csv")
data_3d = pd.read_csv("data_3d.csv")

# Punto 1

## Inicializamos centroides

# Fijar la semilla
np.random.seed(42)

# Funcion que retorna una lista de k puntos aleatorios en la data
def init_centroids(data, k):
    centroids = []
    rand_points = np.random.randint(0, data.shape[0] + 1, k)
    for i in rand_points:
        centroids.append([data.loc[i, 'x'], data.loc[i, 'y']])
    return np.array(centroids)

# Test con k=5
centroids = init_centroids(data_2d, 5)
plt.scatter(data_2d['x'], data_2d['y'], color='blue', label='Data Points', alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], color='red', marker='x', s=200, label='Centroids')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('K-Means Initialization')
plt.legend()
plt.show()

# Funcion clustering
def cluster(data, centroids):
    points = data[['x', 'y']].to_numpy()
    clusters = {i: [] for i in range(len(centroids))}
    distances = np.linalg.norm(points[:, np.newaxis] - centroids, axis=2)
    for idx, point in enumerate(points):
        closest_centroid = np.argmin(distances[idx])
        clusters[closest_centroid].append(point)
    return clusters


# Test alternativa clustering
def clust(data, centroids):
    points = data
    clusters = []
    for point in points:
        distances = []
        for centroid in centroids:
            distances.append(np.linalg.norm(point - centroid))
        clusters.append(distances.index(min(distances)))
        distances.clear
    return clusters
# Ejemplo data
data = np.array([
    [2, 3],  # Point 0
    [3, 3],  # Point 1
    [3, 4],  # Point 2
    [5, 8],  # Point 3
    [6, 8],  # Point 4
    [7, 7],  # Point 5
    [8, 2],  # Point 6
    [7, 3],  # Point 7
    [8, 3],  # Point 8
    [7, 2]   # Point 9
])
initial_centroids = np.array([
    [3, 4],
    [8, 3],
    [6, 8]
])
print(clust(data, initial_centroids))
# output esperado: expected_assignments = [0, 0, 0, 2, 2, 2, 1, 1, 1, 1]



