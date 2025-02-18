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

# (2d) Funcion que retorna una lista de k puntos aleatorios en la data 2d
def init_centroids_2d(data, k):
    centroids = []
    rand_points = np.random.randint(0, data.shape[0] + 1, k)
    for i in rand_points:
        centroids.append([data.loc[i, 'x'], data.loc[i, 'y']])
    return np.array(centroids)

# (3d) Funcion que retorna una lista de k puntos aleatorios en la data 
def init_centroids_3d(data, k):
    centroids = []
    rand_points = np.random.randint(0, data.shape[0] + 1, k)
    for i in rand_points:
        centroids.append([data.loc[i, 'x'], data.loc[i, 'y'], data.loc[i, 'z']])
    return np.array(centroids)

# (2d) Funcion clustering
def cluster_2d(data, centroids):
    points = data[['x', 'y']].to_numpy()
    clusters = [[] for _ in range(centroids.shape[0])]
    for point in points:
        distances = np.linalg.norm(centroids - point, axis=1)
        nearest_index = np.argmin(distances)
        clusters[nearest_index].append(point)
    return np.array(clusters)

# (3d) Funcion clustering
def cluster_3d(data, centroids):
    points = data[['x', 'y', 'z']].to_numpy()
    clusters = [[] for _ in range(centroids.shape[0])]
    for point in points:
        distances = np.linalg.norm(centroids - point, axis=1)
        nearest_index = np.argmin(distances)
        clusters[nearest_index].append(point)
    return np.array(clusters)

def plot_2d(data, centroids):
    plt.scatter(data['x'], data['y'], color='blue', label='Data Points', alpha=0.5)
    plt.scatter(centroids[:, 0], centroids[:, 1], color='red', marker='x', s=200, label='Centroids')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('K-Means')
    plt.legend()
    plt.show()

def plot_3d(data, centroids):
    # Create a new figure and a 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Plot data points; assuming data is a pandas DataFrame with columns 'x', 'y', and 'z'
    ax.scatter(data['x'], data['y'], data['z'], color='blue', label='Data Points', alpha=0.5)
    # Plot centroids; assuming centroids is a NumPy array with shape (n, 3)
    ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2],
               color='red', marker='x', s=200, label='Centroids')
    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # Set title and legend
    ax.set_title('K-Means Clustering (3D)')
    ax.legend()
    # Show the plot
    plt.show()

# Update centroids
def updt_centroids(clusters):
    new_centroids = []
    for cluster in clusters:
        for point in cluster:

        np.mean(cluster, axis=0)

# Lista de puntos en 2D
puntos = np.array([[2, 3], [3, 3], [3, 4]])

# Media aritmética por columnas
centroide = np.mean(puntos, axis=0)