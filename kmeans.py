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


