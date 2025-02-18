# Punto 1

## a) Cargamos los datos e importamos librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
data_2d = pd.read_csv("data_2d.csv")
data_3d = pd.read_csv("data_3d.csv")

## b) Estudio estadístico de las variables
def stats(data):
    print("Estudio estadistico de los datos 2D:")
    print(data.describe())


## c) Gráfica de los datos
def scatter_2d(data):
    plt.figure(figsize=(8, 6))
    plt.scatter(data['x'], data['y'], color='blue', alpha=0.5, label='Data Points')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('2D Scatter Plot')
    plt.legend()
    plt.grid(True)
    plt.show()

def scatter_3d(data):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data['x'], data['y'], data['z'], color='blue', alpha=0.5, label='Data Points')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Scatter Plot')
    ax.legend()
    plt.show()

## d) K-means

### i) Inicializar centroides de manera aleatoria

#### Fijar la semilla
np.random.seed(42)

#### (2d) Funcion que retorna una lista de k puntos aleatorios en la data 2d
def init_centroids_2d(data, k):
    centroids = []
    rand_points = np.random.randint(0, data.shape[0] + 1, k)
    for i in rand_points:
        centroids.append([data.loc[i, 'x'], data.loc[i, 'y']])
    return np.array(centroids)

#### (3d) Funcion que retorna una lista de k puntos aleatorios en la data 3d
def init_centroids_3d(data, k):
    centroids = []
    rand_points = np.random.randint(0, data.shape[0] + 1, k)
    for i in rand_points:
        centroids.append([data.loc[i, 'x'], data.loc[i, 'y'], data.loc[i, 'z']])
    return np.array(centroids)


### ii) Creamos los clusters

#### (2d) Funcion clustering
def cluster_2d(data, centroids):
    points = data[['x', 'y']].to_numpy()
    clusters = [[] for _ in range(centroids.shape[0])]
    for point in points:
        # Calculamos la distancia del punto a los centroides
        distances = np.linalg.norm(centroids - point, axis=1)
        # Encontramos el indice del centroide mas cercano
        nearest_index = np.argmin(distances)
        # Agregamos el punto al cluster (lista)
        clusters[nearest_index].append(point)
    return clusters

#### (3d) Funcion clustering
def cluster_3d(data, centroids):
    points = data[['x', 'y', 'z']].to_numpy()
    clusters = [[] for _ in range(centroids.shape[0])]
    for point in points:
        distances = np.linalg.norm(centroids - point, axis=1)
        nearest_index = np.argmin(distances)
        clusters[nearest_index].append(point)
    return clusters


### iii) Actualizamos los centroides
def update_centroids(clusters, dim):
    new_centroids = np.zeros((len(clusters), dim))
    for i, cluster in enumerate(clusters):
        if len(cluster) > 0:
            new_centroids[i] = np.mean(cluster, axis=0)
        else:
            new_centroids[i] = np.zeros(dim)  
    return new_centroids

### iv) Iteramos los pasos anteriores

# Diccionarios con las funciones dependientes de dimension
init_centroids = {
    2: init_centroids_2d,
    3: init_centroids_3d
}
cluster_funct = {
    2: cluster_2d,
    3: cluster_3d
}

#Implementacion del algoritmo con maximo de iteraciones y tolerancia
def kmeans(data, dim, k, max_iter=100, tol=1e-4):
    centroids = init_centroids[dim](data, k)
    for _ in range(max_iter):
        # Asignar puntos a los clusters más cercanos
        clusters = cluster_funct[dim](data, centroids)
        # Calcular nuevos centroides
        new_centroids = update_centroids(clusters, dim)
        # Verificar convergencia (si los centroides no cambian significativamente)
        if np.linalg.norm(new_centroids - centroids) < tol:
            break
        # Actualizar centroides
        centroids = new_centroids
    return centroids

# Funciones graficadoras
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
    
# Test
centros = init_centroids_2d(data_2d, 5)
centroides = kmeans(data_2d, 2, 5)
plot_2d(data_2d, centros)
plot_2d(data_2d, centroides)