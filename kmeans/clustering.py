import numpy as np

#Funcion inicializadora de centroides
def centroids(data, dim, k):
    centroids = []
    if dim == 2:
        for i in range(k):
            centroids.append([np.random.uniform(data['x'].min(), data['x'].max()), np.random.uniform(data['y'].min(), data['y'].max())])
    if dim == 3:
        for i in range(k):
            centroids.append([np.random.uniform(data['x'].min(), data['x'].max()), 
                              np.random.uniform(data['y'].min(), data['y'].max()),
                              np.random.uniform(data['z'].min(), data['z'].max())])
    return np.array(centroids)

### ii) Creamos los clusters

#### Funcion clustering
def cluster(data, dim, centroids):
    clusters = [[] for _ in range(centroids.shape[0])]
    if dim == 2:
        points = data[['x', 'y']].to_numpy()
    if dim == 3:
        points = data[['x', 'y', 'z']].to_numpy()
    for point in points:
        # Calculamos la distancia del punto a los centroides
        distances = np.linalg.norm(centroids - point, axis=1)
        # Encontramos el indice del centroide mas cercano
        nearest_index = np.argmin(distances)
        # Agregamos el punto al cluster (lista)
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