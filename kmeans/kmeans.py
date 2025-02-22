import numpy as np
import clustering as clust

def kmeans(data, dim, k, max_iter=100, tol=1e-4):
    centroids = clust.centroids(data, dim, k)
    for _ in range(max_iter):
        # Asignar puntos a los clusters más cercanos
        clusters = clust.cluster(data, dim, centroids)
        # Calcular nuevos centroides
        new_centroids = clust.update_centroids(clusters, dim)
        # Verificar convergencia (si los centroides no cambian significativamente)
        if np.linalg.norm(new_centroids - centroids) < tol:
            break
        # Actualizar centroides
        centroids = new_centroids
    return centroids