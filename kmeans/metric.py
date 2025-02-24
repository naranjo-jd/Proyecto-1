import numpy as np

def euclidean(x,y):
    return np.linalg.norm(np.array(x) - np.array(y))

def manhattan(x,y):
    return sum(abs(a - b) for a, b in zip(x, y))

def chebyshev(x,y):
    return max(abs(a - b) for a, b in zip(x, y))

def inertia(clusters, centroids, metric, q):
    inert = 0
    for i, cluster in enumerate(clusters):
        for point in cluster:
            inert += metric(point, centroids[i])**q
    return inert
