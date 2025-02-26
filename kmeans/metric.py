import numpy as np

def euclidean(x,y):
    return [np.linalg.norm(np.array(x) - np.array(y)), 2]

def manhattan(x,y):
    return [sum(abs(a - b) for a, b in zip(x, y)), 1]

def chebyshev(x,y):
    return [max(abs(a - b) for a, b in zip(x, y)), 1]

def inertia(clusters, centroids, metric):
    inert = 0
    for i, cluster in enumerate(clusters):
        for point in cluster:
            d, q = metric(point, centroids[i])
            inert += d**q
    return inert
