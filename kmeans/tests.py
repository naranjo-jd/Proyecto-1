import pandas as pd
import clustering as clust
import graphers as grph
import metric

data_2d = pd.read_csv("data_2d.csv")
data_3d = pd.read_csv("data_3d.csv")

# Test 2d
centros = clust.centroids(data_2d, 2, 5)
centroides = clust.kmeans(data_2d, 2, 5)
grph.plot_2d(data_2d, centros)
grph.plot_2d(data_2d, centroides)

print("Inercia 2d:", metric.inertia(clust.cluster(data_2d, 2, centroides), centroides, metric.euclidean, 2))

# Test 3d
centros = clust.centroids(data_3d, 3, 5)
centroides = clust.kmeans(data_3d, 3, 5)
grph.plot_3d(data_3d, centros)
grph.plot_3d(data_3d, centroides)

print("Inercia 3d:", metric.inertia(clust.cluster(data_3d, 3, centroides), centroides, metric.euclidean, 2))
