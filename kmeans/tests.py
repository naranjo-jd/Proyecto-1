import pandas as pd
import clustering as clust
import kmeans as Kmeans
import graphers as grph

data_2d = pd.read_csv("data_2d.csv")
data_3d = pd.read_csv("data_3d.csv")

# Test 2d
centros = clust.centroids(data_2d, 2, 5)
centroides = Kmeans.kmeans(data_2d, 2, 5)
grph.plot_2d(data_2d, centros)
grph.plot_2d(data_2d, centroides)

# Test 3d
centros = clust.centroids(data_3d, 3, 5)
centroides = Kmeans.kmeans(data_3d, 3, 5)
grph.plot_3d(data_3d, centros)
grph.plot_3d(data_3d, centroides)
