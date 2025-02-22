import pandas as pd
import clustering as clust
import kmeans
import graphics

data_2d = pd.read_csv("data_2d.csv")
data_3d = pd.read_csv("data_3d.csv")

# Test 2d
centros = clust.centroids(data_2d, 2, 5)
centroides = kmeans(data_2d, 2, 5)
plot_2d(data_2d, centros)
plot_2d(data_2d, centroides)

# Test 3d
centros = init_centroids_3d(data_3d, 5)
centroides = kmeans(data_3d, 3, 5)
plot_3d(data_3d, centros)
plot_3d(data_3d, centroides)
