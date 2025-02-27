import pandas as pd
import kmeans
import graphers as grph
import metric
import analytics

data_3d = pd.read_csv("data/data_3d.csv")
k = 5
seed = 42
metrica = metric.euclidean


init_centroids = kmeans.centroids(data_3d, k, seed)
centroids, clusters = kmeans.Kmeans(data_3d, k, metrica, seed)  # Ahora obtenemos clusters
grph.plot_3d(data_3d, init_centroids)
grph.plot_3d(data_3d, centroids)

analisis = analytics.analysis(data_3d, k, seed)
analytics.compare(analisis)