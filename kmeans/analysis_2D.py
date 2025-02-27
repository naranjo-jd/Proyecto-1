import pandas as pd
import kmeans
import graphers as grph
import metric
import analytics

data_2d = pd.read_csv("data/data_2d.csv")
k = 5
seed = 42
metrica = metric.euclidean


init_centroids = kmeans.centroids(data_2d, k, seed)
centroids, clusters = kmeans.Kmeans(data_2d, k, metrica, seed)  # Ahora obtenemos clusters
grph.plot_2d(data_2d, init_centroids)
grph.plot_2d(data_2d, centroids)

analisis = analytics.analysis(data_2d, k, seed)
analytics.compare(analisis)