import pandas as pd
import kmeans
import graphers as grph
import metric

data_3d = pd.read_csv("data/data_3d.csv")

centros = kmeans.centroids(data_3d, 5, 42)
centroides, clusters_3d = kmeans.Kmeans(data_3d, 5, metric.euclidean, 42)  # Ahora obtenemos clusters
grph.plot_3d(data_3d, centros)
grph.plot_3d(data_3d, centroides)

print("Inercia 3d:", metric.inertia(clusters_3d, centroides, metric.euclidean)) 