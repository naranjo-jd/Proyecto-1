import pandas as pd
import kmeans
import graphers as grph
import metric

data_2d = pd.read_csv("data/data_2d.csv")

# Test 2D
centros = kmeans.centroids(data_2d, 5, 42)
centroides, clusters_2d = kmeans.Kmeans(data_2d, 5, metric.euclidean, 42)  # Ahora obtenemos clusters
grph.plot_2d(data_2d, centros)
grph.plot_2d(data_2d, centroides)

print("Inercia 2d:", metric.inertia(clusters_2d, centroides, metric.euclidean))  # Usamos los clusters correctos

