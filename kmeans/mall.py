import numpy as np
import pandas as pd
import new_clustering as clust
import graphers as grph
import metric

mall_data = pd.read_csv("data/Mall_Customers.csv")
# Mostrar las primeras filas del DataFrame
print("Datos de Mall Customers:")
print(mall_data.head())
# Eliminar las variables categóricas
mall_data_numeric = mall_data.select_dtypes(include=[np.number])
mall_data_numeric = mall_data_numeric.drop(columns=['CustomerID'])
# Centralizar las variables restantes
mall_data_centered = mall_data_numeric - mall_data_numeric.mean()
# Dividir por la desviación estándar de cada columna
mall_data_standardized = mall_data_centered / mall_data_centered.std()
# Mostrar las primeras filas del DataFrame estandarizado
print("Datos estandarizados de Mall Customers:")
print(mall_data_standardized.head())

# Test
centros = clust.centroids(mall_data_standardized, 5)
centroides, clusters_3d = clust.kmeans(mall_data_standardized, 5, metric.euclidean)  # Ahora obtenemos clusters
grph.plot_3d(mall_data_standardized, centros)
grph.plot_3d(mall_data_standardized, centroides)

print("Inercia 3d:", metric.inertia(clusters_3d, centroides, metric.euclidean))  # Usamos los clusters correctos
