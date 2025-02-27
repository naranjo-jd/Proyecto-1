import numpy as np
import pandas as pd
import kmeans
import graphers as grph
import metric
import analytics

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
mall_data = mall_data_centered / mall_data_centered.std()
# Mostrar las primeras filas del DataFrame estandarizado
print("Datos estandarizados de Mall Customers:")
print(mall_data.head())

# Test

k = 5
seed = 42
metrica = metric.euclidean

init_centroids = kmeans.centroids(mall_data, k, seed)
centroids, clusters = kmeans.Kmeans(mall_data, k, metrica, seed)  # Ahora obtenemos clusters
grph.plot_3d(mall_data, init_centroids)
grph.plot_3d(mall_data, centroids)

analisis = analytics.analysis(mall_data, k, seed)
analytics.compare(analisis)
