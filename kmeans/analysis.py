import time
import numpy as np
import pandas as pd
import clustering as clust
import metric
import matplotlib.pyplot as plt

# Cargar datos
data_2d = pd.read_csv("data/data_2d.csv")
data_3d = pd.read_csv("data/data_3d.csv")

# Definir métricas
metrics = {
    "Euclidean": metric.euclidean,
    "Manhattan": metric.manhattan,
    "Chebyshev": metric.chebyshev
}

results = []

for name, metric_func in metrics.items():
    start_time = time.time()

    # Ejecutar K-Means con la métrica
    centroides, clusters = clust.kmeans(data_2d, 2, 5, metric_func)

    # Calcular inercia con la métrica usada
    inertia_self = metric.inertia(clusters, centroides, metric_func)

    # Calcular inercia Euclidiana para comparación
    inertia_euclidean = metric.inertia(clusters, centroides, metric.euclidean)

    # Medir tiempo de ejecución
    exec_time = time.time() - start_time

    results.append((name, inertia_self, inertia_euclidean, exec_time))

    print(f"Métrica: {name}")
    print(f"Inercia propia: {inertia_self:.4f}")
    print(f"Inercia Euclidiana: {inertia_euclidean:.4f}")
    print(f"Tiempo de ejecución: {exec_time:.4f} segundos\n")

# Crear gráfico de barras para comparar inercia
df_results = pd.DataFrame(results, columns=["Métrica", "Inercia Propia", "Inercia Euclidiana", "Tiempo"])
df_results.set_index("Métrica").plot(kind="bar", figsize=(10,5))
plt.title("Comparación de métricas en K-Means")
plt.ylabel("Valor")
plt.show()