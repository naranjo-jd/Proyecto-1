import time
import pandas as pd
import kmeans
import metric
import matplotlib.pyplot as plt

## b) Estudio estadístico de las variables
def stats(data):
    print("Estudio estadistico de los datos 2D:")
    print(data.describe())

# Definir métricas
metrics = {
    "Euclidean": metric.euclidean,
    "Manhattan": metric.manhattan,
    "Chebyshev": metric.chebyshev
}

def analysis(data, k, seed=42):
    results = []
    
    for name, metric_func in metrics.items():
        start_time = time.time()

        # Ejecutar K-Means con la métrica actual
        centroids, clusters = kmeans.Kmeans(data, k, metric_func, seed)

        # Calcular inercia con la métrica usada
        inertia_self = metric.inertia(clusters, centroids, metric_func)

        # Calcular inercia con Euclidiana para comparación
        inertia_euclidean = metric.inertia(clusters, centroids, metric.euclidean)

        # Medir tiempo de ejecución
        exec_time = time.time() - start_time

        results.append((name, inertia_self, inertia_euclidean, exec_time))

    return results

def compare(results):
    # Crear DataFrame con resultados
    df_results = pd.DataFrame(results, columns=["Métrica", "Inercia Propia", "Inercia Euclidiana", "Tiempo"])
    
    # Asegurar que la columna "Métrica" sea el índice
    df_results.set_index("Métrica", inplace=True)

    # Crear dos subgráficos
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Gráfico de inercia
    df_results[["Inercia Propia", "Inercia Euclidiana"]].plot(kind="bar", ax=axes[0])
    axes[0].set_title("Comparación de inercia en K-Means")
    axes[0].set_ylabel("Inercia")
    axes[0].grid(True)

    # Gráfico de tiempo
    df_results[["Tiempo"]].plot(kind="bar", ax=axes[1], color='orange')
    axes[1].set_title("Tiempo de ejecución")
    axes[1].set_ylabel("Tiempo (s)")
    axes[1].grid(True)

    # Ajustar diseño
    plt.tight_layout()
    plt.show()