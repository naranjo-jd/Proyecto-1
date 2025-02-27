import time
import pandas as pd
import kmeans
import metric
import matplotlib.pyplot as plt

## b) Estudio estadístico de las variables
def stats(data):
    print("Estudio estadistico de los datos 2D:")
    print(data.describe())

## c) Gráfica de los datos
def scatter_2d(data):
    plt.figure(figsize=(8, 6))
    plt.scatter(data['x'], data['y'], color='blue', alpha=0.5, label='Data Points')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('2D Scatter Plot')
    plt.legend()
    plt.grid(True)
    plt.show()

def scatter_3d(data):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data['x'], data['y'], data['z'], color='blue', alpha=0.5, label='Data Points')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Scatter Plot')
    ax.legend()
    plt.show()

# Definir métricas
metrics = {
    "Euclidean": metric.euclidean,
    "Manhattan": metric.manhattan,
    "Chebyshev": metric.chebyshev
}

def analysis(data, k):
    results = []
    for name, metric_func in metrics.items():
        start_time = time.time()

        # Ejecutar K-Means con la métrica
        centroides, clusters = kmeans.Kmeans(data, k, metric_func)

        # Calcular inercia con la métrica usada
        inertia_self = metric.inertia(clusters, centroides, metric_func)

        # Calcular inercia Euclidiana para comparación
        inertia_euclidean = metric.inertia(clusters, centroides, metric.euclidean)

        # Medir tiempo de ejecución
        exec_time = time.time() - start_time

        results.append((name, inertia_self, inertia_euclidean, exec_time))

        return results

def compare(results):
    # Crear gráfico de barras para comparar inercia
    df_results = pd.DataFrame(results, columns=["Métrica", "Inercia Propia", "Inercia Euclidiana", "Tiempo"])
    df_results.set_index("Métrica").plot(kind="bar", figsize=(10,5))
    plt.title("Comparación de métricas en K-Means")
    plt.ylabel("Valor")
    plt.show()