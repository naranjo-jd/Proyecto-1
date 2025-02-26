import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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


def plot_2d(data, centroids, col_names=None):
    if col_names is None:
        col_names = data.columns[:2]  # Usa las dos primeras columnas por defecto
    
    plt.scatter(data[col_names[0]], data[col_names[1]], color='blue', label='Data Points', alpha=0.5)
    plt.scatter(centroids[:, 0], centroids[:, 1], color='red', marker='x', s=200, label='Centroids')
    
    plt.xlabel(col_names[0])
    plt.ylabel(col_names[1])
    plt.title('K-Means Clustering (2D)')
    plt.legend()
    plt.show()

def plot_3d(data, centroids, col_names=None):
    if col_names is None:
        col_names = data.columns[:3]  # Usa las tres primeras columnas por defecto

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(data[col_names[0]], data[col_names[1]], data[col_names[2]], 
               color='blue', label='Data Points', alpha=0.5)
    ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], 
               color='red', marker='x', s=200, label='Centroids')

    ax.set_xlabel(col_names[0])
    ax.set_ylabel(col_names[1])
    ax.set_zlabel(col_names[2])
    ax.set_title('K-Means Clustering (3D)')
    ax.legend()
    plt.show()