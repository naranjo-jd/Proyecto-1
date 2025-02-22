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

# Funciones graficadoras
def plot_2d(data, centroids):
    plt.scatter(data['x'], data['y'], color='blue', label='Data Points', alpha=0.5)
    plt.scatter(centroids[:, 0], centroids[:, 1], color='red', marker='x', s=200, label='Centroids')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('K-Means')
    plt.legend()
    plt.show()

def plot_3d(data, centroids):
    # Create a new figure and a 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Plot data points; assuming data is a pandas DataFrame with columns 'x', 'y', and 'z'
    ax.scatter(data['x'], data['y'], data['z'], color='blue', label='Data Points', alpha=0.5)
    # Plot centroids; assuming centroids is a NumPy array with shape (n, 3)
    ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2],
               color='red', marker='x', s=200, label='Centroids')
    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # Set title and legend
    ax.set_title('K-Means Clustering (3D)')
    ax.legend()
    # Show the plot
    plt.show()