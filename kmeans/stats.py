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