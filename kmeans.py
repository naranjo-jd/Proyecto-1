import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data_2d = pd.read_csv("data_2d.csv")
data_3d = pd.read_csv("data_3d.csv")

# Estudio estadistico de los datos 2D
print("Estudio estadistico de los datos 2D:")
print(data_2d.describe())

# Gráfica de los datos 2D
plt.figure(figsize=(8, 6))
plt.scatter(data_2d["x"], data_2d["y"], s=50)
plt.title("Datos 2D")

# Gráfica de los datos 3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data_3d["x"], data_3d["y"], data_3d["z"], s=50)
ax.set_title("Datos 3D")
plt.show()

# Fijar la semilla
np.random.seed(42)