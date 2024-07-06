import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def funcion_prueba(x, y): 
    res = np.sqrt(x**2 + y**2)
    return res

# Generación de vectores que barren todo el dominio 
x = np.linspace(-100, 100, 1000)
y = np.linspace(-100, 100, 1000)

# Evaluación de los valores de las variables 
x_ax, y_ax = np.meshgrid(x, y)
fx = funcion_prueba(x_ax, y_ax)

# Representación de los resultados 
figure_3d = plt.figure(figsize=(8, 6))
ax = figure_3d.add_subplot(111, projection="3d")
ax.plot_surface(x_ax, y_ax, fx, cmap=cm.coolwarm)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("f(x,y)")

plt.show()
