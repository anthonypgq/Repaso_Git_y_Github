import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Definición del modelo de ecuaciones diferenciales del nanodron


def nanodrone_model(state, t, alpha, beta, gamma):
    x, y, z = state
    dxdt = alpha * (y - x)
    dydt = x * (beta - z) - y
    dzdt = x * y - gamma * z
    return [dxdt, dydt, dzdt]

# Método de Euler para resolver el sistema de ecuaciones diferenciales


def euler_method(f, y0, t, args=()):
    n = len(t)
    y = np.zeros((n, len(y0)))
    y[0] = y0
    for i in range(1, n):
        dt = t[i] - t[i - 1]
        dy = np.array(f(y[i - 1], t[i - 1], *args))
        y[i] = y[i - 1] + dt * dy
    return y


# Parámetros del sistema del nanodron
alpha = 0.1
beta = 0.1
gamma = 0.1

# Condiciones iniciales
initial_conditions_A = [1.0, 1.0, 1.0]
initial_conditions_B = [0.9, 0.9, 0.9]

# Intervalo de tiempo
t = np.linspace(0, 10, 1000)

# Resolución del sistema utilizando el método de Euler
solution_A = euler_method(
    nanodrone_model, initial_conditions_A, t, args=(alpha, beta, gamma))
solution_B = euler_method(
    nanodrone_model, initial_conditions_B, t, args=(alpha, beta, gamma))

# Configuración de la figura y el eje para la animación
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(min(solution_A[:, 0].min(), solution_B[:, 0].min()), max(
    solution_A[:, 0].max(), solution_B[:, 0].max()))
ax.set_ylim(min(solution_A[:, 1].min(), solution_B[:, 1].min()), max(
    solution_A[:, 1].max(), solution_B[:, 1].max()))
ax.set_zlim(min(solution_A[:, 2].min(), solution_B[:, 2].min()), max(
    solution_A[:, 2].max(), solution_B[:, 2].max()))
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Trayectoria del Nanodron')

# Inicialización de las líneas para las trayectorias
line_A, = ax.plot([], [], [], label='Posición A inicial')
line_B, = ax.plot([], [], [], label='Posición B inicial')
ax.legend()

# Función de inicialización para la animación


def init():
    line_A.set_data([], [])
    line_A.set_3d_properties([])
    line_B.set_data([], [])
    line_B.set_3d_properties([])
    return line_A, line_B

# Función de actualización para la animación


def update(num):
    line_A.set_data(solution_A[:num, 0], solution_A[:num, 1])
    line_A.set_3d_properties(solution_A[:num, 2])
    line_B.set_data(solution_B[:num, 0], solution_B[:num, 1])
    line_B.set_3d_properties(solution_B[:num, 2])
    return line_A, line_B


# Creación de la animación
ani = FuncAnimation(fig, update, frames=len(
    t), init_func=init, blit=True, interval=10, repeat=False)

plt.show()
