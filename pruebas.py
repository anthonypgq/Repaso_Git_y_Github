import numpy as np
import matplotlib.pyplot as plt

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

# Gráfica de la solución
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(solution_A[:, 0], solution_A[:, 1],
        solution_A[:, 2], label='Posición A inicial')
ax.plot(solution_B[:, 0], solution_B[:, 1],
        solution_B[:, 2], label='Posición B inicial')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Trayectoria del Nanodron')
ax.legend()
plt.show()
