import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ipywidgets as widgets
from IPython.display import display

# Definición del modelo de ecuaciones diferenciales


def nanodrone_model(t, state, alpha, beta, gamma):
    x, y, z = state
    dxdt = alpha * (y - x)
    dydt = x * (beta - z) - y
    dzdt = x * y - gamma * z
    return np.array([dxdt, dydt, dzdt])

# Implementación del método de Euler


def euler_method(f, t0, y0, h, n_steps, alpha, beta, gamma):
    t = t0
    y = np.array(y0)
    trajectory = [y0]
    for _ in range(n_steps):
        y = y + h * f(t, y, alpha, beta, gamma)
        t += h
        trajectory.append(y)
    return np.array(trajectory)

# Función para resolver y graficar la trayectoria del nanodron


def simulate_trajectory(alpha, beta, gamma, initial_conditions, t_span, n_steps):
    t0, tf = t_span
    h = (tf - t0) / n_steps
    trajectory = euler_method(
        nanodrone_model, t0, initial_conditions, h, n_steps, alpha, beta, gamma)
    x, y, z = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]
    return x, y, z


# Parámetros iniciales y tiempo de simulación
alpha = 0.1
beta = 0.1
gamma = 0.1
initial_conditions_A = [1.0, 1.0, 1.0]
initial_conditions_B = [0.9, 0.9, 0.9]
t_span = (0, 10)
n_steps = 1000

# Simulación para posición A inicial
x_A, y_A, z_A = simulate_trajectory(
    alpha, beta, gamma, initial_conditions_A, t_span, n_steps)

# Simulación para posición B inicial
x_B, y_B, z_B = simulate_trajectory(
    alpha, beta, gamma, initial_conditions_B, t_span, n_steps)

# Gráfica de las trayectorias
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_A, y_A, z_A, label='Posición A inicial')
ax.plot(x_B, y_B, z_B, label='Posición B inicial')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.title('Trayectoria del Nanodron usando el método de Euler')
plt.show()

# Función interactiva para visualizar diferentes trayectorias


def interactive_simulation(alpha, beta, gamma, x0, y0, z0):
    x, y, z = simulate_trajectory(
        alpha, beta, gamma, [x0, y0, z0], t_span, n_steps)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title(f'Trayectoria del Nanodron: α={alpha}, β={beta}, γ={gamma}')
    plt.show()


alpha_slider = widgets.FloatSlider(
    min=0.01, max=1.0, step=0.01, value=0.1, description='α')
beta_slider = widgets.FloatSlider(
    min=0.01, max=1.0, step=0.01, value=0.1, description='β')
gamma_slider = widgets.FloatSlider(
    min=0.01, max=1.0, step=0.01, value=0.1, description='γ')
x0_slider = widgets.FloatSlider(
    min=0.5, max=1.5, step=0.1, value=1.0, description='x0')
y0_slider = widgets.FloatSlider(
    min=0.5, max=1.5, step=0.1, value=1.0, description='y0')
z0_slider = widgets.FloatSlider(
    min=0.5, max=1.5, step=0.1, value=1.0, description='z0')

widgets.interactive(interactive_simulation, alpha=alpha_slider, beta=beta_slider,
                    gamma=gamma_slider, x0=x0_slider, y0=y0_slider, z0=z0_slider)
