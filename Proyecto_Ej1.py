import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def equations(t, state, alpha, beta, gamma):
    x, y, z = state
    dxdt = alpha * (y - x)
    dydt = x * (beta - z) - y
    dzdt = x * y - gamma * z
    return [dxdt, dydt, dzdt]


def solve_system(initial_conditions, alpha, beta, gamma, t_max):
    t_span = (0, t_max)
    t_eval = np.linspace(0, t_max, 1000)
    sol = solve_ivp(equations, t_span, initial_conditions,
                    args=(alpha, beta, gamma), t_eval=t_eval)
    return sol.t, sol.y


def plot_trajectory(t, trajectory, alpha, beta, gamma, initial_conditions):
    x, y, z = trajectory

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, label=f'α={alpha}, β={beta}, γ={
            gamma}, init={initial_conditions}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.title('Trayectoria del Nanodron')
    plt.show()


def main():
    # Casos de prueba para α, β, γ
    test_cases = [
        (0.1, 0.1, 0.1),
        (0.2, 0.2, 0.2),
        (0.3, 0.3, 0.3)
    ]

    # Posiciones iniciales del nanodron
    initial_positions = [
        [1, 1, 1],
        [0.9, 0.9, 0.9],
        [1.1, 1.1, 1.1]
    ]

    # Tiempo de simulación
    t_max = 10

    for alpha, beta, gamma in test_cases:
        for initial_conditions in initial_positions:
            t, trajectory = solve_system(
                initial_conditions, alpha, beta, gamma, t_max)
            plot_trajectory(t, trajectory, alpha, beta,
                            gamma, initial_conditions)


if __name__ == "__main__":
    main()
