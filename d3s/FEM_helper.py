import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


def compute_A(N):
    """Compute matrices B and C."""
    h = 1 / (N + 1)
    B = np.zeros((N + 1, N + 1))
    C = np.zeros((N + 1, N + 1))

    for i in range(N + 1):
        for j in range(N + 1):
            xi = i * h
            xp = (i + 1) * h
            xm = (i - 1) * h
            if i == j:
                if i == 0:
                    C[i, j] = h / 6
                    B[i, j] = (np.exp(xp) - 1) / h**2
                else:
                    C[i, j] = h / 3
                    B[i, j] = (np.exp(xp) - np.exp(xm)) / h**2
            elif j == i - 1:
                C[i, j] = -(-2 * h + 3 * xi) / 6
                B[i, j] = -(np.exp(xi) - np.exp(xm)) / h**2
            elif j == i + 1:
                C[i, j] = (2 * h + 3 * xi) / 6
                B[i, j] = -(np.exp(xp) - np.exp(xi)) / h**2

    A = B + C

    return A, B, C, h


def compute_y(N):
    """Compute vectors y and F."""
    h = 1 / N
    y = np.zeros(N + 1)
    F = np.zeros(N + 1)
    for i in range(N + 1):
        y[i] = h
    return y


def compute_u_coefficients(N):
    """Compute coefficients of the u function."""
    u = np.zeros(N + 1)
    A, _, _, _ = compute_A(N)
    y = compute_y(N)
    u = np.linalg.solve(A.T, y)
    return u


def u_plotter(N_list, path):
    """Plot the u function."""
    u_list = []
    for N in N_list:
        u = compute_u_coefficients(N)
        u_list.append(u)
        x = np.linspace(0, 1, N + 1)
        plt.plot(x, u, label='N = ' + str(N))
    plt.legend()
    plt.savefig(path, bbox_inches='tight')
    plt.show(block=True)
