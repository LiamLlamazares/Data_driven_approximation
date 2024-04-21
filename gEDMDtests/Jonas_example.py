#In this file we show a visual representation of EDMD for double well potential
# using 4 indicator functions

import numpy as np
import matplotlib.pyplot as plt
import sys
import sys
import os
import scipy.stats as stats
# Get the directory of the current script
script_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

import d3s.algorithms as algorithms
import d3s.domain as domain
import d3s.observables as observables
import d3s.gEDMD_tests_helper_functions as gedmd_helper
import d3s.systems as systems

plt.ion()

# define domain
bounds = np.array([[-4, 4], [-4, 4]])
boxes = np.array([2, 2])
Omega = domain.discretization(bounds, boxes)

# Get sample and basis functions
M = 10000
X = Omega.rand(M)

#Defines indicators. Modify if want different ones. In the indicators in Stefan's code indexing was bootom to top and left to right, that is
# 1 3
# 0 2


def indicators(X):
    [d, M] = X.shape  # d = dimension of state space, m = number of test points
    y = np.zeros([4, M])
    for m in range(M):
        if X[0, m] < 0:
            y[0, m] = 1
        else:
            y[1, m] = 1
        if X[1, m] < 0:
            y[2, m] = 1
        else:
            y[3, m] = 1
    return y


#Double well system Langevin dynamics
alpha = 1


def b(X):
    b1 = X[1:]
    b2 = -(X[0, :]**3 - X[0, :]) - alpha * X[1, :]
    return np.vstack((b1, b2))


# Returns matrix [[0,0],[0, sqrt(2alpha)]] of shape (2, 2, M)
sigma = np.array([[0, 0], [0, np.sqrt(2 * alpha)]])

# Gets solution map to system
n_t = 1000
dt = 0.01


def f(X):
    return gedmd_helper.SDE_solver_2D(X, b, sigma, n_t, dt)


# Gets matrix
A, _, _, _ = gedmd_helper.gedmdMatrices(
    X,
    indicators,
    1,
    Omega,
    1,
    f,
)
print(A)
# Plots the i-th row of A_ex over the domain. The element A_ij is the probability that a point starting in box i will end in box j after one time step.
# for i in range(A.shape[0]):
#     plt.figure()
#     plt.imshow(A[i, :].reshape(boxes), origin='lower')
#     plt.colorbar()
#     plt.title('Probability of going from box ' + str(i) + ' to any other box')
#     plt.show(block=True)
