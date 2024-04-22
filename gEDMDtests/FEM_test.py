#In this file we show plots of the rate of convergence
#of GEDMD for different number of observables
#We compare the rate of convergence for monomials and gaussians

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
from skfem import *
from skfem.helpers import dot, grad
import skfem as fem
import matplotlib.tri as mtri

plt.ion()
# Constants
M = 10**6
number_of_runs = 50
number_of_batches = 10
confidence_level = 0.95
degree_of_monomials = 8
observables_names = ['Monomials', 'Gaussians']
min_number_of_data_points = 250

# ########################################
#Simple deterministic system
# ########################################
# define domain
bounds = np.array([[-2, 2], [-1, 1]])
boxes = np.array([2, 2])
Omega = domain.discretization(bounds, boxes)

# define system
gamma = -0.8
delta = -0.7


def b(x):
    return np.array([gamma * x[0, :], delta * (x[1, :] - x[0, :]**2)])


# define observables
psi_m = observables.monomials(degree_of_monomials)
variance = (bounds[0, 1] - bounds[0, 0]) / boxes[0] / 2
psi_g = observables.gaussians(Omega, sigma=variance)
psi_FEM = observables.FEM_2d(Omega)
#psi_FEM._FEM_2d__get_Triangle(np.array([0.5, 0.5]))
psi_FEM.plot_mesh()
psi_FEM._FEM_2d__get_Triangle(np.array([-1.7, 0.95]))
psi_FEM._FEM_2d__get_Triangle(np.array([0.5, 0.5]))
import time
#set seed
np.random.seed(0)

X = Omega.rand(100000)
start = time.time()
triangle_indices = psi_FEM._FEM_2d__get_Triangles(X)
end = time.time()
print('Time to get triangle indices: ', end - start)
triangle_indices[0:3]
X[:, 0:3]
vertices = psi_FEM.node_coordinates
triangles = psi_FEM.t
k = 2
[psi_FEM.inverse_mappings[k](vertices[triangles[k, i]])
 for i in range(3)]  # Should return [0,0],[1,0],[0,1]

Sigma = 2 * np.eye(2)
start = time.time()
G = psi_FEM.calc_G(X, f=None)
end = time.time()
print('Time to calculate G: ', end - start)
start = time.time()
C = psi_FEM.calc_C(X, b, Sigma, f=None)
end = time.time()
print('Time to calculate C: ', end - start)
A1, G1, C1, _ = gedmd_helper.gedmdMatrices(X,
                                           psi_FEM,
                                           b,
                                           Omega,
                                           sigma=None,
                                           f=None,
                                           sigma_noise=0,
                                           operator='K')
observables_list = [psi_m, psi_g]

# or, save to external file:
# mesh.save('output.vtk', point_data={'solution': x})

# gedmd_helper.plot_errors_data_limit(
#     M,
#     min_number_of_data_points,
#     confidence_level,
#     number_of_runs,
#     number_of_batches,
#     observables_list,
#     observables_names,
#     Omega,
#     b,
#     sigma_noise=0,
#     # title='Simple deterministic system gEDMD',
#     operator='K',
#     path='ODE')

# # ########################################
# #Double well system
# # ########################################
# def b(x):
#     return np.vstack((-4 * x[0, :]**3 + 4 * x[0, :], -2 * x[1, :]))

# def sigma(x):
#     n = x.shape[1]
#     y = np.zeros((2, 2, n))
#     y[0, 0, :] = 0.7
#     y[0, 1, :] = x[0, :]
#     y[1, 1, :] = 0.5
#     return y

# gedmd_helper.plot_errors_data_limit(M,
#                                     min_number_of_data_points,
#                                     confidence_level,
#                                     number_of_runs,
#                                     number_of_batches,
#                                     observables_list,
#                                     observables_names,
#                                     Omega,
#                                     b,
#                                     sigma=sigma,
#                                     path='Double_well')

# f = systems.DoubleWell2D(1e-2, 1000)  #EDMD
# gedmd_helper.plot_errors_data_limit(M,
#                                     min_number_of_data_points,
#                                     confidence_level,
#                                     number_of_runs,
#                                     number_of_batches,
#                                     observables_list,
#                                     observables_names,
#                                     Omega,
#                                     b,
#                                     f=f,
#                                     sigma=sigma,
#                                     path='Double_well_EDMD')

# ########################################
#OU system
# ########################################

# define domain
bounds = np.array([[-2, 2]])
boxes = np.array([degree_of_monomials + 1])
Omega = domain.discretization(bounds, boxes)

# define system
alpha = 1
beta = 4


def b(x):
    return -alpha * x


def sigma(x):
    return np.sqrt(2 / beta) * np.ones((1, 1, x.shape[1]))


# define observables
psi_m = observables.monomials(degree_of_monomials)
variance = (bounds[0, 1] - bounds[0, 0]) / boxes[0] / 2
psi_g = observables.gaussians(Omega, sigma=variance)
psi_FEM = observables.FEM_1d(bounds[0, 0], bounds[0, 1], boxes[0])
observables_list = [psi_m, psi_g, psi_FEM]
observables_names = ['Monomials', 'Gaussians', 'FEM']
#gEDMD Koopman operator
gedmd_helper.plot_errors_data_limit(M,
                                    min_number_of_data_points,
                                    confidence_level,
                                    number_of_runs,
                                    number_of_batches,
                                    observables_list,
                                    observables_names,
                                    Omega,
                                    b,
                                    sigma=sigma,
                                    operator='K',
                                    path='OU_FEM')
# gEDMD Perron-Frobenius operator. Monomials are stable so error is 0
gedmd_helper.plot_errors_data_limit(M,
                                    min_number_of_data_points,
                                    confidence_level,
                                    number_of_runs,
                                    number_of_batches,
                                    observables_list,
                                    observables_names,
                                    Omega,
                                    b,
                                    sigma=sigma,
                                    operator='P',
                                    path='OU_PF_FEM')

h = 0.001  #EDMD
tau = 0.5
f = systems.OrnsteinUhlenbeck(h, int(tau / h))
gedmd_helper.plot_errors_data_limit(M,
                                    min_number_of_data_points,
                                    confidence_level,
                                    number_of_runs,
                                    number_of_batches,
                                    observables_list,
                                    observables_names,
                                    Omega,
                                    b,
                                    sigma=sigma,
                                    f=f,
                                    path='OU_EDMD_FEM')
