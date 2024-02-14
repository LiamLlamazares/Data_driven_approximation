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

plt.ion()
# Constants
M = 1000000
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
boxes = np.array([9, 5])
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
observables_list = [psi_m, psi_g]

# sigma_noise = 0.001
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
#     sigma_noise=sigma_noise,
#     # title='Simple deterministic system gEDMD',
#     block=False,
#     path='ODE_sigma_noise =' + str(sigma_noise))
# sigma_noise = 0.1
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
#     sigma_noise=sigma_noise,
#     # title='Simple deterministic system gEDMD',
#     block=False,
#     path='ODE_sigma_noise =' + str(sigma_noise))
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
observables_list = [psi_m, psi_g]

# gEDMD Perron-Frobenius operator. Monomials are stable so error is 0

sigma_noise = 0.001
gedmd_helper.plot_errors_data_limit(
    M,
    min_number_of_data_points,
    confidence_level,
    number_of_runs,
    number_of_batches,
    observables_list,
    observables_names,
    Omega,
    b,
    sigma_noise=sigma_noise,
    sigma=sigma,
    # title='OU system gEDMD',
    block=False,
    path='OU_sigma_noise =' + str(sigma_noise))
sigma_noise = 0.01
gedmd_helper.plot_errors_data_limit(
    M,
    min_number_of_data_points,
    confidence_level,
    number_of_runs,
    number_of_batches,
    observables_list,
    observables_names,
    Omega,
    b,
    sigma_noise=sigma_noise,
    sigma=sigma,
    # title='OU system gEDMD',
    block=False,
    path='OU_sigma_noise =' + str(sigma_noise))
sigma_noise = 0.1
gedmd_helper.plot_errors_data_limit(
    M,
    min_number_of_data_points,
    confidence_level,
    number_of_runs,
    number_of_batches,
    observables_list,
    observables_names,
    Omega,
    b,
    sigma_noise=sigma_noise,
    sigma=sigma,
    # title='OU system gEDMD',
    block=False,
    path='OU_sigma_noise =' + str(sigma_noise))
