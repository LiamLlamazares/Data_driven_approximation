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
M_exact = 100000
M_approx = 10000
min_number_of_observables = 2
boxes1 = 32
boxes2 = 16
boxes = np.array([boxes1, boxes2])
max_number_of_observables = boxes1 * boxes2

number_of_runs = 50
number_of_batches = 5
confidence_level = 0.95
number_of_monomials = 8

# ########################################
#Simple deterministic system
# ########################################
# define domain
bounds = np.array([[-2, 2], [-1, 1]])
Omega = domain.discretization(bounds, boxes)

# define system
gamma = -0.8
delta = -0.7


def b(x):
    return np.array([gamma * x[0, :], delta * (x[1, :] - x[0, :]**2)])


#define observables
observables_names = ['Gaussians']
gedmd_helper.plot_errors_dictionary_limit(
    min_number_of_observables,
    max_number_of_observables,
    confidence_level,
    number_of_runs,
    number_of_batches,
    observables_names,
    Omega,
    b,
    sigma=None,
    sigma_noise=0.1,
    block=False,
    M_exact=M_exact,
    M_approx=M_approx,
    prob=0.5,
    #title='Simple deterministic system gEDMD',
    path='ODE')


# ########################################
#Double well system
# ########################################
def b(x):
    return np.vstack((-4 * x[0, :]**3 + 4 * x[0, :], -2 * x[1, :]))


def sigma(x):
    n = x.shape[1]
    y = np.zeros((2, 2, n))
    y[0, 0, :] = 0.7
    y[0, 1, :] = x[0, :]
    y[1, 1, :] = 0.5
    return y


observables_names = ['Gaussians']
gedmd_helper.plot_errors_dictionary_limit(
    min_number_of_observables,
    max_number_of_observables,
    confidence_level,
    number_of_runs,
    number_of_batches,
    observables_names,
    Omega,
    b,
    sigma=None,
    block=False,
    M_exact=M_exact,
    M_approx=M_approx,
    prob=0.5,
    #title='Double well system gEDMD',
    path='Double_well')

f = systems.DoubleWell2D(1e-2, 1000)  #EDMD
gedmd_helper.plot_errors_dictionary_limit(
    min_number_of_observables,
    max_number_of_observables,
    confidence_level,
    number_of_runs,
    number_of_batches,
    observables_names,
    Omega,
    b,
    sigma=None,
    f=f,
    block=False,
    M_exact=M_exact,
    M_approx=M_approx,
    prob=0.5,
    #title='Double well system EDMD'
    path='Double_well_EDMD')

# ########################################
#OU system
# ########################################

# define domain
bounds = np.array([[-2, 2]])
boxes = np.array([max_number_of_observables])
Omega = domain.discretization(bounds, boxes)

# define system
alpha = 1
beta = 4


def b(x):
    return -alpha * x


def sigma(x):
    return np.sqrt(2 / beta) * np.ones((1, 1, x.shape[1]))


#define observables
observables_names = ['Gaussians']
gedmd_helper.plot_errors_dictionary_limit(
    min_number_of_observables,
    max_number_of_observables,
    confidence_level,
    number_of_runs,
    number_of_batches,
    observables_names,
    Omega,
    b,
    sigma=None,
    block=False,
    M_exact=M_exact,
    M_approx=M_approx,
    prob=0.5,
    #title='OU system'
    path='OU')

h = 0.001
tau = 0.5
f = systems.OrnsteinUhlenbeck(h, int(tau / h))  #EDMD
gedmd_helper.plot_errors_dictionary_limit(
    min_number_of_observables,
    max_number_of_observables,
    confidence_level,
    number_of_runs,
    number_of_batches,
    observables_names,
    Omega,
    b,
    sigma=None,
    f=f,
    block=True,
    M_exact=M_exact,
    M_approx=M_approx,
    prob=0.5,
    #title='OU system EDMD',
    path='OU_EDMD')
