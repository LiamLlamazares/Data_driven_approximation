#In this file we show plots of the rate of convergence
#of GEDMD for different number of observables
#We compare the rate of convergence for monomials and gaussians

import numpy as np
import matplotlib.pyplot as plt
import sys
import sys
import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

import d3s.algorithms as algorithms
import d3s.domain as domain
import d3s.observables as observables
import d3s.gEDMD_tests_helper_functions as gedmd_helper

plt.ion()

#%% Simple deterministic system -------------------------------------------------------------------

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
psi_m = observables.monomials(8)
psi_g = observables.gaussians(Omega, sigma=0.2)
types_of_observables_number = 2
if psi_m.length() != psi_g.length():
    raise Exception('psi_monomials and psi_gaussians have different length')
N = psi_m.length()

# generate data
M = 100000

number_of_loops = int(np.floor(np.log2(M / (2 * N))))
data_points_number = [N * 2**x for x in range(0, number_of_loops)]
number_of_runs = 2

operator_errors = np.zeros(
    (number_of_loops, types_of_observables_number, number_of_runs))
operator_errors_average = np.zeros(
    (number_of_loops, types_of_observables_number))

for m in range(number_of_runs):
    for i in range(number_of_loops):
        # generate data
        Xexact = Omega.rand(M)
        X = Omega.rand(data_points_number[i])
        operator_error_m, A_norm_m, G_norm_m, C_norm_m = gedmd_helper.gedmdErrors(
            Xexact, X, psi_m, b, Omega=Omega)
        operator_error_g, A_norm_g, G_norm_g, C_norm_g = gedmd_helper.gedmdErrors(
            Xexact, X, psi_g, b, Omega=Omega)
        operator_errors[i, :, m] = [operator_error_m, operator_error_g]
        print('i = ', i, 'm = ', m)

operator_errors_average = np.mean(operator_errors, axis=2)

#error plots
plt.figure()
plt.loglog(data_points_number, operator_errors_average)
#plt.loglog(data_points_number, frobenius_errors_average)

#slopes
plt.loglog(
    data_points_number,
    np.power(np.float64(data_points_number), -1) *
    operator_errors_average[1, 1] /
    np.power(np.float64(data_points_number[1]), -1))
plt.loglog(
    data_points_number,
    np.power(np.float64(data_points_number), -0.5) *
    operator_errors_average[1, 1] /
    np.power(np.float64(data_points_number[1]), -0.5))
plt.xlabel('number of data points')

#plot legends
plt.legend([
    'Gaussian operator error', 'Monomial operator error', 'slope -1',
    'slope -0.5'
])
plt.title('log-log-plot of error of operators vs number of observables')
plt.show(block=True)
