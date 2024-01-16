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

# define domain
bounds = np.array([[-2, 2]])
boxes = np.array([10])
Omega = domain.discretization(bounds, boxes)

# define system
alpha = 1
beta = 4


def b(x):
    return -alpha * x


def sigma(x):
    return np.sqrt(2 / beta) * np.ones((1, 1, x.shape[1]))


# define observables
psi_m = observables.monomials(10)
variance = (bounds[0, 1] - bounds[0, 0]) / boxes[0] / 2
psi_g = observables.gaussians(Omega, sigma=variance)
types_of_observables_number = 2
min_number_of_data_points = 200

# generate data
M = 200000

number_of_loops = int(np.floor(np.log2(M / min_number_of_data_points))) + 1
data_points_number = [
    min_number_of_data_points * 2**x for x in range(0, number_of_loops)
]
print('max data_points_number = ',
      min_number_of_data_points * 2**number_of_loops, 'number_of_loops = ',
      number_of_loops)
number_of_runs = 2

operator_errors = np.zeros(
    (number_of_loops, types_of_observables_number, number_of_runs))
operator_errors_average = np.zeros(
    (number_of_loops, types_of_observables_number))

for m in range(number_of_runs):
    X_exact = Omega.rand(M)
    A_exact_m, _, _ = gedmd_helper.gedmdMatrices(X_exact, psi_m, b, Omega,
                                                 sigma)
    A_exact_g, _, _ = gedmd_helper.gedmdMatrices(X_exact, psi_g, b, Omega,
                                                 sigma)
    A_exact_m__operator_norm = np.linalg.norm(A_exact_m, ord=2)
    A_exact_g__operator_norm = np.linalg.norm(A_exact_g, ord=2)
    for i in range(number_of_loops):
        X = Omega.rand(data_points_number[i])
        A_m, _, _ = gedmd_helper.gedmdMatrices(X, psi_m, b, Omega, sigma)
        A_g, _, _ = gedmd_helper.gedmdMatrices(X, psi_g, b, Omega, sigma)
        operator_error_m = np.linalg.norm(A_exact_m - A_m,
                                          ord=2) / A_exact_m__operator_norm
        operator_error_g = np.linalg.norm(A_exact_g - A_g,
                                          ord=2) / A_exact_g__operator_norm

        operator_errors[i, :, m] = [operator_error_m, operator_error_g]
        print('i = ', i, 'data points=', data_points_number[i],
              'loop number = ', m)

operator_errors_average = np.mean(operator_errors, axis=2)

#error plots
plt.figure()
plt.loglog(data_points_number, operator_errors_average)
#plt.loglog(data_points_number, frobenius_errors_average)

#slopes
plt.loglog(
    data_points_number,
    np.power(np.float64(data_points_number), -1) *
    operator_errors_average[0, 1] /
    np.power(np.float64(data_points_number[0]), -1))
plt.loglog(
    data_points_number,
    np.power(np.float64(data_points_number), -0.5) *
    operator_errors_average[0, 1] /
    np.power(np.float64(data_points_number[0]), -0.5))
plt.xlabel('number of data points')

#plot legends
plt.legend([
    'Gaussian operator error', 'Monomial operator error', 'slope -1',
    'slope -0.5'
])
plt.title('error of gEDMD for double well generator vs number of observables')
plt.show(block=True)
