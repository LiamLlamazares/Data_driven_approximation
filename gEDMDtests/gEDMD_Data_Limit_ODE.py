#In this file we show plots of the rate of convergence
#of GEDMD for different number of observables
#We compare the rate of convergence for monomials and gaussians

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
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
min_number_of_data_points = 200

# generate data
M = 200000
confidence_level = 0.95

number_of_loops = int(np.floor(np.log2(M / min_number_of_data_points))) + 1
data_points_number = [
    min_number_of_data_points * 2**x for x in range(0, number_of_loops)
]
print('max data_points_number = ',
      min_number_of_data_points * 2**number_of_loops, 'number_of_loops = ',
      number_of_loops)
number_of_runs = 15

operator_errors = np.zeros(
    (number_of_loops, types_of_observables_number, number_of_runs))
operator_errors_average = np.zeros(
    (number_of_loops, types_of_observables_number))

for m in range(number_of_runs):
    X_exact = Omega.rand(M)
    A_exact_m, _, _ = gedmd_helper.gedmdMatrices(X_exact, psi_m, b, Omega)
    A_exact_g, _, _ = gedmd_helper.gedmdMatrices(X_exact, psi_g, b, Omega)
    A_exact_m__operator_norm = np.linalg.norm(A_exact_m, ord=2)
    A_exact_g__operator_norm = np.linalg.norm(A_exact_g, ord=2)
    for i in range(number_of_loops):
        X = Omega.rand(data_points_number[i])
        A_m, _, _ = gedmd_helper.gedmdMatrices(X, psi_m, b, Omega)
        A_g, _, _ = gedmd_helper.gedmdMatrices(X, psi_g, b, Omega)
        operator_error_m = np.linalg.norm(A_exact_m - A_m,
                                          ord=2) / A_exact_m__operator_norm
        operator_error_g = np.linalg.norm(A_exact_g - A_g,
                                          ord=2) / A_exact_g__operator_norm

        operator_errors[i, :, m] = [operator_error_m, operator_error_g]
        print('i = ', i, 'data points=', data_points_number[i],
              'run number = ', m)

operator_errors_average = np.mean(operator_errors, axis=2)
#calculate confidence intervals for the average error of the operators (95% confidence) for each number of data points
#first we divide the error data into number_of_batches batches
number_of_batches = 3
batch_size = int(np.floor(number_of_runs /
                          number_of_batches))  #number of runs in each batch
#GIves error if batch size is 0
if batch_size == 0:
    raise Exception(
        'batch size is 0. Please increase number of runs or decrease number of batches'
    )
operator_errors_batches = np.zeros(
    (number_of_loops, types_of_observables_number, number_of_batches))
for i in range(number_of_batches):
    operator_errors_batches[:, :, i] = np.mean(
        operator_errors[:, :, i * batch_size:(i + 1) * batch_size], axis=2)
#now we calculate the average and standard deviation of each batch
operator_errors_average = np.mean(operator_errors_batches, axis=2)
operator_errors_std = np.std(operator_errors_batches, axis=2, ddof=1)
#the batch averages can be interpreted as being Gaussian for large number of runs
#so we can calculate the confidence intervals using student's t-distribution
#we define the t_value for 95% confidence and 9 degrees of freedom
t_value = stats.t.ppf((1 + 0.95) / 2, number_of_batches - 1)
operator_errors_confidence_interval = t_value * operator_errors_std / np.sqrt(
    number_of_batches)

#error plots
# ... rest of your code ...

# error plots
plt.figure()
plt.loglog(data_points_number, operator_errors_average)

# plot confidence intervals as shaded regions
lower_bound = operator_errors_average - operator_errors_confidence_interval
upper_bound = operator_errors_average + operator_errors_confidence_interval
plt.fill_between(data_points_number,
                 lower_bound[:, 0],
                 upper_bound[:, 0],
                 color='blue',
                 alpha=0.2)
plt.fill_between(data_points_number,
                 lower_bound[:, 1],
                 upper_bound[:, 1],
                 color='orange',
                 alpha=0.2)

# slopes
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

# plot legends
plt.legend([
    'Gaussian operator error', 'Monomial operator error', 'slope -1',
    'slope -0.5'
])
plt.title('log-log-plot of error of operators vs number of observables')
plt.show(block=True)
1 - 1
