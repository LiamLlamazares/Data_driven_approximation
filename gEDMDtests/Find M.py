# Code to find the number of data points needed to get an operator error smaller than epsilon for a given number of elements in the dictionary
import sysconfig
import pybind11

python_include_path = sysconfig.get_paths()["include"]
pybind11_include_path = pybind11.get_include()
print(f"Python include path: {python_include_path}")
print(f"pybind11 include path: {pybind11_include_path}")
import numpy as np
import numpy.polynomial
import scipy as sp
import matplotlib.pyplot as plt
import os
import sys

# Get the directory of the current script
script_dir = os.path.dirname(os.path.realpath(__file__))

# Get the parent directory
parent_dir = os.path.dirname(script_dir)

# Add the parent directory to the system path
sys.path.insert(0, parent_dir)

print('The parent directory is: ' + parent_dir)

import d3s.algorithms as algorithms
import d3s.domain as domain
import d3s.observables as observables
import d3s.gEDMD_tests_helper_functions as gedmd_helper
#import d3s.systems as systems
from d3s.tools import printVector, printMatrix

plt.ion()

#%% Simple deterministic system -------------------------------------------------------------------
# define domain
bounds = np.array([[-2, 2], [-2, 2]])
boxes = np.array([50, 50])
Omega = domain.discretization(bounds, boxes)

# define system
gamma = -0.8
delta = -0.7


def b(x):
    return np.array([gamma * x[0, :], delta * (x[1, :] - x[0, :]**2)])


# define observables
N = 5
psi = observables.monomials(N)

# Error we want to achieve
epsilon = 0.01

#Initial number of data points
observables_number = int((N + 1) * (N + 2) / 2)

#Loop to find the number of data points needed to get an error smaller than epsilon
operator_errors = []
observables_number_list = []
predicted = []
operator_error = epsilon + 1

while operator_error > epsilon:
    observables_number_exact = 10 * observables_number
    X_exact = Omega.rand(observables_number_exact)
    X = Omega.rand(observables_number)
    operator_error, _, _, norm_K_exact = gedmd_helper.gedmdErrors(X_exact,
                                                                  X,
                                                                  psi,
                                                                  b,
                                                                  Omega=Omega)
    operator_errors.append(operator_error)
    print("error is " + str(operator_error) + " for " +
          str(observables_number) + " data points")
    observables_number_list.append(observables_number)
    observables_number = 2 * observables_number
    #Would be good to have plots on bound on error as observables increase

plt.figure()
plt.loglog(observables_number_list, operator_errors)
plt.loglog(observables_number_list, predicted)
plt.loglog(
    observables_number_list,
    np.power(observables_number_list, -1 / 2) * operator_errors[0] /
    np.power(observables_number_list[0], -1 / 2))
plt.loglog(
    observables_number_list,
    np.power(observables_number_list, -1.0) * operator_errors[0] /
    np.power(observables_number_list[0], -1.0))
plt.xlabel("Number of data points")
plt.ylabel("Operator error")
plt.title("Operator error as a function of the number of data points for N=8")
plt.legend(["Operator error", "Predicted", "slope -1/2", "slope -1"])
plt.show(block=True)

# #predicted number of data points needed to get an error smaller than epsilon
# predicted = (operator_errors[0]**2 + int(
#     (N + 1) * (N + 2) / 2) * epsilon**2 - (epsilon**2)) / (epsilon**2)

# plt.loglog(observables_number_list, predicted)
1 - 1
