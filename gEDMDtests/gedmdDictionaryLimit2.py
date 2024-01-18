# In this file we calculate for several dictionary lengths the operator error given by gEDMD
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
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

plt.ion()

#%% Simple deterministic system -------------------------------------------------------------------

# define domain
bounds = np.array([[-1, 1], [-1, 1]])
boxes = np.array([50, 50])
Omega = domain.discretization(bounds, boxes)

# define system
gamma = -0.8
delta = -0.7


#This corresponds to the ODE dx1=gamma*x1dt, dx2=delta*(x2-x1^2)dt
def b(x):
    return np.array([gamma * x[0, :], delta * (x[1, :] - x[0, :]**2)])


number_dictionary_lengths = 15
dictionary_lengths = range(0, number_dictionary_lengths)
number_of_observables = np.zeros(number_dictionary_lengths)
data_points = 10000
data_points_exact = 100000
operator_errors = np.zeros(number_dictionary_lengths)
Xexact = Omega.rand(data_points_exact)
for i in dictionary_lengths:
    X = Omega.rand(data_points)
    psi = observables.monomials(i + 1)
    number_of_observables[i] = psi.length()
    operator_error = gedmd_helper.gedmdErrors(Xexact, X, psi, b,
                                              Omega=Omega)[0]

    print(operator_error)
    operator_errors[i] = operator_error
#log-log plots
plt.figure()
plt.loglog(number_of_observables, operator_errors, 'o')
plt.xlabel('Number of observables')
plt.ylabel('Operator error')
plt.title('Operator error for different dictionary lengths')
plt.show()
1 - 1
# repeat 10 times and take average error
number_of_runs = 5
operator_errors_average = np.zeros(number_dictionary_lengths)
for i in range(number_of_runs):
    print("run number: ", i)
    X = Omega.rand(data_points)
    for j in dictionary_lengths:
        psi = observables.monomials(j + 1)
        operator_errors_average[j] += gedmd_helper.gedmdErrors(Xexact,
                                                               X,
                                                               psi,
                                                               b,
                                                               Omega=Omega)[0]
operator_errors_average /= number_of_runs

# log-log plots
plt.figure()
plt.loglog(number_of_observables, operator_errors_average, 'o')
plt.xlabel('Number of observables')
plt.ylabel('Average operator error')
plt.title('Average operator error for different dictionary lengths')
plt.show()
1 - 1 - 1
1 - 1
