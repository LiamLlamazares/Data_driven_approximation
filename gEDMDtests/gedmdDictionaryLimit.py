# In this file we compute the error of the eigenvalues and eigenfunctions of the operator matrix Kexact for different dictionary lengths
# We plot the errors as a function of the number of observables
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


number_of_loops = 10
dictionarylengths = range(0, number_of_loops)
datapoints = 10000
datapointsexact = 100000

operator_errors = np.zeros((len(dictionarylengths)))
frobenius_operator_errors = np.zeros((len(dictionarylengths)))
eigenvalue_errors = np.zeros((len(dictionarylengths)))
number_of_observables = np.zeros((len(dictionarylengths)))
operator_norms_K_exact = np.zeros((len(dictionarylengths)))

Xexact = Omega.rand(datapointsexact)
for i in dictionarylengths:
    # generate data
    X = Omega.rand(datapoints)
    psi = observables.monomials(i)
    number_of_observables[i] = psi.length(X)
    operator_error, frobenius_operator_error, eigenvalue_error, operator_norm_K_exact = gedmd_helper.gedmdErrors(
        Xexact, X, psi, b, Omega=Omega)
    operator_errors[i] = operator_error
    frobenius_operator_errors[i] = frobenius_operator_error
    eigenvalue_errors[i] = eigenvalue_error
    operator_norms_K_exact[i] = operator_norm_K_exact

# Log-Log-plots of the operator norm of Kexact versus the number of observables
plt.figure()
plt.loglog(number_of_observables, operator_norms_K_exact)
plt.loglog(
    number_of_observables,
    np.power(number_of_observables, 1) * operator_norms_K_exact[1] /
    np.power(number_of_observables[1], 1))
plt.loglog(
    number_of_observables,
    np.power(number_of_observables, 2) * operator_norms_K_exact[1] /
    np.power(number_of_observables[1], 2))
plt.legend(['operator norm of Kexact', 'slope 1', 'slope 2'])
plt.xlabel('number of observables')
plt.ylabel('operator norm of Kexact')
plt.figure()

# log log plot the error of the operators vs the number of observables
plt.loglog(number_of_observables[1:11], operator_errors[1:11])
# log log plot of the error of the eigenvalues vs the number of observables
plt.loglog(number_of_observables[1:11], eigenvalue_errors[1:11])
# also plot a line with log log slope 1 to see if the error of the operators is proportional to the number of observables squared
plt.loglog(
    number_of_observables[1:11],
    np.power(number_of_observables[1:11], 1) * operator_errors[1] /
    np.power(number_of_observables[1], 1))
# also plot a line with log log slope 0.5 to see if the error of the operators is proportional to the number of observables to the power 1.5
plt.loglog(
    number_of_observables[1:11],
    np.power(number_of_observables[1:11], 0.5) * operator_errors[1] /
    np.power(number_of_observables[1], 0.5))

plt.xlabel('number of observables')

#plot legends
plt.legend(
    ['error of operators', 'error of eigenvalues', 'slope 1', 'slope 0.5'])
plt.title('log-log-plot of error of operators vs number of observables')
plt.show()

M = 50
eigenvalue_errors = np.zeros((len(dictionarylengths), M))
operator_errors = np.zeros((len(dictionarylengths), M))
frobenius_errors = np.zeros((len(dictionarylengths), M))
operator_errors_average = np.zeros((len(dictionarylengths)))
frobenius_errors_average = np.zeros((len(dictionarylengths)))
eigenvalues_error_average = np.zeros((len(dictionarylengths)))

#We repeat the above for M runs
for i in dictionarylengths:
    for m in range(M):
        print('dictionary length: ', i, 'run number: ', m)
        # generate data
        Xexact = Omega.rand(datapointsexact)
        Yexact = b(Xexact)
        X = Omega.rand(datapoints)
        Y = b(X)
        psi = observables.monomials(i)
        evs = psi.length(X)
        operator_error, frobenius_error, eigenvalue_error, operator_norm_K_exact = gedmd_helper.gedmdErrors(
            Xexact, X, psi, b, Omega=Omega)
        operator_errors[i, m] = operator_error
        frobenius_errors[i, m] = frobenius_error
        eigenvalue_errors[i, m] = eigenvalue_error

    operator_errors_average[i] = np.average(operator_errors[i, :])
    frobenius_errors_average[i] = np.average(frobenius_errors[i, :])
    eigenvalues_error_average[i] = np.average(eigenvalue_errors[i, :])

#calculate confidence intervals for the average error of the operators (95% confidence) for each number of dictionary elements
#first we divide the error data into 3 batches
number_of_batches = 10
batch_size = int(np.floor(M / number_of_batches))
operator_errors_batches = np.zeros((len(dictionarylengths), number_of_batches))
for i in range(number_of_batches):
    operator_errors_batches[:, i] = np.mean(
        operator_errors[:, i * batch_size:(i + 1) * batch_size], axis=1)
#now we calculate the average and standard deviation of each batch
operator_errors_average = np.mean(operator_errors_batches, axis=1)
operator_errors_std = np.std(operator_errors_batches, axis=1)
#the batch averages can be interpreted as being Gaussian for large number of runs
#so we can calculate the confidence intervals using student's t-distribution
#we define the t_value for 95% confidence and 9 degrees of freedom
t_value = 2.262

operator_errors_confidence_interval = t_value * operator_errors_std / np.sqrt(
    number_of_batches)

#error plots
plt.figure()
plt.loglog(number_of_observables[1:11], operator_errors_average[1:11])

#slope
plt.loglog(
    number_of_observables[1:11],
    np.power(number_of_observables[1:11], 1) * operator_errors_average[1] /
    np.power(number_of_observables[1], 1))
plt.loglog(
    number_of_observables[1:11],
    np.power(number_of_observables[1:11], 0.5) * operator_errors_average[1] /
    np.power(number_of_observables[1], 0.5))

plt.legend([
    'average operator error', 'average frobenius error',
    "average error of eigenvalues", 'slope 1', 'slope 0.5'
])
plt.xlabel('number of observables')
plt.show()
1 - 1
