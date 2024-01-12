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
bounds = np.array([[-2, 2], [-2, 2]])
boxes = np.array([50, 50])
Omega = domain.discretization(bounds, boxes)

# define system
gamma = -0.8
delta = -0.7

def b(x):
    return np.array([gamma*x[0, :], delta*(x[1, :] - x[0, :]**2)])

# define observables
psi = observables.monomials(8)

# generate data
Xexact = Omega.rand(1000000) # generate test points
Yexact = b(Xexact)

data_points_number= [psi.length(Xexact)*2**x for x in range(0,15)]
operator_errors=np.zeros((len(data_points_number)))
frobenius_errors=np.zeros((len(data_points_number)))
eigenvalue_errors=np.zeros((len(data_points_number)))
eigenfunction_errors=np.zeros((len(data_points_number)))
operator_norms_K_exact=np.zeros((len(data_points_number)))


for i in range(len(data_points_number)):
    # generate data
    X=Omega.rand(data_points_number[i])
    operator_error,frobenius_error,eigenvalue_error,  eigenfunction_error, operator_norm_K_exact=gedmd_helper.gedmdErrors(Xexact, X, psi, b, Omega=Omega)
    operator_errors[i]=operator_error
    frobenius_errors[i]=frobenius_error
    eigenvalue_errors[i]=eigenvalue_error
    eigenfunction_errors[i]=eigenfunction_error
    operator_norms_K_exact[i]=operator_norm_K_exact


plt.loglog(data_points_number,operator_errors)
plt.loglog(data_points_number,frobenius_errors)
plt.loglog(data_points_number,eigenvalue_errors)
plt.loglog(data_points_number, eigenfunction_errors)

#slopes
plt.loglog(data_points_number, np.power(np.float64(data_points_number), -1) * operator_errors[1] / np.power(np.float64(data_points_number[1]), -1))
plt.loglog(data_points_number, np.power(np.float64(data_points_number), -0.5) * operator_errors[1] / np.power(np.float64(data_points_number[1]), -0.5))
plt.xlabel('number of data points')

#plot legends
plt.legend(['operator error','frobenius error','eigenvalue error','eigenfunction error','slope 1','slope 0.5'])
plt.title('log-log-plot of error of operators vs number of observables')
plt.show(block=True)

    
