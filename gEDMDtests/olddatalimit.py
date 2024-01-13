import numpy as np
import matplotlib.pyplot as plt
import sys
import sys
import os
# Get the directory of the current script
script_dir = os.path.dirname(os.path.realpath(__file__))

# Get the parent directory
parent_dir = os.path.dirname(script_dir)

# Add the parent directory to the system path
sys.path.insert(0, parent_dir)

import d3s.algorithms as algorithms
import d3s.domain as domain
import d3s.observables as observables
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
psi = observables.monomials(8)

# generate data
Xexact = Omega.rand(1000000)  # generate test points
Yexact = b(Xexact)

# apply generator EDMD

#In the notation we use in our new paper:
# C= <A psi,psi>, where A is the generator of the system
# G = <psi,psi>, is the Gramm matrix
# A =G^{-1} C^T is the operator matrix
# In the notation in the old paper by Stefan et al:
# K = operator matrix = A
# C0 = gram matrix = G
# C1 = stiffness matrix = C

evs = 8  # number of eigenvalues/eigenfunctions to be computed

# Gets the operator matrix Kexact, the eigenvalues dexact and the eigenvectors Vexact for a lot of test points
Kexact, dexact, Vexact = algorithms.gedmd(Xexact,
                                          Yexact,
                                          None,
                                          psi,
                                          evs=evs,
                                          operator='K')

#This normalizes the columns of V by dividing by their norm
normalizedVexact = np.zeros((Vexact.shape[0], Vexact.shape[1]))
for i in range(Vexact.shape[1]):
    normalizedVexact[:, i] = Vexact[:, i] / np.linalg.norm(Vexact[:, i])

#A loop that repeats the above, with fewer test points the number we use is 100,1000,10000,100000,500000
# generate data
testpoints = [Vexact.shape[0] * 2**x for x in range(0, 15)]

Vnormalized = np.zeros((Vexact.shape[0], Vexact.shape[1], len(testpoints)))
K = np.zeros((Vexact.shape[0], Vexact.shape[0], len(testpoints)))
for i in range(len(testpoints)):
    X = Omega.rand(testpoints[i])
    Y = b(X)
    K2, d, V = algorithms.gedmd(X, Y, None, psi, evs=evs, operator='K')
    K[:, :, i] = K2

K.shape
# For each amount of data points, we calculate the distance between the eigefunction i of the two different sets of test points V and Vnew for each i
#Here V[:,i,k] is the i-th eigenvector of the operator matrix K for the k-th set of test points
eigenfunction_errors = np.zeros(len(testpoints))
for k in range(len(testpoints)):
    distances = []
    for i in range(evs):
        for j in range(evs):
            distances.append(
                np.linalg.norm(Vnormalized[:, i, k] - Vexact[:, j]))
            distances.append(
                np.linalg.norm(Vnormalized[:, i, k] + Vexact[:, j]))
    distances.sort()
    # print("Error of " + str(evs) + " normalized eigenfunctions for " +
    #   str(testpoints[k]) + " test points")
    # We only print the first evs entries of the list distances, because they are the ones where the same eigenfunctions are compared
    # print(distances[:evs])
    print("error norm of eigenvalues")
    print(np.linalg.norm(distances[:evs]))
    eigenfunction_errors[k] = np.linalg.norm(distances[:evs])
    print("Frobenius distance of operators")
    print(np.linalg.norm(K[:, :, k] - Kexact))
print(Vexact)

#loglog plot of eigenfunction error as a function of number of test points
plt.figure()
plt.loglog(testpoints, eigenfunction_errors)
plt.xlabel('number of test points')
plt.ylabel('error norm of eigenfunctions')
plt.title(
    'log-log-plot of error norm of eigenfunctions vs number of test points')
plt.show(block=True)

#Prints the expression of the approximate normalized eigenfunctions as a linear combination of the monomials
for i in range(len(testpoints)):
    for j in range(evs):
        psi.display(np.real(Vnormalized[:, j, i]), 2, 'phi_%d' % (j + 1))

#Prints the expression of the 'exact' normalized eigenfunctions as a linear combination of the monomials
for i in range(evs):
    psi.display(np.real(normalizedVexact[:, i]), 2, 'phi_%d' % (i + 1))
print('')

# We plot the frobenius error between operator matrices as a function of number of test points
frobeniusnorm = np.zeros((len(testpoints), 1))
for k in range(len(testpoints)):
    frobeniusnorm[k] = np.linalg.norm(K[:, :, k] - Kexact)

#loglog plot
plt.loglog(testpoints, frobeniusnorm)
plt.xlabel('number of test points')
plt.ylabel('frobenius norm of operator difference')
plt.title(
    'log-log-plot of frobenius norm of operator difference vs number of test points'
)
plt.show()
#This calculates the operator norm of Kexact, which is defined as the largest singular value of Kexact
#First we calculate the singular values of Kexact
u, s, vh = np.linalg.svd(Kexact)
#Then we take the largest singular value
operatornorm_exact = np.max(s)
#This calculates the operator norm of Kexact-K, which is defined as the largest singular value of Kexact-K
#First we calculate the singular values of Kexact-K
errorratio = np.zeros((len(testpoints), 1))
for k in range(len(testpoints)):
    u, s, vh = np.linalg.svd(K[:, :, k] - Kexact)
    #Then we take the largest singular value
    operatornorm_approx = np.max(s)
    print("operator norm of Kexact-K for " + str(testpoints[k]) +
          " test points")
    print(operatornorm_approx)
    print(
        "ratio of operator norm of Kexact-K to operator norm of Kexact for " +
        str(testpoints[k]) + " test points")
    print(operatornorm_approx / operatornorm_exact)
    errorratio[k] = np.linalg.norm(K[:, :, k] -
                                   Kexact) / np.linalg.norm(Kexact)
    print(
        "ratio of operator norm of Kexact-K to operator norm of Kexact for " +
        str(testpoints[k]) + " test points")
    print(errorratio[k])
# log log plot the ratio of the operator norm of Kexact-K to the operator norm of Kexact as a function of the number of test points
plt.figure()
plt.loglog(testpoints, errorratio)
plt.xlabel('number of test points')
plt.ylabel('ratio of operator norm of Kexact-K to operator norm of Kexact')
plt.title(
    'log-log-plot of ratio of operator norm of Kexact-K to operator norm of Kexact vs number of test points'
)
plt.show()
#loglog plot of frobenius norm  error minus operator norm error in absolute value
plt.figure()
plt.loglog(testpoints, np.abs(frobeniusnorm / operatornorm_exact - errorratio))
plt.xlabel('number of test points')
plt.ylabel(
    'absolute value of difference between frobenius norm error and operator norm error'
)
plt.title(
    'log-log-plot of absolute value of difference between frobenius norm error and operator norm error vs number of test points'
)
plt.show(block=True)

# # Prints individually ratio erros for frobenius norm and operator norm
# for k in range(len(testpoints)):
#     print("ratio of frobenius norm error and error ratio to operator norm error for "+ str(testpoints[k])+ " test points")
#     print([frobeniusnorm[k]/operatornorm_exact,errorratio[k]])
