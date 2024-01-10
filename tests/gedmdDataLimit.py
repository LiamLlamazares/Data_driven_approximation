import sysconfig
import pybind11
python_include_path = sysconfig.get_paths()["include"]
pybind11_include_path = pybind11.get_include()

print(f"Python include path: {python_include_path}")
print(f"pybind11 include path: {pybind11_include_path}")



# python_include_path = sysconfig.get_paths()["include"]
# pybind11_include_path = pybind11.get_include()
# print(f"Python include path: {python_include_path}")
# print(f"pybind11 include path: {pybind11_include_path}")
#!/usr/bin/env 4python3
# -*- coding: utf-8 -*-
import numpy as np
import numpy.polynomial
import scipy as sp
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

print('The parent directory is: ' + parent_dir)
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
    return np.array([gamma*x[0, :], delta*(x[1, :] - x[0, :]**2)])

# define observables
psi = observables.monomials(8)

# generate data
Xexact = Omega.rand(100000) # generate test points
Yexact = b(Xexact)

# apply generator EDMD
evs = 8 # number of eigenvalues/eigenfunctions to be computed
Kexact, dexact, Vexact = algorithms.gedmd(Xexact, Yexact, None, psi, evs=evs, operator='K')
# printMatrix(K, 'K_gEDMD')

#This normalizes the columns of V by dividing by their norm
normalizedVexact=np.zeros((Vexact.shape[0],Vexact.shape[1]))
for i in range(Vexact.shape[1]):
    normalizedVexact[:,i]=Vexact[:,i]/np.linalg.norm(Vexact[:,i])
for i in range(Vexact.shape[1]):
    print(np.linalg.norm(normalizedVexact[:,i]))

#A loop that repeats the above, with fewer test points the number we use is 100,1000,10000,100000,500000
# generate data
testpoints= [Vexact.shape[0]*2**x for x in range(0,15)]

Vnormalized=np.zeros((Vexact.shape[0],Vexact.shape[1],len(testpoints)))
K=np.zeros((Vexact.shape[0],Vexact.shape[0],len(testpoints)))
for i in range(len(testpoints)):
    X=Omega.rand(testpoints[i])
    Y=b(X)
    K2, d, V = algorithms.gedmd(X, Y, None, psi, evs=evs, operator='K')
    K[:,:,i]=K2
    for j in range(V.shape[1]):
        Vnormalized[:,j,i]=V[:,j]/np.linalg.norm(V[:,j])
        #print the norm of the columns of Vnormalized
        print(np.linalg.norm(Vnormalized[:,j,i]))  

K.shape              
1-1    
# For each amount of data points, we calculate the distance between the eigefunction i of the two different sets of test points V and Vnew for each i
for k in range(len(testpoints)):
    distances = []
    for i in range(evs):
        for j in range(evs):
            distances.append(np.linalg.norm(Vnormalized[:,i,k]-Vexact[:, j]))
            distances.append(np.linalg.norm(Vnormalized[:,i,k]+Vexact[:, j]))
    distances.sort()
    print("Error of "+ str(evs)+ " normalized eigenfunctions for "+ str(testpoints[k])+ " test points")
    print(distances[:evs])
    print("error norm of eigenvalues")
    print(np.linalg.norm(distances[:evs]))
    print("error norm of operators")
    print(np.linalg.norm(K[:,:,k]-Kexact))
print(Vexact)


for i in range(len(testpoints)):
    for j in range(evs):
        psi.display(np.real(Vnormalized[:,j, i]), 2, 'phi_%d' % (j+1))

for i in range(evs):
    psi.display(np.real(normalizedVexact[:, i]), 2, 'phi_%d' % (i+1))
print('')

1-1
frobeniusnorm=np.zeros((len(testpoints),1))
for k in range(len(testpoints)):
    frobeniusnorm[k]=np.linalg.norm(K[:,:,k]-Kexact)
    
plt.plot(testpoints,frobeniusnorm)
plt.xlabel('number of test points')
plt.ylabel('frobenius norm of operator difference')
plt.title('frobenius norm of operator difference vs number of test points')
plt.show()
#loglog plot
plt.loglog(testpoints,frobeniusnorm)
plt.xlabel('number of test points')
plt.ylabel('frobenius norm of operator difference')
plt.title('log-log-plot of frobenius norm of operator difference vs number of test points')
plt.show()

#This calculates the operator norm of Kexact, which is defined as the largest singular value of Kexact
#First we calculate the singular values of Kexact
u,s,vh=np.linalg.svd(Kexact)
#Then we take the largest singular value
operatornorm=np.max(s)

#This calculates the operator norm of Kexact-K, which is defined as the largest singular value of Kexact-K
#First we calculate the singular values of Kexact-K
for k in range(len(testpoints)):
    u,s,vh=np.linalg.svd(K[:,:,k]-Kexact)
    #Then we take the largest singular value
    operatornorm2=np.max(s)
    print("operator norm of Kexact-K for "+ str(testpoints[k])+ " test points")
    print(operatornorm2)
    print("ratio of operator norm of Kexact-K to operator norm of Kexact for "+ str(testpoints[k])+ " test points")
    print(operatornorm2/operatornorm)
    print("ratio of operator norm of Kexact-K to operator norm of Kexact for "+ str(testpoints[k])+ " test points")
    print(np.linalg.norm(K[:,:,k]-Kexact)/np.linalg.norm(Kexact))
1-1