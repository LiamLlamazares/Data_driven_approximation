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

# Number of observables
N=5
# define observables
psi = observables.monomials(N)
# Error we want to achieve
epsilon = 0.01
#Initial number of data points
datapoints=int((N+1)*(N+2)/2)
error=10**10
#list to store the error of each step
errorlist=[]
# Loop that increases the number of data points until the operator error is smaller than epsilon
#list to store the number of data points used in each step
datapointslist=[]
while error>0.3:
    datapointsexact=10*datapoints
    Xexact = Omega.rand(datapointsexact) 
    Yexact = b(Xexact)
    X=Omega.rand(datapoints)
    Y=b(X)
    # apply generator EDMD
    Kexact, dexact, Vexact = algorithms.gedmd(Xexact, Yexact, None, psi, evs=N, operator='K')
    K, d, V = algorithms.gedmd(X, Y, None, psi, evs=N, operator='K')
    #We calculate the singular values of Kexact-K
    u,s,vh=np.linalg.svd(Kexact-K)
    #Then we take the square root of the largest singular value
    error=np.max(s)
    #We store the error in a list
    errorlist.append(error)
    print("error is "+ str(error) + " for "+ str(datapoints) + " data points")
    #Store the number of data points used in each step
    datapointslist.append(datapoints)
    #We increase the number of data points by a factor of 2
    datapoints=datapoints*2


plt.figure()
# table of datapoints used in loop
#datapointslist=[int((N+1)*(N+2)/2)*2**x for x in range(0,len(errorlist))]
#We log-log plot the error as a function of the number of data points
plt.loglog(datapointslist,errorlist)
#We also log-log plot the line with log-log slope -1/2 starting at the first error and datapoint
plt.loglog(datapointslist,np.power(datapointslist,-1/2)*errorlist[0]/np.power(datapointslist[0],-1/2))
#We also log-log plot the line with log-log slope -1 starting at the first error and datapoint
plt.loglog(datapointslist,np.power(datapointslist,-1.0)*errorlist[0]/np.power(datapointslist[0],-1.0))



plt.xlabel("Number of data points")
plt.ylabel("Operator error")
plt.title("Operator error as a function of the number of data points for N=8")
#Legends
plt.legend(["Operator error","slope -1/2","slope -1"])
plt.show(block=True)

1-1

#predicted number of data points needed to get an error smaller than epsilon
predicted=(errorlist[0]**2+int((N+1)*(N+2)/2)*epsilon**2-(epsilon**2))/(epsilon**2)





plt.loglog(datapointslist,predicted)
1-1
