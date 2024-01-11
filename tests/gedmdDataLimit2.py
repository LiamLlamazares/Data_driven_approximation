
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

import d3s.algorithms as algorithms
import d3s.domain as domain
import d3s.observables as observables
import d3s.systems as systems

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
N=8
evs = int((N+1)*(N+2)/2)

psi = observables.monomials(N)

# generate data
Xexact = Omega.rand(1000000) # generate test points
Yexact = b(Xexact)
Xexact.shape[0]
# apply generator EDMD
 # number of eigenvalues/eigenfunctions to be computed
Kexact, dexact, Vexact = algorithms.gedmd(Xexact, Yexact, None, psi, evs=evs, operator='K')
# printMatrix(K, 'K_gEDMD')

#This normalizes the columns of V by dividing by their norm
normalizedVexact=np.zeros((Vexact.shape[0],Vexact.shape[1]),dtype="complex_")
for i in range(Vexact.shape[1]):
    normalizedVexact[:,i]=Vexact[:,i]/np.linalg.norm(Vexact[:,i])
#for i in range(Vexact.shape[1]):
 #   print(np.linalg.norm(normalizedVexact[:,i]))

#A loop that repeats the above, with fewer test points the number we use is 100,1000,10000,100000,500000
# generate data
testpoints= [Vexact.shape[0]*2**x for x in range(0,15)]

Vnormalized=np.zeros((Vexact.shape[0],Vexact.shape[1],len(testpoints)),dtype="complex_")
K=np.zeros((Vexact.shape[0],Vexact.shape[0],len(testpoints)))
for i in range(len(testpoints)):
    X=Omega.rand(testpoints[i])
    Y=b(X)
    K2, d, V = algorithms.gedmd(X, Y, None, psi, evs=evs, operator='K')
    K[:,:,i]=K2
    for j in range(V.shape[1]):
        Vnormalized[:,j,i]=V[:,j]/np.linalg.norm(V[:,j])
        #print the norm of the columns of Vnormalized
       # print(np.linalg.norm(Vnormalized[:,j,i]))  


# For each amount of data points, we calculate the distance between the eigefunction i of the two different sets of test points V and Vnew for each i
for k in range(len(testpoints)):
    distances = []
    for i in range(evs):
        for j in range(evs):
            distances.append(np.linalg.norm(Vnormalized[:,i,k]-Vexact[:, j]))
            distances.append(np.linalg.norm(Vnormalized[:,i,k]+Vexact[:, j]))
    distances.sort()
    #print("Error of "+ str(evs)+ " normalized eigenfunctions for "+ str(testpoints[k])+ " test points")
    #print(distances[:evs])
    #print("error norm of eigenvalues")
    #print(np.linalg.norm(distances[:evs]))
    #print("error norm of operators")
    #print(np.linalg.norm(K[:,:,k]-Kexact))
#print(Vexact)


#for i in range(len(testpoints)):
 #   for j in range(evs):
  #      psi.display(np.real(Vnormalized[:,j, i]), 2, 'phi_%d' % (j+1))

#for i in range(evs):
#    psi.display(np.real(normalizedVexact[:, i]), 2, 'phi_%d' % (i+1))
#print('')


operatorerror=np.zeros((len(testpoints),1))
for k in range(len(testpoints)):
    #We calculate the singular values of Kexact-K
    u,s,vh=np.linalg.svd(Kexact-K[:,:,k])
    #Then we take the square root of the largest singular value
    operatorerror[k]=np.max(s)

#Frobenius Error
frobeniuserror=np.zeros((len(testpoints),1))
for k in range(len(testpoints)):
    frobeniuserror[k]=np.linalg.norm(K[:,:,k]-Kexact)




#loglog plot
plt.figure()
plt.loglog(testpoints,operatorerror)
plt.loglog(testpoints,frobeniuserror)
# also plot a line with log log slope -1/2 to see if the error of the operators is proportional to the number of observables squared
plt.loglog(testpoints,(np.power(testpoints,-0.5)*operatorerror[1]/np.power(testpoints[1],-0.5)))
# also plot a line with log log slope -1/3 to see if the error of the operators is proportional to the number of observables squared
#plt.loglog(testpoints,(np.power(testpoints,-0.3333)*operatorerror[1]/np.power(testpoints[1],-0.3333)))
# also plot a line with log log slope -1/4 to see if the error of the operators is proportional to the number of observables squared
#plt.loglog(testpoints,(np.power(testpoints,-1/4)*operatorerror[1]/np.power(testpoints[1],-1/4)))

plt.xlabel('number of test points')
plt.ylabel('opertor error')
plt.title('log-log-plot of operator error vs number of test points')
plt.legend(['operator error',"frobenius error",'slope -1/2','slope -1/3','slope -1/4'])
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