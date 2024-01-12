
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
    return np.array([gamma*x[0, :], delta*(x[1, :] - x[0, :]**2)])

dictionarylengths= range(0,11+1)
datapoints=1000
datapointsexact=100000

operator_errors=np.zeros((len(dictionarylengths)))
frobenius_operator_errors=np.zeros((len(dictionarylengths)))
eigenvalue_errors=np.zeros((len(dictionarylengths)))
eigenfunction_errors=np.zeros((len(dictionarylengths)))
number_of_observables=np.zeros((len(dictionarylengths)))
operator_norms_K_exact=np.zeros((len(dictionarylengths)))



for i in dictionarylengths:
    # generate data
    Xexact = Omega.rand(datapointsexact) 
    X=Omega.rand(datapoints)
    psi = observables.monomials(i)
    number_of_observables[i]=psi.length(X)
    operator_error, frobenius_operator_error, eigenvalue_error,eigenfunction_error, operator_norm_K_exact=gedmd_helper.gedmdErrors(Xexact, X, psi, b, Omega=Omega)
    operator_errors[i]=operator_error
    frobenius_operator_errors[i]=frobenius_operator_error
    eigenvalue_errors[i]=eigenvalue_error
    eigenfunction_errors[i]=eigenfunction_error
    operator_norms_K_exact[i]=operator_norm_K_exact
    

   


# Log-Log-plots of the operator norm of Kexact versus the number of observables
plt.figure()
plt.loglog(number_of_observables,operator_norms_K_exact)
plt.loglog(number_of_observables,np.power(number_of_observables,1)*operator_norms_K_exact[1]/np.power(number_of_observables[1],1))
plt.loglog(number_of_observables,np.power(number_of_observables,2)*operator_norms_K_exact[1]/np.power(number_of_observables[1],2))
plt.legend(['operator norm of Kexact','slope 1','slope 2'])
plt.xlabel('number of observables')
plt.ylabel('operator norm of Kexact')
plt.figure()

# log log plot the norm of the eigenfunctions vs the number of dictionary lengths
plt.loglog(number_of_observables[5:11],eigenfunction_errors[5:11])
# log log plot the error of the operators vs the number of observables
plt.loglog(number_of_observables[1:11],operator_errors[1:11])
# log log plot of the error of the eigenvalues vs the number of observables
plt.loglog(number_of_observables[1:11],eigenvalue_errors[1:11])
# also plot a line with log log slope 1 to see if the error of the operators is proportional to the number of observables squared
plt.loglog(number_of_observables[1:11],np.power(number_of_observables[1:11],1)*operator_errors[1]/np.power(number_of_observables[1],1))
# also plot a line with log log slope 0.5 to see if the error of the operators is proportional to the number of observables to the power 1.5
plt.loglog(number_of_observables[1:11],np.power(number_of_observables[1:11],0.5)*operator_errors[1]/np.power(number_of_observables[1],0.5))

plt.xlabel('number of observables')

#plot legends
plt.legend(['norm of eigenfunctions','error of operators','error of eigenvalues','slope 1','slope 0.5'])
plt.title('log-log-plot of error of operators vs number of observables')
plt.show()



M=10
eigenfunction_errors=np.zeros((len(dictionarylengths),M))
evdistanceslists=[]
eigenvalue_errors=np.zeros((len(dictionarylengths),M))
operator_errors=np.zeros((len(dictionarylengths),M))
frobenius_errors=np.zeros((len(dictionarylengths),M))
operator_errors_average=np.zeros((len(dictionarylengths)))
frobenius_errors_average=np.zeros((len(dictionarylengths)))
eigenvalues_error_average=np.zeros((len(dictionarylengths)))
eigenfunction_errors_average=np.zeros((len(dictionarylengths)))

#We repeat the above for M runs
for i in dictionarylengths:
    for m in range(M):
        # generate data
        Xexact = Omega.rand(datapointsexact) 
        Yexact = b(Xexact)
        X=Omega.rand(datapoints)
        Y=b(X)
        psi = observables.monomials(i)
        evs = psi.length(X)
        operator_error, frobenius_error, eigenvalue_error,eigenfunction_error, operator_norm_K_exact=gedmd_helper.gedmdErrors(Xexact, X, psi, b, Omega=Omega)
        operator_errors[i,m]=operator_error
        frobenius_errors[i,m]=frobenius_error
        eigenvalue_errors[i,m]=eigenvalue_error
        eigenfunction_errors[i,m]=eigenfunction_error
        
        
    operator_errors_average[i]=np.average(operator_errors[i,:])
    frobenius_errors_average[i]=np.average(frobenius_errors[i,:])
    eigenvalues_error_average[i]=np.average(eigenvalue_errors[i,:])
    eigenfunction_errors_average[i]=np.average(eigenfunction_errors[i,:])      

        

#error plots
plt.figure()
plt.loglog(number_of_observables[1:11],operator_errors_average[1:11])
plt.loglog(number_of_observables[1:11],frobenius_errors_average[1:11]) #In this case the frobenius error is almost the same as the operator error
plt.loglog(number_of_observables[1:11],eigenvalues_error_average[1:11])
plt.loglog(number_of_observables[5:11],eigenfunction_errors_average[5:11])

#slope
plt.loglog(number_of_observables[1:11],np.power(number_of_observables[1:11],1)*operator_errors_average[1]/np.power(number_of_observables[1],1))
plt.loglog(number_of_observables[1:11],np.power(number_of_observables[1:11],0.5)*operator_errors_average[1]/np.power(number_of_observables[1],0.5))

plt.legend(['average operator error', 'average frobenius error', "average error of eigenvalues",'average error of eigenfunctions','slope 1','slope 0.5'])
plt.xlabel('number of observables')
plt.show(block=True)
