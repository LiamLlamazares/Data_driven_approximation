
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

plt.ion()

#%% Simple deterministic system -------------------------------------------------------------------

# define domain
bounds = np.array([[-1, 1], [-1, 1]])
boxes = np.array([50, 50])
Omega = domain.discretization(bounds, boxes)

# define system
gamma = -0.8
delta = -0.7



#A loop that repeats the above, with fewer dictionary lengths the number we use is 100,1000,10000,100000,500000
# generate data
dictionarylengths= range(0,11+1)
datapoints=1000
datapointsexact=100000
def b(x):
    return np.array([gamma*x[0, :], delta*(x[1, :] - x[0, :]**2)])
#We define a list in which we store the error of the eigenvalues for each dictionary length
eigenvalueserrorlist=[]
#We define a list in which we store the norm of the first evs elements of evdistances for each dictionary length 
evdistancesnorm=np.zeros((len(dictionarylengths)))
#We define a list in which we store the first evs elements of evdistances for each dictionary length
evdistanceslist=[]
# we define a list in which we store the singular values of Kexact-K for each dictionary length
operatorerror=np.zeros((len(dictionarylengths)))
#We define a list in which we store the operator norm of Kexact for each dictionary length
operatornormKexact=np.zeros((len(dictionarylengths)))


for i in dictionarylengths:
    # generate data
    Xexact = Omega.rand(datapointsexact) 
    Yexact = b(Xexact)
    X=Omega.rand(datapoints)
    Y=b(X)
    #dictionary
    psi = observables.monomials(i)
    #We calculate as many eigenvalues and eigenfunctions as we have observations in the dictionary psi
    evs = int((i+1)*(i+2)/2)
    # apply generator EDMD
    Kexact, dexact, Vexact = algorithms.gedmd(Xexact, Yexact, None, psi, evs=evs, operator='K')
    K, d, V = algorithms.gedmd(X, Y, None, psi, evs=evs, operator='K')
    #this calculates the sorted eigenvalues of Kexact and K
    eigenvaluesKexact=np.sort(np.linalg.eigvals(Kexact))
    eigenvaluesK=np.sort(np.linalg.eigvals(K))
    #this calculates the error between the eigenvalues of Kexact and K
    eigenvalueserror=np.linalg.norm(eigenvaluesKexact-eigenvaluesK)
    eigenvalueserrorlist.append(eigenvalueserror)
    #This normalizes the columns of V by dividing by their norm
    Vnormalizedexact=np.zeros((Vexact.shape[0],Vexact.shape[1]))
    Vnormalized=np.zeros((Vexact.shape[0],Vexact.shape[1]))
    for l in range(Vexact.shape[1]):
        Vnormalizedexact[:,l]=Vexact[:,l]/np.linalg.norm(Vexact[:,l])
        Vnormalized[:,l]=V[:,l]/np.linalg.norm(V[:,l])
    
    
    

       
    #We calculate the distance between the eigefunction i of the two operators V and Vexact for each i
    evdistances = []
    for j in range(evs):
        for k in range(evs):
            evdistances.append(np.linalg.norm(Vnormalizedexact[:,j]-Vexact[:, k]))
            evdistances.append(np.linalg.norm(Vnormalizedexact[:,j]+Vexact[:, k]))
    evdistances.sort()
    
    evdistanceslist.append(evdistances[:evs])
     
    evdistancesnorm[i]=np.linalg.norm(evdistances[:evs])
    print("Error of "+ str(evs)+ " normalized eigenfunctions")
    print(evdistances[:evs])
    print("error norm of eigenvalues "+str(np.linalg.norm(evdistances[:evs]) ))

    #We calculate the singular values of Kexact-K
    u,s,vh=np.linalg.svd(Kexact-K)
    #Then we take the square root of the largest singular value
    print("operator norm of Kexact-K for "+ str(i)+ " dictionary lengths")
    operatorerror[i]=np.max(s)
    #this calculates the operator norm of Kexact using its singular values
    u,s,vh=np.linalg.svd(Kexact)
    #Now we add it to the list of operator norms
    operatornormKexact[i]=np.max(s)
print("operator norm of Kexact-K for each dictionary length")
print(operatorerror)
print(evdistancesnorm) 


numberobservables=np.zeros((len(dictionarylengths)))
for i in range(len(dictionarylengths)):
    numberobservables[i]=int((i+1)*(i+2)/2)

# Log-Log-plot the operator norm of Kexact versus the number of observables
plt.figure()
plt.loglog(numberobservables,operatornormKexact)
#Line of log-log plot with slope 1 to see if the operator norm of Kexact is proportional to the number of observables
plt.loglog(numberobservables,np.power(numberobservables,1)*operatornormKexact[1]/np.power(numberobservables[1],1))
#Line of log-log plot with slope 2 to see if the operator norm of Kexact is proportional to the number of observables squared
plt.loglog(numberobservables,np.power(numberobservables,2)*operatornormKexact[1]/np.power(numberobservables[1],2))
#legends
plt.legend(['operator norm of Kexact','slope 1','slope 2'])
plt.xlabel('number of observables')
plt.ylabel('operator norm of Kexact')


plt.figure()
# log log plot the norm of the eigenvectors vs the number of dictionary lengths
plt.loglog(numberobservables[5:11],evdistancesnorm[5:11])
# log log plot the error of the operators vs the number of observables
plt.loglog(numberobservables[1:11],operatorerror[1:11])
# log log plot of the error of the eigenvalues vs the number of observables
plt.loglog(numberobservables[1:11],eigenvalueserrorlist[1:11])
# also plot a line with log log slope 2 to see if the error of the operators is proportional to the number of observables squared
plt.loglog(numberobservables[1:11],np.power(numberobservables[1:11],2)*operatorerror[1]/np.power(numberobservables[1],2))
# also plot a line with log log slope 1.5 to see if the error of the operators is proportional to the number of observables to the power 1.5
plt.loglog(numberobservables[1:11],np.power(numberobservables[1:11],1.5)*operatorerror[1]/np.power(numberobservables[1],1.5))

plt.xlabel('number of observables')

#plot legends
plt.legend(['norm of eigenvectors','error of operators','error of eigenvalues','slope 2','slope 1.5'])
plt.title('log-log-plot of error of operators vs number of observables')
plt.show()


# repeat code lines 60-100 10 times and take the average of evdistancesnorm and operatorerror
M=10
#We define a list in which we store the norm of the first evs elements of evdistances for each dictionary length and each run
evdistancesnorms=np.zeros((len(dictionarylengths),M))
#We define a list in which we store the first evs elements of evdistances for each dictionary length
evdistanceslists=[]
# We define a list in which we store the eigenvalueserror for each dictionary length and each run
eigenvalueserrorlists=np.zeros((len(dictionarylengths),M))
# we define a list in which we store the singular values of Kexact-K for each dictionary lengthand each run
operatorerrors=np.zeros((len(dictionarylengths),M))
#We define a list in which we store the average of the norm of the first evs elements of evdistances for each dictionary length
evdistancesnormaverage=np.zeros((len(dictionarylengths)))
operatorerroraverage=np.zeros((len(dictionarylengths)))
eigenvalueserroraverage=np.zeros((len(dictionarylengths)))

#We repeat the above for M runs
for m in range(M):
    for i in dictionarylengths:
        # generate data
        Xexact = Omega.rand(datapointsexact) 
        Yexact = b(Xexact)
        X=Omega.rand(datapoints)
        Y=b(X)
        #dictionary
        psi = observables.monomials(i)
        #We calculate as many eigenvalues and eigenfunctions as we have observations in the dictionary psi
        evs = int((i+1)*(i+2)/2)
        # apply generator EDMD
        Kexact, dexact, Vexact = algorithms.gedmd(Xexact, Yexact, None, psi, evs=evs, operator='K')
        K, d, V = algorithms.gedmd(X, Y, None, psi, evs=evs, operator='K')

        #this calculates the sorted eigenvalues of Kexact and K
        eigenvaluesKexact=np.sort(np.linalg.eigvals(Kexact))
        eigenvaluesK=np.sort(np.linalg.eigvals(K))
        #this calculates the error between the eigenvalues of Kexact and K
        eigenvalueserror=np.linalg.norm(eigenvaluesKexact-eigenvaluesK)
        eigenvalueserrorlists[i,m]=eigenvalueserror
        
        #This normalizes the columns of V by dividing by their norm
        Vnormalizedexact=np.zeros((Vexact.shape[0],Vexact.shape[1]))
        Vnormalized=np.zeros((Vexact.shape[0],Vexact.shape[1]))
        for l in range(Vexact.shape[1]):
            Vnormalizedexact[:,l]=Vexact[:,l]/np.linalg.norm(Vexact[:,l])
            Vnormalized[:,l]=V[:,l]/np.linalg.norm(V[:,l])
        
        #We calculate the distance between the eigefunction i of the two operators V and Vexact for each i
        evdistances = []
        for j in range(evs):
            for k in range(evs):
                evdistances.append(np.linalg.norm(Vnormalizedexact[:,j]-Vexact[:, k]))
                evdistances.append(np.linalg.norm(Vnormalizedexact[:,j]+Vexact[:, k]))
        evdistances.sort()
        
        evdistanceslists.append(evdistances[:evs])
        
        evdistancesnorms[i,m]=np.linalg.norm(evdistances[:evs])
        #We calculate the singular values of Kexact-K
        u,s,vh=np.linalg.svd(Kexact-K)
        #Then we take the largest singular value
        operatorerrors[i,m]=np.max(s)
#average over the runs
for i in dictionarylengths:
    evdistancesnormaverage[i]=np.average(evdistancesnorms[i,:])
    operatorerroraverage[i]=np.average(operatorerrors[i,:])
    eigenvalueserroraverage[i]=np.average(eigenvalueserrorlists[i,:])
#plots
plt.figure()
# log log plot the average error of the eigenvectors vs the number of dictionary lengths
plt.loglog(numberobservables[5:11],evdistancesnormaverage[5:11])
# log log plot the average error of the operators vs the number of observables
plt.loglog(numberobservables[1:11],operatorerroraverage[1:11])
# log log plot of the error of the eigenvalues vs the number of observables
plt.loglog(numberobservables[1:11],eigenvalueserroraverage[1:11])
# also plot a line with log log slope 1 to see if the error of the operators is proportional to the number of observables squared
plt.loglog(numberobservables[1:11],np.power(numberobservables[1:11],2)*operatorerroraverage[1]/np.power(numberobservables[1],2))
# also plot a line with log log slope 1,25 to see if the error of the operators is proportional to the number of observables squared
plt.loglog(numberobservables[1:11],np.power(numberobservables[1:11],1.5)*operatorerroraverage[1]/np.power(numberobservables[1],1.5))
# plot legends
plt.legend(['error of eigenvectors','error of operators',"error of eigenvalues",'slope 2','slope 1.5'])
plt.xlabel('number of observables')
plt.show()
1-1
1-1    