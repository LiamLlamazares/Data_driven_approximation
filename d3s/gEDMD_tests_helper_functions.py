import numpy as np
import matplotlib.pyplot as plt

#Function to calculate operator error, eigenvalue error and eigenvector error, given
# data points for the exact and the approximate system
# a collection of observables
# and the drift function b
#and the domain Omega (by default a square [-1,1]^2)
import d3s.algorithms as algorithms
import d3s.domain as domain
import d3s.observables as observables

bounds = np.array([[-1, 1], [-1, 1]])
boxes = np.array([50, 50])
Omega = domain.discretization(bounds, boxes)

def gedmdErrors(X_exact, X, psi, b, Omega=Omega):
    """
    Calculate the operator error, eigenvalue error, and eigenfunction error between an exact system and an approximate system.

    Parameters:
    X_exact (numpy.ndarray): Data points for the exact system.
    X (numpy.ndarray): Data points for the approximate system.
    psi (function): A collection of observables.
    b (function): The drift function.
    evs (int, optional): Number of eigenvalues/eigenfunctions to be computed. Defaults to 8.
    Omega (d3s.domain.discretization, optional): The domain, by default a square [-1,1]^2. Defaults to Omega.

    Returns:
    operator_error_rescaled (float): The rescaled operator error between the exact and approximate system.
    eigenvalue_error_rescaled (float): The rescaled eigenvalue error between the exact and approximate system.
    eigenfunction_error (float): The error between the eigenfunctions of the exact and approximate system.

    Example:
    ```python
    import numpy as np
    import d3s.algorithms as algorithms
    import d3s.domain as domain
    import d3s.observables as observables
    import d3s.gEDMD_tests_helper_functions as gedmd_helper

    ```
    """
    evs= psi.length(X)
    Yexact = b(X_exact)
    Y = b(X)
    Kexact, dexact, V_exact = algorithms.gedmd(X_exact, Yexact, None, psi, evs=evs, operator='K')

    K, d, V = algorithms.gedmd(X, Y, None, psi, evs=evs, operator='K')
    
    #Operator norm error
    u,s,vh=np.linalg.svd(Kexact)
    ue,se,vhe=np.linalg.svd(Kexact-K)
    operator_norm_K_exact=s[0]
    operator_error_rescaled=se[0]/operator_norm_K_exact
    
    #Eigenvalue error
    eigenvalues_K_exact=np.sort(np.linalg.eigvals(Kexact))
    eigenvalues_K=np.sort(np.linalg.eigvals(K))
    eigenvalue_error_rescaled=np.linalg.norm(eigenvalues_K_exact-eigenvalues_K)/operator_norm_K_exact
    
    #Eigenfunction error
    V_normalized_exact=np.zeros((V_exact.shape[0],V_exact.shape[1]))
    V_normalized=np.zeros((V_exact.shape[0],V_exact.shape[1]))
    for l in range(V_exact.shape[1]):
        V_normalized_exact[:,l]=V_exact[:,l]/np.linalg.norm(V_exact[:,l])
        V_normalized[:,l]=V[:,l]/np.linalg.norm(V[:,l])
    eigenfunction_error = []
    #Need to compare the 'same' eigenfunctions, sign included
    for j in range(evs):
        for k in range(evs):
            eigenfunction_error.append(np.linalg.norm(V_normalized_exact[:,j]-V_normalized[:, k]))
            eigenfunction_error.append(np.linalg.norm(V_normalized_exact[:,j]+V_normalized[:, k]))
    # Create a new list with duplicates removed, but keep all zeros
    eigenfunction_error_no_duplicates = [0]*eigenfunction_error.count(0)
    eigenfunction_error_no_duplicates += list(set([i for i in eigenfunction_error if i != 0]))

    # Sort the list
    eigenfunction_error_no_duplicates.sort()
    eigenfunction_error = np.linalg.norm(eigenfunction_error_no_duplicates[:evs])
    
    return operator_error_rescaled, eigenvalue_error_rescaled, eigenfunction_error, operator_norm_K_exact