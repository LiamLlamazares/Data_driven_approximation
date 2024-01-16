import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

#Function to calculate operator error, eigenvalue error and eigenvector error, given
# data points for the exact and the approximate system
# a collection of observables
# and the drift function b
#and the domain Omega (by default a square [-1,1]^2)
import d3s.algorithms as algorithms
import d3s.domain as domain
import d3s.observables as observables


def gedmdMatrix(X, psi, b, Omega):
    """
    Calculates the gEDMD matrix for the ODE dX=b(X)dt.

    Parameters:
    X (numpy.ndarray): Data points for the system.
    psi (function): A collection of observables.
    b (function): The drift function.
    evs (int, optional): Number of eigenvalues/eigenfunctions to be computed. Defaults to 8.
    Omega (d3s.domain.discretization, optional): The domain, by default a square [-1,1]^2. Defaults to Omega.

    Returns:
    A (numpy.ndarray): The gEDMD matrix.

    Example:
    ```python
    bounds = np.array([[-1, 1], [-1, 1]])
    boxes = np.array([50, 50])
    Omega = domain.discretization(bounds, boxes)
    # define system
    gamma = -0.8
    delta = -0.7

    #This corresponds to the ODE dx1=gamma*x1dt, dx2=delta*(x2-x1^2)dt
    def b(x):
        return np.array([gamma*x[0, :], delta*(x[1, :] - x[0, :]**2)])
    
    X=Omega.rand(10000)
    psi = observables.monomials(i)
    A=gedmd_helper.gedmdMatrix(Xexact, X, psi, b, Omega=Omega)

    ```
    """
    Y = b(X)

    PsiX = psi(X)
    dPsiY = np.einsum('ijk,jk->ik', psi.diff(X), Y)
    G = PsiX @ PsiX.T
    C = PsiX @ dPsiY.T

    A = sp.linalg.pinv(G) @ C
    return A


def gedmdErrors(X_exact, X, psi, b, Omega):
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
    frobenius_error_rescaled (float): The Frobenius norm between the exact and approximate system.
    eigenvalue_error_rescaled (float): The rescaled eigenvalue error between the exact and approximate system.
    eigenfunction_error (float): The error between the eigenfunctions of the exact and approximate system.
    operator_norm_K_exact (float): The operator norm of the exact system.

    Example:
    ```python
    bounds = np.array([[-1, 1], [-1, 1]])
    boxes = np.array([50, 50])
    Omega = domain.discretization(bounds, boxes)
    # define system
    gamma = -0.8
    delta = -0.7

    #This corresponds to the ODE dx1=gamma*x1dt, dx2=delta*(x2-x1^2)dt
    def b(x):
        return np.array([gamma*x[0, :], delta*(x[1, :] - x[0, :]**2)])
    
    Xexact = Omega.rand(1000000) 
    X=Omega.rand(10000)
    psi = observables.monomials(i)
    operator_error, frobenius_operator_error, eigenvalue_error, operator_norm_K_exact=gedmd_helper.gedmdErrors(Xexact, X, psi, b, Omega=Omega)

    ```
    """
    #Calculate the operator matrix
    A = gedmdMatrix(X, psi, b, Omega)
    A_exact = gedmdMatrix(X_exact, psi, b, Omega)

    #Operator norm error
    operator_norm_K_exact = sp.linalg.norm(A_exact, 2)
    operator_norm_error = sp.linalg.norm(A - A_exact, 2)
    operator_error_rescaled = operator_norm_error / operator_norm_K_exact

    #Frobeinus norm error
    frobenius_error_rescaled = np.linalg.norm(A_exact -
                                              A) / operator_norm_K_exact

    #Eigenvalue error
    eigenvalues_K_exact = np.sort(np.linalg.eigvals(A_exact))
    eigenvalues_K = np.sort(np.linalg.eigvals(A))
    eigenvalue_error_rescaled = np.linalg.norm(
        eigenvalues_K_exact - eigenvalues_K) / operator_norm_K_exact

    return operator_error_rescaled, frobenius_error_rescaled, eigenvalue_error_rescaled, operator_norm_K_exact


#In the notation we use in our new paper:
# C= <A psi,psi>, where A is the generator of the system
# G = <psi,psi>, is the Gramm matrix
# A =G^{-1} C^T is the operator matrix
# In the notation in the old paper by Stefan et al:
# A = operator matrix = A
# C0 = gram matrix = G
# C1 = stiffness matrix = C
