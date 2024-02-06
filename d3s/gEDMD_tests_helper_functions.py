import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.stats as stats

#Function to calculate operator error, eigenvalue error and eigenvector error, given
# data points for the exact and the approximate system
# a collection of observables
# and the drift function b
#and the domain Omega (by default a square [-1,1]^2)
import d3s.algorithms as algorithms
import d3s.domain as domain
import d3s.observables as observables


def gedmdMatrices(X, psi, b, Omega, sigma=None):
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
    G (numpy.ndarray): The Gramm matrix.
    C (numpy.ndarray): The stiffness matrix.

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
    A, G, C = gedmd_helper.gedmdMatrix(Xexact, X, psi, b, Omega=Omega)

    ```
    """
    Y = b(X)
    PsiX = psi(X)
    dPsiY = np.einsum('ijk,jk->ik', psi.diff(X), Y)

    if not (sigma is None):  # stochastic dynamical system
        Z = sigma(X)
        n = PsiX.shape[0]  # number of basis functions
        ddPsiX = psi.ddiff(X)  # second-order derivatives
        S = np.einsum('ijk,ljk->ilk', Z, Z)  # sigma \cdot sigma^T
        for i in range(n):
            dPsiY[i, :] += 0.5 * np.sum(ddPsiX[i, :, :, :] * S, axis=(0, 1))

    G = PsiX @ PsiX.T
    C = PsiX @ dPsiY.T
    T = dPsiY @ dPsiY.T

    #A = sp.linalg.pinv(G) @ C.T
    A = sp.linalg.solve(G, C.T)

    return A, G, C, T


def theoretical_error(gamma, A_exact_norm, G_exact_norm, C_exact_norm, M):
    """
    Calculate the theoretical operator error between an exact system and an approximate system.

    Parameters:
    gamma (float): The upper bound on the |basis functions|_infty and |operator applied to the basis functions|_infty
    A_exact_norm (float): The operator norm of the exact system.
    G_exact_norm (float): The operator norm of the Gramm matrix of the exact system.
    C_exact_norm (float): The operator norm of the stiffness matrix of the exact system.
    M (int): The number of data points.

    Returns:
    theoretical_operator_error (float): The theoretical operator error between the exact and approximate system.

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

    M=10000
    Xexact = Omega.rand(1000000) 
    X=Omega.rand(M)
    psi = observables.monomials(8)
        A, G, C = gedmd_helper.gedmdMatrix(Xexact, X, psi, b, Omega=Omega)
    A_exact_norm = sp.linalg.norm(A, 2)
    G_exact_norm = sp.linalg.norm(G, 2)
    C_exact_norm = sp.linalg.norm(C, 2)
    theoretical_operator_error = gedmd_helper.theoretical_error(gamma, A_exact_norm, G_exact_norm, C_exact_norm, M)

    ```
    """
    return np.sqrt(gamma)


def plot_errors_data_limit(
    M,
    min_number_of_data_points,
    confidence_level,
    number_of_runs,
    number_of_batches,
    observables_list,
    observables_names,
    Omega,
    b,
    sigma=None,
    block=True,
):
    """
    Plots the matrix error for different number of data points and observables.

    Parameters:
    M (int): The number of data points.
    min_number_of_data_points (int): The minimum number of data points.
    confidence_level (float): The confidence level for the confidence intervals.
    number_of_runs (int): The number of runs.
    number_of_batches (int): The number of batches.
    observables_list (list): A list of observables.
    observables_names (list): A list of names for the observables.
    Omega (d3s.domain.discretization, optional): The domain, by default a square [-1,1]^2. Defaults to Omega.
    b (function): The drift function.
    sigma (function, optional): The diffusion function. Defaults to None.
    block (bool, optional): Whether to block the plot. Defaults to True.

    Example:
    ```python
 M = 20000
number_of_runs = 20
number_of_batches = 5
confidence_level = 0.95
number_of_monomials = 8
observables_names = ['Monomials', 'Gaussians']
min_number_of_data_points = 200

# ########################################
#Simple deterministic system
# ########################################
# define domain
bounds = np.array([[-2, 2], [-1, 1]])
boxes = np.array([9, 5])
Omega = domain.discretization(bounds, boxes)

# define system
gamma = -0.8
delta = -0.7


def b(x):
    return np.array([gamma * x[0, :], delta * (x[1, :] - x[0, :]**2)])


# define observables
psi_m = observables.monomials(number_of_monomials)
variance = (bounds[0, 1] - bounds[0, 0]) / boxes[0] / 2
psi_g = observables.gaussians(Omega, sigma=variance)
observables_list = [psi_m, psi_g]
gedmd_helper.plot_errors_data_limit(M,
                                    min_number_of_data_points,
                                    confidence_level,
                                    number_of_runs,
                                    number_of_batches,
                                    observables_list,
                                    observables_names,
                                    Omega,
                                    b,
                                    block=True)
    """
    # generate data
    number_of_loops_data_points = int(
        np.floor(np.log2(M / min_number_of_data_points))) + 1

    data_points_number = [
        min_number_of_data_points * 2**x
        for x in range(0, number_of_loops_data_points)
    ]
    print('max data_points_number = ',
          min_number_of_data_points * 2**number_of_loops_data_points,
          'number_of_loops = ', number_of_loops_data_points)
    types_of_observables_number = len(observables_list)
    A_exact = []
    A_exact_matrix_norm = []
    matrix_errors = np.zeros((number_of_loops_data_points,
                              types_of_observables_number, number_of_runs))
    matrix_errors_average = np.zeros(
        (number_of_loops_data_points, types_of_observables_number))
    X_exact = Omega.rand(M)

    for type in range(types_of_observables_number):
        # Exacts operators are the same over all runs to save time
        A_ex, _, _ = gedmdMatrices(X_exact, observables_list[type], b, Omega,
                                   sigma)
        A_exact.append(A_ex)
        A_exact_matrix_norm.append(np.linalg.norm(A_ex, ord=2))

        for m in range(number_of_runs):
            print('runs completed = ', m, '/', number_of_runs, "type = ",
                  observables_names[type])
            for i in range(number_of_loops_data_points):
                X = Omega.rand(data_points_number[i])
                A, _, _ = gedmdMatrices(X, observables_list[type], b, Omega,
                                        sigma)
                matrix_errors[i, type, m] = np.linalg.norm(
                    A_exact[type] - A, ord=2) / A_exact_matrix_norm[type]

        matrix_errors_average = np.mean(matrix_errors, axis=2)
    #calculate confidence intervals for the average error of the matrices (95% confidence) for each number of data points
    #first we divide the error data into number_of_batches batches
    batch_size = int(np.floor(
        number_of_runs / number_of_batches))  #number of runs in each batch
    #Gives error if batch size is 0
    if batch_size == 0:
        raise Exception(
            'batch size is 0. Please increase number of runs or decrease number of batches'
        )
    matrix_errors_batches = np.zeros(
        (number_of_loops_data_points, types_of_observables_number,
         number_of_batches))
    for nb in range(number_of_batches):
        matrix_errors_batches[:, :, nb] = np.mean(
            matrix_errors[:, :, nb * batch_size:(nb + 1) * batch_size], axis=2
        )  #average error over runs divided into batches for each number of data points and each type of observable
    #now we calculate the average and standard deviation of each batch
    matrix_errors_average = np.mean(matrix_errors_batches, axis=2)
    matrix_errors_std = np.std(matrix_errors_batches, axis=2, ddof=1)
    #the batch averages can be interpreted as being Gaussian for large number of runs
    #so we can calculate the confidence intervals using student's t-distribution
    #we define the t_value for 95% confidence and 9 degrees of freedom
    t_value = stats.t.ppf((1 + confidence_level) / 2, number_of_batches - 1)
    matrix_errors_confidence_interval = t_value * matrix_errors_std / np.sqrt(
        number_of_batches)

    #error plots
    # ... rest of your code ...

    # error plots
    plt.figure()
    plt.loglog(data_points_number, matrix_errors_average)

    # plot confidence intervals as shaded regions
    lower_bound = matrix_errors_average - matrix_errors_confidence_interval
    upper_bound = matrix_errors_average + matrix_errors_confidence_interval
    plt.fill_between(data_points_number,
                     lower_bound[:, 0],
                     upper_bound[:, 0],
                     color='blue',
                     alpha=0.2)
    plt.fill_between(data_points_number,
                     lower_bound[:, 1],
                     upper_bound[:, 1],
                     color='orange',
                     alpha=0.2)

    # slopes
    plt.loglog(
        data_points_number,
        np.power(np.float64(data_points_number), -1) *
        matrix_errors_average[0, 1] /
        np.power(np.float64(data_points_number[0]), -1))
    plt.loglog(
        data_points_number,
        np.power(np.float64(data_points_number), -0.5) *
        matrix_errors_average[0, 1] /
        np.power(np.float64(data_points_number[0]), -0.5))
    plt.xlabel('number of data points M')

    # plot legends
    for type in range(types_of_observables_number):
        plt.legend(observables_names[type])

    observables_error_labels = [f'{name} error' for name in observables_names]
    CI_labels = [f'CI {name}' for name in observables_names]
    legend_labels = observables_error_labels + CI_labels + [
        '$M^{-1}$', '$M^{-0.5}$'
    ]

    plt.legend(legend_labels)
    plt.title('log-log-plot of error in matrix norm vs number of observables')
    print('The number of observables of each method is:')

    for i in range(len(observables_list)):
        print(observables_names[i], ':', observables_list[i].length())
    plt.show(block=block)


def plot_errors_dictionary_limit(min_number_of_observables,
                                 max_number_of_observables,
                                 confidence_level,
                                 number_of_runs,
                                 number_of_batches,
                                 observables_list,
                                 observables_names,
                                 Omega,
                                 b,
                                 sigma=None,
                                 block=True,
                                 M_exact=None,
                                 M_approx=None,
                                 p=0.5):
    """
    Plots the matrix error for different number of dictionary elements and observables.

    Parameters:
    min_number_of_observables (int): The minimum number of dictionary functions.
    max_number_of_observables (int): The maximum number of dictionary functions.
    confidence_level (float): The confidence level for the confidence intervals.
    number_of_runs (int): The number of runs.
    number_of_batches (int): The number of batches.
    observables_list (list): A list of observables.
    observables_names (list): A list of names for the observables.
    Omega (d3s.domain.discretization, optional): The domain, by default a square [-1,1]^2. Defaults to Omega.
    b (function): The drift function.
    sigma (function, optional): The diffusion function. Defaults to None.
    block (bool, optional): Whether to block the plot. Defaults to True.
    M (int): The number of data points. Defaults to None in which case is calculated according to theoretical error.

    Example:
    ```python
 M_exact = 20000
 M_approx = 10000
number_of_runs = 20
number_of_batches = 5
confidence_level = 0.95
number_of_monomials = 8
observables_names = ['Monomials', 'Gaussians']

# ########################################
#Simple deterministic system
# ########################################
# define domain
bounds = np.array([[-2, 2], [-1, 1]])
boxes = np.array([9, 5])
Omega = domain.discretization(bounds, boxes)

# define system
gamma = -0.8
delta = -0.7


def b(x):
    return np.array([gamma * x[0, :], delta * (x[1, :] - x[0, :]**2)])


# define observables
psi_m = observables.monomials(number_of_monomials)
variance = (bounds[0, 1] - bounds[0, 0]) / boxes[0] / 2
psi_g = observables.gaussians(Omega, sigma=variance)
observables_list = [psi_m, psi_g]
gedmd_helper.plot_errors_dictionary_limit(min_number_of_dictionary_functions,
                                    confidence_level,
                                    number_of_runs,
                                    number_of_batches,
                                    observables_list,
                                    observables_names,
                                    Omega,
                                    b,
                                    block=True,
                                    M_exact = M_exact,
                                    M = M,
                                    p =0.9)
    """
    # generate data
    number_of_loops_observables = int(
        np.floor(np.log2(
            max_number_of_observables / min_number_of_observables))) + 1

    observables_numbers = [
        min_number_of_observables * 2**x
        for x in range(0, max_number_of_observables)
    ]
    print('max observables = ',
          min_number_of_observables * 2**number_of_loops_observables,
          'number_of_loops = ', number_of_loops_observables)
    types_of_observables_number = len(observables_list)
    A_exact = []
    A_exact_matrix_norm = []
    matrix_errors = np.zeros((number_of_loops_observables,
                              types_of_observables_number, number_of_runs))
    matrix_errors_average = np.zeros(
        (number_of_loops_observables, types_of_observables_number))
    X_exact = Omega.rand(M_exact)
    X_approx = Omega.rand(M_approx)

    for type in range(types_of_observables_number):
        # Exacts operators are the same over all runs to save time
        A_ex, _, _ = gedmdMatrices(X_exact, observables_list[type], b, Omega,
                                   sigma)
        A_exact.append(A_ex)
        A_exact_matrix_norm.append(np.linalg.norm(A_ex, ord=2))

        for m in range(number_of_runs):
            print('runs completed = ', m, '/', number_of_runs, "type = ",
                  observables_names[type])
            for i in range(number_of_loops_data_points):
                X = Omega.rand(data_points_number[i])
                A, _, _ = gedmdMatrices(X, observables_list[type], b, Omega,
                                        sigma)
                matrix_errors[i, type, m] = np.linalg.norm(
                    A_exact[type] - A, ord=2) / A_exact_matrix_norm[type]

        matrix_errors_average = np.mean(matrix_errors, axis=2)
    #calculate confidence intervals for the average error of the matrices (95% confidence) for each number of data points
    #first we divide the error data into number_of_batches batches
    batch_size = int(np.floor(
        number_of_runs / number_of_batches))  #number of runs in each batch
    #Gives error if batch size is 0
    if batch_size == 0:
        raise Exception(
            'batch size is 0. Please increase number of runs or decrease number of batches'
        )
    matrix_errors_batches = np.zeros(
        (number_of_loops_data_points, types_of_observables_number,
         number_of_batches))
    for nb in range(number_of_batches):
        matrix_errors_batches[:, :, nb] = np.mean(
            matrix_errors[:, :, nb * batch_size:(nb + 1) * batch_size], axis=2
        )  #average error over runs divided into batches for each number of data points and each type of observable
    #now we calculate the average and standard deviation of each batch
    matrix_errors_average = np.mean(matrix_errors_batches, axis=2)
    matrix_errors_std = np.std(matrix_errors_batches, axis=2, ddof=1)
    #the batch averages can be interpreted as being Gaussian for large number of runs
    #so we can calculate the confidence intervals using student's t-distribution
    #we define the t_value for 95% confidence and 9 degrees of freedom
    t_value = stats.t.ppf((1 + confidence_level) / 2, number_of_batches - 1)
    matrix_errors_confidence_interval = t_value * matrix_errors_std / np.sqrt(
        number_of_batches)

    #error plots
    # ... rest of your code ...

    # error plots
    plt.figure()
    plt.loglog(data_points_number, matrix_errors_average)

    # plot confidence intervals as shaded regions
    lower_bound = matrix_errors_average - matrix_errors_confidence_interval
    upper_bound = matrix_errors_average + matrix_errors_confidence_interval
    plt.fill_between(data_points_number,
                     lower_bound[:, 0],
                     upper_bound[:, 0],
                     color='blue',
                     alpha=0.2)
    plt.fill_between(data_points_number,
                     lower_bound[:, 1],
                     upper_bound[:, 1],
                     color='orange',
                     alpha=0.2)

    # slopes
    plt.loglog(
        data_points_number,
        np.power(np.float64(data_points_number), -1) *
        matrix_errors_average[0, 1] /
        np.power(np.float64(data_points_number[0]), -1))
    plt.loglog(
        data_points_number,
        np.power(np.float64(data_points_number), -0.5) *
        matrix_errors_average[0, 1] /
        np.power(np.float64(data_points_number[0]), -0.5))
    plt.xlabel('number of data points M')

    # plot legends
    for type in range(types_of_observables_number):
        plt.legend(observables_names[type])

    observables_error_labels = [f'{name} error' for name in observables_names]
    CI_labels = [f'CI {name}' for name in observables_names]
    legend_labels = observables_error_labels + CI_labels + [
        '$M^{-1}$', '$M^{-0.5}$'
    ]

    plt.legend(legend_labels)
    plt.title('log-log-plot of error in matrix norm vs number of observables')
    print('The number of observables of each method is:')

    for i in range(len(observables_list)):
        print(observables_names[i], ':', observables_list[i].length())
    plt.show(block=block)


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
    A, _, _ = gedmdMatrices(X, psi, b, Omega)
    A_exact, G_exact, C_exact = gedmdMatrices(X_exact, psi, b, Omega)

    #Operator norms
    operator_norm_A_exact = sp.linalg.norm(A_exact, 2)
    operator_norm_G_exact = sp.linalg.norm(G_exact, 2)
    operator_norm_C_exact = sp.linalg.norm(C_exact, 2)

    #Operator error
    operator_norm_error = sp.linalg.norm(A - A_exact, 2)
    operator_error_rescaled = operator_norm_error / operator_norm_A_exact

    return operator_error_rescaled, operator_norm_A_exact, operator_norm_G_exact, operator_norm_C_exact


#In the notation we use in our new paper:
# C= <A psi,psi>, where A is the generator of the system
# G = <psi,psi>, is the Gramm matrix
# A =G^{-1} C^T is the operator matrix
# In the notation in the old paper by Stefan et al:
# A = operator matrix = A
# C0 = gram matrix = G
# C1 = stiffness matrix = C
