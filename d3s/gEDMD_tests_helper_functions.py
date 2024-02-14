import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.stats as stats

#Function to calculate operator error, eigenvalue error and eigenvector error, given
# data points for the exact and the approximate system
# a collection of observables
# and the drift function b
#and the domain Omega (by default a square [-1,1]^2)
import d3s.domain as domain
import d3s.observables as observables


def gedmdMatrices(X,
                  psi,
                  b,
                  Omega,
                  sigma=None,
                  f=None,
                  sigma_noise=0,
                  operator='K'):
    """
    Calculates the gEDMD matrix for the ODE dX=b(X)dt.

    Parameters:
    X (numpy.ndarray): Data points for the system.
    psi (function): A collection of observables.
    b (function): The drift function.
    evs (int, optional): Number of eigenvalues/eigenfunctions to be computed. Defaults to 8.
    Omega (d3s.domain.discretization, optional): The domain, by default a square [-1,1]^2. Defaults to Omega.
    sigma (function, optional): The diffusion function. Only used for gEDMD in stochastic systems. Defaults to None.
    f (function, optional): The forward operator. Only used for EDMD. Defaults to None.
    sigma_noise (float, optional): The standard deviation of the noise added to the observations psi_i Apsi_i. Defaults to 0.

    Returns:
    A (numpy.ndarray): The gEDMD matrix.
    G (numpy.ndarray): The Gramm matrix.
    C (numpy.ndarray): The stiffness matrix.
    T (numpy.ndarray): The graph matrix <L psi_i, L psi_j>.
    uniform_norm_psi_A_psi (float): The uniform norm of (psi,A psi)

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
    A, G, C, T = gedmd_helper.gedmdMatrices(Xexact, X, psi, b, Omega=Omega)

    ```
    """
    if f is None:  #gEDMD
        Y = b(X)
        PsiX = psi(X)
        dPsiY = np.einsum('ijk,jk->ik', psi.diff(X), Y)

        if not (sigma is None):  # stochastic dynamical system
            Z = sigma(X)
            n = PsiX.shape[0]  # number of basis functions
            ddPsiX = psi.ddiff(X)  # second-order derivatives
            S = np.einsum('ijk,ljk->ilk', Z, Z)  # sigma \cdot sigma^T
            for i in range(n):
                dPsiY[i, :] += 0.5 * np.sum(ddPsiX[i, :, :, :] * S,
                                            axis=(0, 1))

        if not (sigma_noise is None):  #Noise if added
            PsiX += sigma_noise * np.random.randn(*PsiX.shape)
            dPsiY += sigma_noise * np.random.randn(*dPsiY.shape)

        G = PsiX @ PsiX.T
        C = PsiX @ dPsiY.T
        if operator == 'P': C = C.T
        T = dPsiY @ dPsiY.T
        uniform_norm_psi_A_psi = max(PsiX.max(), dPsiY.max())
    else:  #EDMD
        Y = f(X)
        PsiX = psi(X)
        PsiY = psi(Y)

        if not (sigma_noise is None):  #Noise if added
            PsiX += sigma_noise * np.random.randn(*PsiX.shape)
            PsiY += sigma_noise * np.random.randn(*PsiY.shape)

        G = PsiX @ PsiX.T
        C = PsiX @ PsiY.T
        if operator == 'P': C = C.T
        T = PsiY @ PsiY.T
        uniform_norm_psi_A_psi = max(PsiX.max(), PsiY.max())

    #A = sp.linalg.pinv(G) @ C.T
    A = sp.linalg.solve(G, C)

    return A, G, C, T, uniform_norm_psi_A_psi


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
    f=None,
    sigma_noise=0,
    block=True,
    title=None,
    power_1=-1,
    power_2=-0.5,
    operator='K',
    path=None,
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
    sigma_noise (float, optional): The standard deviation of the noise added to the observations psi_i Apsi_i. Defaults to 0.
    f (function, optional): The forward operator for EDMD. Defaults to None.
    block (bool, optional): Whether to block the plot. Defaults to True.
    power_1 (float, optional): The power of the first slope. Defaults to -1.
    power_2 (float, optional): The power of the second slope. Defaults to -0.5.
    operator (str, optional): The operator to be used (Koopman, Perron Frobenius). Defaults to 'K'.
    path (str, optional): The path to save the plot. Defaults to None.

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
        np.floor(np.log2(M / min_number_of_data_points)))

    data_points_number = [
        min_number_of_data_points * 2**x
        for x in range(0, number_of_loops_data_points)
    ]
    print(title, ' max data_points_number = ',
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
        A_ex, _, _, _, _ = gedmdMatrices(X_exact,
                                         observables_list[type],
                                         b,
                                         Omega,
                                         sigma,
                                         f,
                                         operator=operator)
        A_exact.append(A_ex)
        A_exact_matrix_norm.append(np.linalg.norm(A_ex, ord=2))

        for m in range(number_of_runs):
            print('runs completed = ', m, '/', number_of_runs, "type = ",
                  observables_names[type])
            for i in range(number_of_loops_data_points):
                X = Omega.rand(data_points_number[i])
                A, _, _, _, _, = gedmdMatrices(X,
                                               observables_list[type],
                                               b,
                                               Omega,
                                               sigma,
                                               f,
                                               sigma_noise=sigma_noise,
                                               operator=operator)
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

    # error plots
    plt.figure()
    plt.loglog(data_points_number, matrix_errors_average, marker='o')

    # plot confidence intervals as shaded regions
    lower_bound = matrix_errors_average - matrix_errors_confidence_interval
    upper_bound = matrix_errors_average + matrix_errors_confidence_interval
    colours = [
        'blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray',
        'olive', 'cyan'
    ]

    for i in range(lower_bound.shape[1]):
        plt.fill_between(data_points_number,
                         lower_bound[:, i],
                         upper_bound[:, i],
                         color=colours[i % len(colours)],
                         alpha=0.2)
    # slopes
    plt.loglog(
        data_points_number,
        np.power(np.float64(data_points_number), power_1) *
        matrix_errors_average[0, 0] /
        np.power(np.float64(data_points_number[0]), power_1))
    plt.loglog(
        data_points_number,
        np.power(np.float64(data_points_number), power_2) *
        matrix_errors_average[0, 0] /
        np.power(np.float64(data_points_number[0]), power_2))
    plt.xlabel(f'$M$')
    plt.ylabel(f'$\epsilon$')

    # plot legends
    for type in range(types_of_observables_number):
        plt.legend(observables_names[type], loc='lower left')

    observables_error_labels = [f'{name} error' for name in observables_names]
    CI_labels = [f'CI {name}' for name in observables_names]
    legend_labels = observables_error_labels + CI_labels + [
        f'$M^{{{power_1}}}$', f'$M^{{{power_2}}}$'
    ]
    plt.legend(legend_labels, loc='lower left')
    if title is not None:
        title = (' for ' + title
                 if title else '') + (f' with noise $\sigma = {sigma_noise}$'
                                      if sigma_noise != 0 else '')
        plt.title('Error in matrix norm vs number of data points' + title)
    print('The number of observables of each method is:')

    for i in range(len(observables_list)):
        print(observables_names[i], ':', observables_list[i].length())

    plt.show(block=block)
    #Saves the data
    data = {
        'data_points_number': data_points_number,
        'matrix_errors_average': matrix_errors_average,
        'matrix_errors_confidence_interval': matrix_errors_confidence_interval,
        'observables_names': observables_names,
        'observables_list': observables_list,
        'title': path
    }
    np.savez('gEDMDtests/Simulation_data/Data_Limit/' + path + '.npz', **data)


def plot_errors_dictionary_limit(min_number_of_observables,
                                 max_number_of_observables,
                                 confidence_level,
                                 number_of_runs,
                                 number_of_batches,
                                 observables_names,
                                 Omega,
                                 b,
                                 sigma=None,
                                 f=None,
                                 sigma_noise=0,
                                 operator='K',
                                 gamma=1,
                                 multiplier=1.3,
                                 block=True,
                                 M_exact=None,
                                 M_approx=None,
                                 prob=0.5,
                                 title=None,
                                 power_1=0.5,
                                 power_2=0.25,
                                 path=None):
    """
    Plots the matrix error for different number of dictionary elements and observables.

    Parameters:
    min_number_of_observables (int): The minimum number of dictionary functions.
    max_number_of_observables (int): The maximum number of dictionary functions.
    confidence_level (float): The confidence level for the confidence intervals.
    number_of_runs (int): The number of runs.
    number_of_batches (int): The number of batches.
    observables_names (list): A list of names for the observables. Accepts 'Monomials' and 'Gaussians'.
    Omega (d3s.domain.discretization, optional): The domain, by default a square [-1,1]^2. Defaults to Omega.
    b (function): The drift function.
    sigma (function, optional): The diffusion function. Defaults to None.
    f (function, optional): The forward operator, defaults to None. If supplied, EDMD is used. Otherwise, gEDMD is used
    sigma_noise (float, optional): The standard deviation of the noise added to the observations psi_i A(psi_i). Defaults to 0.
    gamma (float,optional): The L^infinity (sup) norm of psi and A psi, defaults to 1
    block (bool, optional): Whether to block the plot. Defaults to True.
    M_exact (int, optional): The number of data points for the exact system. Defaults to None.
    M_approx (int, optional): The number of data points for the approximate system. Defaults to None.
    prob (float, optional): The probability of the error. Defaults to 0.5.
    title (str, optional): The title of the plot. Defaults to None.
    power_1 (float, optional): The power of the first slope. Defaults to 0.5.
    power_2 (float, optional): The power of the second slope. Defaults to 0.25.
    path (str, optional): The path to save the plot. Defaults to None.

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
                                    f=None,
                                    block=True,
                                    M_exact = M_exact,
                                    M = M_approx,
                                    p =0.9,
                                    title='Simple deterministic system')
    """
    p = 0  #Monomials degree
    while int((p + 1) * (p + 2) / 2) < max_number_of_observables:
        p = p + 1
    # get finer and finer mesh for gaussians
    boxes_list = [np.array(Omega._boxes)]
    Omega_list = [Omega]
    observables_numbers = [Omega.numBoxes()]
    while observables_numbers[-1] < max_number_of_observables:
        boxes_list.append(
            np.array([
                int(max(np.round(number), number + 1))
                for number in multiplier * np.array(boxes_list[-1])
            ]))
        Omega_list.append(domain.discretization(Omega._bounds, boxes_list[-1]))
        observables_numbers.append(Omega_list[-1].numBoxes())

    number_of_loops_observables = len(observables_numbers)

    print(title, ' max observables = ', observables_numbers[-1],
          'number_of_loops = ', number_of_loops_observables)
    types_of_observables_number = len(observables_names)
    matrix_errors = np.zeros((number_of_loops_observables,
                              types_of_observables_number, number_of_runs))
    matrix_errors_average = np.zeros(
        (number_of_loops_observables, types_of_observables_number))
    theoretical_errors = np.zeros(
        (number_of_loops_observables, types_of_observables_number))
    X_exact = Omega.rand(M_exact)
    for type in range(types_of_observables_number):
        for i in range(number_of_loops_observables):
            Omega = Omega_list[i]
            if observables_names[type] == 'Monomials':
                dictionary = observables.monomials(p=p,
                                                   n=observables_numbers[i])
            elif observables_names[type] == 'Gaussians':
                variance = (Omega._bounds[0, 1] -
                            Omega._bounds[0, 0]) / Omega._boxes[0] / 2
                dictionary = observables.gaussians(Omega, sigma=variance)
                print('Gaussians created', dictionary.length())
        # Exacts operators are the same over all runs to save time

            A_exact, G_exact, C_exact, T_exact, gamma = gedmdMatrices(
                X_exact, dictionary, b, Omega, sigma, f, operator=operator)
            A_exact_matrix_norm = np.linalg.norm(A_exact, ord=2)
            G_exact_matrix_norm = np.linalg.norm(G_exact, ord=2)
            _, s, _ = np.linalg.svd(G_exact)
            G_inv_exact_matrix_norm = 1 / np.min(s)
            C_exact_matrix_norm = np.linalg.norm(C_exact, ord=2)
            T_exact_matrix_norm = np.linalg.norm(T_exact, ord=2)
            theoretical_errors[i, type] = np.sqrt(
                (gamma + sigma_noise) *
                max(G_exact_matrix_norm, T_exact_matrix_norm) /
                max(A_exact_matrix_norm, 1) * C_exact_matrix_norm**2 *
                G_inv_exact_matrix_norm**4 * np.log(observables_numbers[i] /
                                                    (1 - prob)) /
                M_exact) / A_exact_matrix_norm

            print('loops completed = ', i, '/', number_of_loops_observables,
                  "type = ", observables_names[type])
            for m in range(number_of_runs):
                X_approx = Omega.rand(M_approx)
                A_approx, _, _, _, _ = gedmdMatrices(X_approx,
                                                     dictionary,
                                                     b,
                                                     Omega,
                                                     sigma,
                                                     f,
                                                     sigma_noise=sigma_noise,
                                                     operator=operator)
                matrix_errors[i, type, m] = np.linalg.norm(
                    A_exact - A_approx, ord=2) / A_exact_matrix_norm

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
        (number_of_loops_observables, types_of_observables_number,
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

    # error plots
    plt.figure()
    plt.loglog(observables_numbers, matrix_errors_average)

    # plot confidence intervals as shaded regions
    lower_bound = matrix_errors_average - matrix_errors_confidence_interval
    upper_bound = matrix_errors_average + matrix_errors_confidence_interval
    colours = [
        'blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray',
        'olive', 'cyan'
    ]

    for i in range(lower_bound.shape[1]):
        plt.fill_between(observables_numbers,
                         lower_bound[:, i],
                         upper_bound[:, i],
                         color=colours[i % len(colours)],
                         alpha=0.2)
    #Theoretical error starts from the average error of the first batch
    for i in range(types_of_observables_number):
        theoretical_errors[:,
                           i] = theoretical_errors[:,
                                                   i] + matrix_errors_average[
                                                       0,
                                                       i] - theoretical_errors[
                                                           0, i]
    plt.loglog(observables_numbers, np.minimum(theoretical_errors, 1))

    # slopes
    plt.loglog(
        observables_numbers,
        np.power(np.float64(observables_numbers), power_1) *
        matrix_errors_average[0, 0] /
        np.power(np.float64(observables_numbers[0]), power_1))
    plt.loglog(
        observables_numbers,
        np.power(np.float64(observables_numbers), power_2) *
        matrix_errors_average[0, 0] /
        np.power(np.float64(observables_numbers[0]), power_2))
    plt.xlabel(f'$N$')
    plt.ylabel(f'$\epsilon$')

    # plot legends
    observables_error_labels = [f'{name} error' for name in observables_names]
    CI_labels = [f'CI {name}' for name in observables_names]
    theoretical_labels = [f'Theoretical {name}' for name in observables_names]
    legend_labels = observables_error_labels + CI_labels + theoretical_labels + [
        f'$M^{{{power_1}}}$', f'$M^{{{power_2}}}$'
    ]

    plt.legend(legend_labels, loc='lower left')
    if title is not None:
        title = (' for ' + title
                 if title else '') + (f' with noise $\sigma = {sigma_noise}$'
                                      if sigma_noise != 0 else '')
        plt.title('Error in matrix norm vs number of observables' + title)
    print('The number of observables of each method is:')

    for i in range(len(observables_names)):
        print('Number of ', observables_names[i], ':', observables_numbers)
        print('Theoretical error is:', theoretical_errors)
    if path is not None:
        plt.savefig('gEDMDtests/Simulation_figures/Dictionary_Limit/' + path +
                    '.pdf',
                    bbox_inches='tight')
    plt.show(block=block)
    #Saves the plot


def plot_data(paths,
              observables_names,
              powers,
              xlabel='$M$',
              ylabel='$\epsilon$',
              font_size=12,
              font_size_ticks=10,
              block=False,
              colours=[
                  'blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink',
                  'gray', 'olive', 'cyan'
              ]):

    types_of_observables_number = len(observables_names)
    observables_error_labels = [f'{name} error' for name in observables_names]
    CI_labels = [f'CI {name}' for name in observables_names]
    power_labels = [f'$M^{{{power}}}$' for power in powers]
    legend_labels = observables_error_labels + CI_labels + power_labels

    # Create a separate figure for the legend
    fig_legend = plt.figure(figsize=(3, 3))
    ax_legend = fig_legend.add_subplot(111)
    # Create dummy lines for the legend
    lines = [
        ax_legend.plot([], [], color=colours[i % len(colours)], label=label)[0]
        for i, label in enumerate(legend_labels)
    ]
    fig_legend.legend(lines, legend_labels, loc='center', fontsize=font_size)
    ax_legend.axis('off')
    fig_legend.savefig('legend.pdf', bbox_inches='tight')

    for path in paths:
        #Extracts the data for each plot
        data = np.load('gEDMDtests/Simulation_data/Data_Limit/' + path +
                       '.npz')
        data_points_number = data['data_points_number']
        matrix_errors_average = data['matrix_errors_average']
        matrix_errors_confidence_interval = data[
            'matrix_errors_confidence_interval']
        lower_bound = matrix_errors_average - matrix_errors_confidence_interval
        upper_bound = matrix_errors_average + matrix_errors_confidence_interval

        # Plotting data
        plt.figure()
        plt.loglog(data_points_number, matrix_errors_average, marker='o')

        for i in range(lower_bound.shape[1]):
            plt.fill_between(data_points_number,
                             lower_bound[:, i],
                             upper_bound[:, i],
                             color=colours[i % len(colours)],
                             alpha=0.2)

        for power in powers:
            plt.loglog(
                data_points_number,
                np.power(np.float64(data_points_number), power) *
                matrix_errors_average[0, 0] /
                np.power(np.float64(data_points_number[0]), power))

        plt.xlabel(xlabel, fontsize=font_size)
        plt.ylabel(ylabel, fontsize=font_size)
        plt.tick_params(axis='both', which='major', labelsize=font_size_ticks)
        plt.tick_params(axis='both', which='minor', labelsize=font_size_ticks)
        plt.savefig('gEDMDtests/Simulation_figures/Data_Limit/' + path +
                    '.pdf',
                    bbox_inches='tight')
        plt.close(
        )  # Close the plot to prevent it from displaying in the notebook


#In the notation we use in our new paper:
# C= <A psi,psi>, where A is the generator of the system
# G = <psi,psi>, is the Gramm matrix
# A =G^{-1} C^T is the operator matrix
# In the notation in the old paper by Stefan et al:
# A = operator matrix = A
# C0 = gram matrix = G
# C1 = stiffness matrix = C
