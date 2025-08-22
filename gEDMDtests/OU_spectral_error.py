#In this file we show plots of the rate of convergence
#of GEDMD for different number of observables
#We compare the rate of convergence for monomials and gaussians

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import scipy.stats as stats
import scipy as sp
# Get the directory of the current script
script_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

import d3s.algorithms as algorithms
import d3s.domain as domain
import d3s.observables as observables
import d3s.gEDMD_tests_helper_functions as gedmd_helper
# import d3s.systems as systems

# Constants
M = 10**6
number_of_runs = 4
number_of_batches = 2
confidence_level = 0.95
degree_of_monomials = 8
observables_names = ['Monomials', 'Gaussians', 'FEM']
min_number_of_data_points = 1000

#Seed for reproducibility
np.random.seed(2010)

# # ########################################
# # # OU system
# # ########################################
paths = [
    'OU_Spectral',
    'OU_PF_Spectral', 'OU_EDMD_Spectral'
]
# define domain
bounds = np.array([[-2, 2]])
boxes = np.array([degree_of_monomials + 1])
Omega = domain.discretization(bounds, boxes)

# define system
alpha = 1
beta = 4

def b(x):
    return -alpha * x

def sigma(x):
    return np.sqrt(2 / beta) * np.ones((1, 1, x.shape[1]))

# define observables
psi_m = observables.monomials(degree_of_monomials)
variance = (bounds[0, 1] - bounds[0, 0]) / boxes[0] / 2
psi_g = observables.gaussians(Omega, sigma=variance)
psi_FEM = observables.FEM_1d(bounds[0, 0], bounds[0, 1], boxes[0])
observables_list = [psi_m, psi_g, psi_FEM]
observables_names = ['Monomials', 'Gaussians', 'FEM']
#gEDMD Koopman operator

gedmd_helper.plot_spectrum_errors_data_limit(
    M,
    min_number_of_data_points,
    confidence_level,
    number_of_runs,
    number_of_batches,
    observables_list,
    observables_names,
    Omega,
    b,
    sigma=sigma,
    operator='K',
    path=paths[0],
    eigen_values_exact=None,
)

# gEDMD Perron-Frobenius operator. Monomials are stable so error is 0
gedmd_helper.plot_spectrum_errors_data_limit(M,
                                    min_number_of_data_points,
                                    confidence_level,
                                    number_of_runs,
                                    number_of_batches,
                                    observables_list,
                                    observables_names,
                                    Omega,
                                    b,
                                    sigma=sigma,
                                    operator='P',
                                    path=paths[1],
                                    eigen_values_exact=None)

#EDMD
T = 0.5
theta =1 
sigma = np.sqrt(1/2)
f = gedmd_helper.OU_solution_f(theta,sigma,T)
gedmd_helper.plot_spectrum_errors_data_limit(M,
                                    min_number_of_data_points,
                                    confidence_level,
                                    number_of_runs,
                                    number_of_batches,
                                    observables_list,
                                    observables_names,
                                    Omega,
                                    b,
                                    sigma=sigma,
                                    f=f,
                                    path=paths[2],
                                    eigen_values_exact=None)


powers = [-0.5, -1]
observables_names = ['Monomials', 'Gaussians', 'FEM']
font_size = 30
font_size_ticks = 30
xlabel='$M$'
ylabel='$\epsilon$'
colours_observed=[
    'blue', 'orange', 'green', 'red', 'purple', 'brown',
    'pink', 'gray', 'olive', 'cyan'
]
colours_slopes=[
    'red', 'purple', 'cyan', 'brown', 'pink', 'gray',
    'olive', 'cyan'
]
# gedmd_helper.plot_data_limit(paths_data,
#                              observables_names_data,
#                              powers,
#                              xlabel='$M$',
#                              ylabel='$\epsilon$',
#                              font_size=font_size,
#                              font_size_ticks=font_size_ticks)

  # Create legend labels
observables_error_labels = [f'{name} error' for name in observables_names]
CI_labels = [f'CI {name}' for name in observables_names]
power_labels = [f'$M^{{{power}}}$' for power in powers]
legend_labels = observables_error_labels + CI_labels + power_labels

for path in paths:
    #Extracts the data for each plot
    data = np.load('gEDMDtests/Simulation_data/Data_Limit_Spectrum/' + path +
                    '.npz')
    data_points_number = data['data_points_number']
    spectrum_errors_average = data['spectrum_errors_average']
    spectrum_errors_confidence_interval = data[
        'spectrum_errors_confidence_interval']
    lower_bound = spectrum_errors_average - spectrum_errors_confidence_interval
    upper_bound = spectrum_errors_average + spectrum_errors_confidence_interval

    # Plotting data
    plt.figure()
    plt.loglog(data_points_number, spectrum_errors_average, marker='o')

    for i in range(lower_bound.shape[1]):
        plt.fill_between(data_points_number,
                            lower_bound[:, i],
                            upper_bound[:, i],
                            color=colours_observed[i % len(colours_observed)],
                            alpha=0.2,
                            label=CI_labels[i])

    for i, power in enumerate(powers):
        plt.loglog(data_points_number,
                    np.power(np.float64(data_points_number), power) *
                    spectrum_errors_average[0, 0] /
                    np.power(np.float64(data_points_number[0]), power),
                    label=power_labels[i],
                    color=colours_slopes[i % len(colours_slopes)])

    plt.xlabel(xlabel, fontsize=font_size)
    plt.ylabel(ylabel, fontsize=font_size)
    plt.tick_params(axis='both', which='major', labelsize=font_size_ticks)
    plt.tick_params(axis='both', which='minor', labelsize=font_size_ticks)
    #Gets the legend for first plot to export and then removes it
    if path == paths[-1]:
        legend = plt.legend(
            legend_labels,  # we place the legend on bottom left of the plot
            loc='lower left',
            fontsize=font_size,
            ncol=2)
        gedmd_helper.export_legend(
            legend,
            filename='gEDMDtests/Simulation_figures/Data_Limit_Spectrum/legend.pdf')
        plt.legend().remove()
    plt.savefig('gEDMDtests/Simulation_figures/Data_Limit_Spectrum/' + path +
                '.pdf',
                bbox_inches='tight')
    plt.close(
    )  # Close the plot to prevent it from displaying in the notebook
