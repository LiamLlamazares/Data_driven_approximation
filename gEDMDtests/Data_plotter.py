import numpy as np
import matplotlib.pyplot as plt
import sys
import sys
import os
import scipy.stats as stats
# Get the directory of the current script
script_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

import d3s.algorithms as algorithms
import d3s.domain as domain
import d3s.observables as observables
import d3s.gEDMD_tests_helper_functions as gedmd_helper
import d3s.systems as systems

paths = ['ODE', 'Double_well', 'Double_well_EDMD', 'OU', 'OU_PF', 'OU_EDMD']
powers = [-0.5, -1]
observables_names = ['Monomials', 'Gaussians']
gedmd_helper.plot_data(paths,
                       observables_names,
                       powers,
                       xlabel='$M$',
                       ylabel='$\epsilon$',
                       font_size=15,
                       font_size_ticks=12,
                       colours=[
                           'blue', 'orange', 'green', 'red', 'purple', 'brown',
                           'pink', 'gray', 'olive', 'cyan'
                       ])
