import numpy as np
import matplotlib.pyplot as plt
import sys
import os
# Get the directory of the current script
script_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)
import d3s.FEM_helper as FEM_helper

N_list = [5, 10, 100, 1000]
N = 10
A, B, C, h = FEM_helper.compute_A(N)

u = FEM_helper.compute_u_coefficients(N)

FEM_helper.u_plotter(N_list, "u.pdf")
# x = np.linspace(0, 1, N + 1)
# plt.plot(
#     x,
#     u,
#     label='u',
# )
# plt.show(block=True)
# plt.savefig("u.pdf", bbox_inches='tight')
#
