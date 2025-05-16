#!/bin/python3

"""
This code is suitably licensed:
https://github.com/eamonnmchugh/PH510-ACP/blob/Assignment-4/MIT%20Licence

Smaller version of the file 'assignment4.py' used to plot the Green's functions and number of site
visits in Task 3. This could not be done in the main file as plt.show() could not be used when
submitting a job.
"""


import numpy as np
from poisson_solver import PoissonSolver2D

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Setting the number of samples to be inversely proportional to the number of ranks. Each rank runs
# the set number of samples, meaning the total number of samples used is equal for all number of
# ranks.
no_of_samples = np.int32(100000)

# Question 3
# if rank==0:
print("Exercise 3")
print("Green's function evaluation for a square grid of side length 10cm:")
init_grid = PoissonSolver2D(0.10, 101, no_of_samples)

# (a)
A_MONTE = init_grid.greens_function(50, 50)
#A_CALC = A_MONTE.parallelisation_array()
init_grid.plot_value(A_MONTE, "Green's Function At (5cm, 5cm)", 10, "Probability")
init_grid.plot_value(init_grid.site_visits, 'Number of Site Visits At (5cm, 5cm)', 10,
'Number of Site Visits')

# (b)
B_MONTE = init_grid.greens_function(25, 25)
init_grid.plot_value(B_MONTE, "Green's Function At (2.5cm, 2.5cm)", 10, "Probability")
init_grid.plot_value(init_grid.site_visits, 'Number of Site Visits At (2.5cm, 2.5cm)', 10,
'Number of Site Visits')

# (c)
C_MONTE = init_grid.greens_function(1, 25)
init_grid.plot_value(C_MONTE, "Green's Function At (0.1cm, 2.5cm)", 10, "Probability")
init_grid.plot_value(init_grid.site_visits, 'Number of Site Visits At (0.1cm, 2.5cm)', 10,
'Number of Site Visits')

# (d)
D_MONTE = init_grid.greens_function(1, 1)
init_grid.plot_value(D_MONTE, "Green's Function At (0.1cm, 0.1cm)", 10, "Probability")
init_grid.plot_value(init_grid.site_visits, 'Number of Site Visits At (0.1cm, 0.1cm)', 10,
'Number of Site Visits')
print(f"At centre point (5cm, 5cm):\n{A_MONTE}")
print(f"At (2.5cm, 2.5cm):\n{B_MONTE}")
print(f"At (0.1cm, 2.5cm):\n{C_MONTE}")
print(f"At (0.1cm, 0.1cm):\n{D_MONTE}")
print()
