#!/bin/python3

"""
This code is suitably licensed:
https://github.com/eamonnmchugh/PH510-ACP/blob/Assignment-4/MIT%20Licence

This program makes use of Monte Carlo simulations to find the estimate for a given function. Monte 
Carlo simulations are used to simplify complex integrations through repeated random sampling. This 
is done by finding the average value of the function across a specified range. This expectation 
value is then multiplied by the range, which approximates the integral of the function. Finally, 
the function's variance can be found to obtain the uncertainty in the integral's estimate.

In this case, the Monte Carlo is ran for the class 'PoissonSolver2D'. This class generates an NxN
grid of points whose potentials and charges can be manually set. after applying these charges and
potentials, and over-relaxing until the grid is in a state of equilibrium, random walkers are used
by freely moving throughout the grid starting at a point (i, j) until they reach a boundary
(x_b, y_b), at which point the potential is recorded. After repeated use of these walkers, a
probability map (Green's function) is generated, giving us an estimate of the potential at the
starting point.
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
