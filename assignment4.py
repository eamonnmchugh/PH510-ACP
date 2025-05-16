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

import time
import numpy as np
from mpi4py import MPI
from monte_carlo import MonteCarlo
from poisson_solver import PoissonSolver2D

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Initialising the MPI environment and drawing key information from it. The number of ranks allows
# us to know how many processors are being used, which allows for equal distribtuion of the
# workload between the processors.
comm = MPI.COMM_WORLD
no_of_ranks = comm.Get_size()
rank = comm.Get_rank()

# Recording the start time of the code
if rank==0:
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(f"{no_of_ranks} Processors:")
    print()
    start_time = time.time()



# Setting the number of samples to be inversely proportional to the number of ranks. Each rank runs
# the set number of samples, meaning the total number of samples used is equal for all number of
# ranks.
no_of_samples = np.int32(100000/no_of_ranks)

# Question 3
if rank==0:
    print("Exercise 3")
    print("Green's function evaluation for a square grid of side length 10cm:")
init_grid = PoissonSolver2D(0.10, 101, no_of_samples)

# (a)
A_MONTE = MonteCarlo(init_grid, init_grid.greens_function, -1, 1, 50, 50)
A_CALC = A_MONTE.parallelisation_array()
#init_grid.plot_value(A_CALC[1], "Green's Function At (5cm, 5cm)", 4)
#init_grid.plot_value(init_grid.site_visits, 'Number of Site Visits At (5cm, 5cm)', 4)

# (b)
B_MONTE = MonteCarlo(init_grid, init_grid.greens_function, -1, 1, 25, 25)
B_CALC = B_MONTE.parallelisation_array()
#init_grid.plot_value(B_CALC[1], "Green's Function At (2.5cm, 2.5cm)", 4)
#init_grid.plot_value(init_grid.site_visits, 'Number of Site Visits At (2.5cm, 2.5cm)', 4)

# (c)
C_MONTE = MonteCarlo(init_grid, init_grid.greens_function, -1, 1, 1, 25)
C_CALC = C_MONTE.parallelisation_array()
#init_grid.plot_value(C_CALC[1], "Green's Function At (0.1cm, 2.5cm)", 4)
#init_grid.plot_value(init_grid.site_visits, 'Number of Site Visits At (0.1cm, 2.5cm)', 4)

# (d)
D_MONTE = MonteCarlo(init_grid, init_grid.greens_function, -1, 1, 1, 1)
D_CALC = D_MONTE.parallelisation_array()
#init_grid.plot_value(D_CALC[1], "Green's Function At (0.1cm, 0.1cm)", 4)
#init_grid.plot_value(init_grid.site_visits, 'Number of Site Visits At (0.1cm, 0.1cm)', 4)
if rank==0:
    print(f"At centre point (5cm, 5cm) the error of the Green's function was:\n{A_CALC[2]}")
    print(f"At (2.5cm, 2.5cm) the error of the Green's function was:\n{B_CALC[2]}")
    print(f"At (0.1cm, 2.5cm) the error of the Green's function was:\n{C_CALC[2]}")
    print(f"At (0.1cm, 0.1cm) the error of the Green's function was:\n{D_CALC[2]}")
    print()

# Question 4
    print("Exercise 4")
    print("Potential calculation via Green's function for a square grid of side length 10cm:")

# (a)
    print("a) with boundary conditions: All edges uniformly at +1V")
init_grid = PoissonSolver2D(0.10, 21, no_of_samples)

phi_1 = init_grid.phi
phi_1 = init_grid.set_boundary_conditions('all_1V')
phi_1 = init_grid.overrelax()
POTENTIAL_50_50_4A = MonteCarlo(init_grid, init_grid.potential_via_greens, 0, 10, 10, 10)
POTENTIAL_50_50_4A_CALC = POTENTIAL_50_50_4A.parallelisation()
POTENTIAL_25_25_4A = MonteCarlo(init_grid, init_grid.potential_via_greens, 0, 10, 5, 5)
POTENTIAL_25_25_4A_CALC = POTENTIAL_25_25_4A.parallelisation()
POTENTIAL_1_25_4A = MonteCarlo(init_grid, init_grid.potential_via_greens, 0, 10, 1, 5)
POTENTIAL_1_25_4A_CALC = POTENTIAL_1_25_4A.parallelisation()
POTENTIAL_1_1_4A = MonteCarlo(init_grid, init_grid.potential_via_greens, 0, 10, 1, 1)
POTENTIAL_1_1_4A_CALC = POTENTIAL_1_1_4A.parallelisation()
if rank==0:
    print(f"At (5cm, 5cm): {POTENTIAL_50_50_4A_CALC[1]:.4f}V")
    print(f"At (2.5cm, 2.5cm): {POTENTIAL_25_25_4A_CALC[1]:.4f}V")
    print(f"At (0.1cm, 2.5cm): {POTENTIAL_1_25_4A_CALC[1]:.4f}V")
    print(f"At (0.1cm, 0.1cm): {POTENTIAL_1_1_4A_CALC[1]:.4f}V")
    print()

# (b)
    print("b) with boundary conditions: Top and bottom edges: +1V, left and right edges: -1V")
init_grid = PoissonSolver2D(0.10, 21, no_of_samples)

phi_2 = init_grid.phi
phi_2 = init_grid.set_boundary_conditions('tb1_lr-1')
phi_2 = init_grid.overrelax()
POTENTIAL_50_50_4B = MonteCarlo(init_grid, init_grid.potential_via_greens, 0, 10, 10, 10)
POTENTIAL_50_50_4B_CALC = POTENTIAL_50_50_4B.parallelisation()
POTENTIAL_25_25_4B = MonteCarlo(init_grid, init_grid.potential_via_greens, 0, 10, 5, 5)
POTENTIAL_25_25_4B_CALC = POTENTIAL_25_25_4B.parallelisation()
POTENTIAL_1_25_4B = MonteCarlo(init_grid, init_grid.potential_via_greens, 0, 10, 1, 5)
POTENTIAL_1_25_4B_CALC = POTENTIAL_1_25_4B.parallelisation()
POTENTIAL_1_1_4B = MonteCarlo(init_grid, init_grid.potential_via_greens, 0, 10, 1, 1)
POTENTIAL_1_1_4B_CALC = POTENTIAL_1_1_4B.parallelisation()
if rank==0:
    print(f"At (5cm, 5cm): {POTENTIAL_50_50_4B_CALC[1]:.4f}V")
    print(f"At (2.5cm, 2.5cm): {POTENTIAL_25_25_4B_CALC[1]:.4f}V")
    print(f"At (0.1cm, 2.5cm): {POTENTIAL_1_25_4B_CALC[1]:.4f}V")
    print(f"At (0.1cm, 0.1cm): {POTENTIAL_1_1_4B_CALC[1]:.4f}V")
    print()

# (c)
    print("c) with boundary conditions: Top and left edges: +2V, bottom edge: 0V, right edge: -4V")
init_grid = PoissonSolver2D(0.10, 21, no_of_samples)

phi_3 = init_grid.phi
phi_3 = init_grid.set_boundary_conditions('tl2_b0_r-4')
phi_3 = init_grid.overrelax()
POTENTIAL_50_50_4C = MonteCarlo(init_grid, init_grid.potential_via_greens, 0, 10, 10, 10)
POTENTIAL_50_50_4C_CALC = POTENTIAL_50_50_4C.parallelisation()
POTENTIAL_25_25_4C = MonteCarlo(init_grid, init_grid.potential_via_greens, 0, 10, 5, 5)
POTENTIAL_25_25_4C_CALC = POTENTIAL_25_25_4C.parallelisation()
POTENTIAL_1_25_4C = MonteCarlo(init_grid, init_grid.potential_via_greens, 0, 10, 1, 5)
POTENTIAL_1_25_4C_CALC = POTENTIAL_1_25_4C.parallelisation()
POTENTIAL_1_1_4C = MonteCarlo(init_grid, init_grid.potential_via_greens, 0, 10, 1, 1)
POTENTIAL_1_1_4C_CALC = POTENTIAL_1_1_4C.parallelisation()
if rank==0:
    print(f"At (5cm, 5cm): {POTENTIAL_50_50_4C_CALC[1]:.4f}V")
    print(f"At (2.5cm, 2.5cm): {POTENTIAL_25_25_4C_CALC[1]:.4f}V")
    print(f"At (0.1cm, 2.5cm): {POTENTIAL_1_25_4C_CALC[1]:.4f}V")
    print(f"At (0.1cm, 0.1cm): {POTENTIAL_1_1_4C_CALC[1]:.4f}V")
    print()


# (d)
    print("d) Repeat, with uniform charge of 10C throughout the grid")

# (i)
    print("i) with BC: All edges uniformly at +1V")
init_grid = PoissonSolver2D(0.10, 21, no_of_samples)

phi_4 = init_grid.phi
phi_4 = init_grid.set_boundary_conditions('all_1V')
phi_4 = init_grid.overrelax()
f_4 = init_grid.apply_charge_distribution('uniform_10C')
POTENTIAL_50_50_4DI = MonteCarlo(init_grid, init_grid.potential_via_greens, 0, 10, 10, 10)
POTENTIAL_50_50_4DI_CALC = POTENTIAL_50_50_4DI.parallelisation()
POTENTIAL_25_25_4DI = MonteCarlo(init_grid, init_grid.potential_via_greens, 0, 10, 5, 5)
POTENTIAL_25_25_4DI_CALC = POTENTIAL_25_25_4DI.parallelisation()
POTENTIAL_1_25_4DI = MonteCarlo(init_grid, init_grid.potential_via_greens, 0, 10, 1, 5)
POTENTIAL_1_25_4DI_CALC = POTENTIAL_1_25_4DI.parallelisation()
POTENTIAL_1_1_4DI = MonteCarlo(init_grid, init_grid.potential_via_greens, 0, 10, 1, 1)
POTENTIAL_1_1_4DI_CALC = POTENTIAL_1_1_4DI.parallelisation()
if rank==0:
    print(f"At (5cm, 5cm): {POTENTIAL_50_50_4DI_CALC[1]:.4f}V")
    print(f"At (2.5cm, 2.5cm): {POTENTIAL_25_25_4DI_CALC[1]:.4f}V")
    print(f"At (0.1cm, 2.5cm): {POTENTIAL_1_25_4DI_CALC[1]:.4f}V")
    print(f"At (0.1cm, 0.1cm): {POTENTIAL_1_1_4DI_CALC[1]:.4f}V")
    print()

# (ii)
    print("ii) with BC: Top and bottom edges: +1V, left and right edges: -1V")
init_grid = PoissonSolver2D(0.10, 21, no_of_samples)

phi_5 = init_grid.phi
phi_5 = init_grid.set_boundary_conditions('tb1_lr-1')
phi_5 = init_grid.overrelax()
f_5 = init_grid.apply_charge_distribution('uniform_10C')
POTENTIAL_50_50_4DII = MonteCarlo(init_grid, init_grid.potential_via_greens, 0, 10, 10, 10)
POTENTIAL_50_50_4DII_CALC = POTENTIAL_50_50_4DII.parallelisation()
POTENTIAL_25_25_4DII = MonteCarlo(init_grid, init_grid.potential_via_greens, 0, 10, 5, 5)
POTENTIAL_25_25_4DII_CALC = POTENTIAL_25_25_4DII.parallelisation()
POTENTIAL_1_25_4DII = MonteCarlo(init_grid, init_grid.potential_via_greens, 0, 10, 1, 5)
POTENTIAL_1_25_4DII_CALC = POTENTIAL_1_25_4DII.parallelisation()
POTENTIAL_1_1_4DII = MonteCarlo(init_grid, init_grid.potential_via_greens, 0, 10, 1, 1)
POTENTIAL_1_1_4DII_CALC = POTENTIAL_1_1_4DII.parallelisation()
if rank==0:
    print(f"At (5cm, 5cm): {POTENTIAL_50_50_4DII_CALC[1]:.4f}V")
    print(f"At (2.5cm, 2.5cm): {POTENTIAL_25_25_4DII_CALC[1]:.4f}V")
    print(f"At (0.1cm, 2.5cm): {POTENTIAL_1_25_4DII_CALC[1]:.4f}V")
    print(f"At (0.1cm, 0.1cm): {POTENTIAL_1_1_4DII_CALC[1]:.4f}V")
    print()

# (iii)
    print("iii) with BC: Top and left edges: +2V, bottom edge: 0V, right edge: -4V")
init_grid = PoissonSolver2D(0.10, 21, no_of_samples)

phi_6 = init_grid.phi
phi_6 = init_grid.set_boundary_conditions('tl2_b0_r-4')
phi_6 = init_grid.overrelax()
f_6 = init_grid.apply_charge_distribution('uniform_10C')
POTENTIAL_50_50_4DIII = MonteCarlo(init_grid, init_grid.potential_via_greens, 0, 10, 10, 10)
POTENTIAL_50_50_4DIII_CALC = POTENTIAL_50_50_4DIII.parallelisation()
POTENTIAL_25_25_4DIII = MonteCarlo(init_grid, init_grid.potential_via_greens, 0, 10, 5, 5)
POTENTIAL_25_25_4DIII_CALC = POTENTIAL_25_25_4DIII.parallelisation()
POTENTIAL_1_25_4DIII = MonteCarlo(init_grid, init_grid.potential_via_greens, 0, 10, 1, 5)
POTENTIAL_1_25_4DIII_CALC = POTENTIAL_1_25_4DIII.parallelisation()
POTENTIAL_1_1_4DIII = MonteCarlo(init_grid, init_grid.potential_via_greens, 0, 10, 1, 1)
POTENTIAL_1_1_4DIII_CALC = POTENTIAL_1_1_4DIII.parallelisation()
if rank==0:
    print(f"At (5cm, 5cm): {POTENTIAL_50_50_4DIII_CALC[1]:.4f}V")
    print(f"At (2.5cm, 2.5cm): {POTENTIAL_25_25_4DIII_CALC[1]:.4f}V")
    print(f"At (0.1cm, 2.5cm): {POTENTIAL_1_25_4DIII_CALC[1]:.4f}V")
    print(f"At (0.1cm, 0.1cm): {POTENTIAL_1_1_4DIII_CALC[1]:.4f}V")
    print()

# (e)
    print("e) Repeat, with uniform charge gradient from 1C at top to 0C at bottom")

# (i)
    print("i) with BC: All edges uniformly at +1V")
init_grid = PoissonSolver2D(0.10, 21, no_of_samples)

phi_7 = init_grid.phi
phi_7 = init_grid.set_boundary_conditions('all_1V')
phi_7 = init_grid.overrelax()
f_7 = init_grid.apply_charge_distribution('linear_gradient_top_to_bottom')
POTENTIAL_50_50_4EI = MonteCarlo(init_grid, init_grid.potential_via_greens, 0, 10, 10, 10)
POTENTIAL_50_50_4EI_CALC = POTENTIAL_50_50_4EI.parallelisation()
POTENTIAL_25_25_4EI = MonteCarlo(init_grid, init_grid.potential_via_greens, 0, 10, 5, 5)
POTENTIAL_25_25_4EI_CALC = POTENTIAL_25_25_4EI.parallelisation()
POTENTIAL_1_25_4EI = MonteCarlo(init_grid, init_grid.potential_via_greens, 0, 10, 1, 5)
POTENTIAL_1_25_4EI_CALC = POTENTIAL_1_25_4EI.parallelisation()
POTENTIAL_1_1_4EI = MonteCarlo(init_grid, init_grid.potential_via_greens, 0, 10, 1, 1)
POTENTIAL_1_1_4EI_CALC = POTENTIAL_1_1_4EI.parallelisation()
if rank==0:
    print(f"At (5cm, 5cm): {POTENTIAL_50_50_4EI_CALC[1]:.4f}V")
    print(f"At (2.5cm, 2.5cm): {POTENTIAL_25_25_4EI_CALC[1]:.4f}V")
    print(f"At (0.1cm, 2.5cm): {POTENTIAL_1_25_4EI_CALC[1]:.4f}V")
    print(f"At (0.1cm, 0.1cm): {POTENTIAL_1_1_4EI_CALC[1]:.4f}V")
    print()

# (ii)
    print("ii) with BC: Top and bottom edges: +1V, left and right edges: -1V")
init_grid = PoissonSolver2D(0.10, 21, no_of_samples)

phi_8 = init_grid.phi
phi_8 = init_grid.set_boundary_conditions('tb1_lr-1')
phi_8 = init_grid.overrelax()
f_8 = init_grid.apply_charge_distribution('linear_gradient_top_to_bottom')
POTENTIAL_50_50_4EII = MonteCarlo(init_grid, init_grid.potential_via_greens, 0, 10, 10, 10)
POTENTIAL_50_50_4EII_CALC = POTENTIAL_50_50_4EII.parallelisation()
POTENTIAL_25_25_4EII = MonteCarlo(init_grid, init_grid.potential_via_greens, 0, 10, 5, 5)
POTENTIAL_25_25_4EII_CALC = POTENTIAL_25_25_4EII.parallelisation()
POTENTIAL_1_25_4EII = MonteCarlo(init_grid, init_grid.potential_via_greens, 0, 10, 1, 5)
POTENTIAL_1_25_4EII_CALC = POTENTIAL_1_25_4EII.parallelisation()
POTENTIAL_1_1_4EII = MonteCarlo(init_grid, init_grid.potential_via_greens, 0, 10, 1, 1)
POTENTIAL_1_1_4EII_CALC = POTENTIAL_1_1_4EII.parallelisation()
if rank==0:
    print(f"At (5cm, 5cm): {POTENTIAL_50_50_4EII_CALC[1]:.4f}V")
    print(f"At (2.5cm, 2.5cm): {POTENTIAL_25_25_4EII_CALC[1]:.4f}V")
    print(f"At (0.1cm, 2.5cm): {POTENTIAL_1_25_4EII_CALC[1]:.4f}V")
    print(f"At (0.1cm, 0.1cm): {POTENTIAL_1_1_4EII_CALC[1]:.4f}V")
    print()

# (iii)
    print("iii) with BC: Top and left edges: +2V, bottom edge: 0V, right edge: -4V")
init_grid = PoissonSolver2D(0.10, 21, no_of_samples)

phi_9 = init_grid.phi
phi_9 = init_grid.set_boundary_conditions('tl2_b0_r-4')
phi_9 = init_grid.overrelax()
f_9 = init_grid.apply_charge_distribution('linear_gradient_top_to_bottom')
POTENTIAL_50_50_4EIII = MonteCarlo(init_grid, init_grid.potential_via_greens, 0, 10, 10, 10)
POTENTIAL_50_50_4EIII_CALC = POTENTIAL_50_50_4EIII.parallelisation()
POTENTIAL_25_25_4EIII = MonteCarlo(init_grid, init_grid.potential_via_greens, 0, 10, 5, 5)
POTENTIAL_25_25_4EIII_CALC = POTENTIAL_25_25_4EIII.parallelisation()
POTENTIAL_1_25_4EIII = MonteCarlo(init_grid, init_grid.potential_via_greens, 0, 10, 1, 5)
POTENTIAL_1_25_4EIII_CALC = POTENTIAL_1_25_4EIII.parallelisation()
POTENTIAL_1_1_4EIII = MonteCarlo(init_grid, init_grid.potential_via_greens, 0, 10, 1, 1)
POTENTIAL_1_1_4EIII_CALC = POTENTIAL_1_1_4EIII.parallelisation()
if rank==0:
    print(f"At (5cm, 5cm): {POTENTIAL_50_50_4EIII_CALC[1]:.4f}V")
    print(f"At (2.5cm, 2.5cm): {POTENTIAL_25_25_4EIII_CALC[1]:.4f}V")
    print(f"At (0.1cm, 2.5cm): {POTENTIAL_1_25_4EIII_CALC[1]:.4f}V")
    print(f"At (0.1cm, 0.1cm): {POTENTIAL_1_1_4EIII_CALC[1]:.4f}V")
    print()

# (f)
    print("f) Repeat, with exponentially decaying charge exp(-2000|r|) placed at centre of grid")

# (i)
    print("i) with BC: All edges uniformly at +1V")
init_grid = PoissonSolver2D(0.10, 21, no_of_samples)

phi_10 = init_grid.phi
phi_10 = init_grid.set_boundary_conditions('all_1V')
phi_10 = init_grid.overrelax()
f_10 = init_grid.apply_charge_distribution('exp_decay')
POTENTIAL_50_50_4FI = MonteCarlo(init_grid, init_grid.potential_via_greens, 0, 10, 10, 10)
POTENTIAL_50_50_4FI_CALC = POTENTIAL_50_50_4FI.parallelisation()
POTENTIAL_25_25_4FI = MonteCarlo(init_grid, init_grid.potential_via_greens, 0, 10, 5, 5)
POTENTIAL_25_25_4FI_CALC = POTENTIAL_25_25_4FI.parallelisation()
POTENTIAL_1_25_4FI = MonteCarlo(init_grid, init_grid.potential_via_greens, 0, 10, 1, 5)
POTENTIAL_1_25_4FI_CALC = POTENTIAL_1_25_4FI.parallelisation()
POTENTIAL_1_1_4FI = MonteCarlo(init_grid, init_grid.potential_via_greens, 0, 10, 1, 1)
POTENTIAL_1_1_4FI_CALC = POTENTIAL_1_1_4FI.parallelisation()
if rank==0:
    print(f"At (5cm, 5cm): {POTENTIAL_50_50_4FI_CALC[1]:.4f}V")
    print(f"At (2.5cm, 2.5cm): {POTENTIAL_25_25_4FI_CALC[1]:.4f}V")
    print(f"At (0.1cm, 2.5cm): {POTENTIAL_1_25_4FI_CALC[1]:.4f}V")
    print(f"At (0.1cm, 0.1cm): {POTENTIAL_1_1_4FI_CALC[1]:.4f}V")
    print()

# (ii)
    print("ii) with BC: Top and bottom edges: +1V, left and right edges: -1V")
init_grid = PoissonSolver2D(0.10, 21, no_of_samples)

phi_11 = init_grid.phi
phi_11 = init_grid.set_boundary_conditions('tb1_lr-1')
phi_11 = init_grid.overrelax()
f_11 = init_grid.apply_charge_distribution('exp_decay')
POTENTIAL_50_50_4FII = MonteCarlo(init_grid, init_grid.potential_via_greens, 0, 10, 10, 10)
POTENTIAL_50_50_4FII_CALC = POTENTIAL_50_50_4FII.parallelisation()
POTENTIAL_25_25_4FII = MonteCarlo(init_grid, init_grid.potential_via_greens, 0, 10, 5, 5)
POTENTIAL_25_25_4FII_CALC = POTENTIAL_25_25_4FII.parallelisation()
POTENTIAL_1_25_4FII = MonteCarlo(init_grid, init_grid.potential_via_greens, 0, 10, 1, 5)
POTENTIAL_1_25_4FII_CALC = POTENTIAL_1_25_4FII.parallelisation()
POTENTIAL_1_1_4FII = MonteCarlo(init_grid, init_grid.potential_via_greens, 0, 10, 1, 1)
POTENTIAL_1_1_4FII_CALC = POTENTIAL_1_1_4FII.parallelisation()
if rank==0:
    print(f"At (5cm, 5cm): {POTENTIAL_50_50_4FII_CALC[1]:.4f}V")
    print(f"At (2.5cm, 2.5cm): {POTENTIAL_25_25_4FII_CALC[1]:.4f}V")
    print(f"At (0.1cm, 2.5cm): {POTENTIAL_1_25_4FII_CALC[1]:.4f}V")
    print(f"At (0.1cm, 0.1cm): {POTENTIAL_1_1_4FII_CALC[1]:.4f}V")
    print()

# (iii)
    print("iii) with BC: Top and left edges: +2V, bottom edge: 0V, right edge: -4V")
init_grid = PoissonSolver2D(0.10, 21, no_of_samples)

phi_12 = init_grid.phi
phi_12 = init_grid.set_boundary_conditions('tl2_b0_r-4')
phi_12 = init_grid.overrelax()
f_12 = init_grid.apply_charge_distribution('exp_decay')
POTENTIAL_50_50_4FIII = MonteCarlo(init_grid, init_grid.potential_via_greens, 0, 10, 10, 10)
POTENTIAL_50_50_4FIII_CALC = POTENTIAL_50_50_4FIII.parallelisation()
POTENTIAL_25_25_4FIII = MonteCarlo(init_grid, init_grid.potential_via_greens, 0, 10, 5, 5)
POTENTIAL_25_25_4FIII_CALC = POTENTIAL_25_25_4FIII.parallelisation()
POTENTIAL_1_25_4FIII = MonteCarlo(init_grid, init_grid.potential_via_greens, 0, 10, 1, 5)
POTENTIAL_1_25_4FIII_CALC = POTENTIAL_1_25_4FIII.parallelisation()
POTENTIAL_1_1_4FIII = MonteCarlo(init_grid, init_grid.potential_via_greens, 0, 10, 1, 1)
POTENTIAL_1_1_4FIII_CALC = POTENTIAL_1_1_4FIII.parallelisation()
if rank==0:
    print(f"At (5cm, 5cm): {POTENTIAL_50_50_4FIII_CALC[1]:.4f}V")
    print(f"At (2.5cm, 2.5cm): {POTENTIAL_25_25_4FIII_CALC[1]:.4f}V")
    print(f"At (0.1cm, 2.5cm): {POTENTIAL_1_25_4FIII_CALC[1]:.4f}V")
    print(f"At (0.1cm, 0.1cm): {POTENTIAL_1_1_4FIII_CALC[1]:.4f}V")
    print()

# Question 5
    print("Exercise 5")
    print("Potential after over-relaxation of grid with side length 10cm:")
    print("i) with BC: All edges uniformly at +1V")
    print(f"At (5cm, 5cm): {phi_1[10, 10]:.4f}V")
    print(f"At (2.5cm, 2.5cm): {phi_1[5, 5]:.4f}V")
    print(f"At (0.1cm, 2.5cm): {phi_1[1, 5]:.4f}V")
    print(f"At (0.1cm, 0.1cm): {phi_1[1, 1]:.4f}V")
    print()
    print("ii) with BC: Top and bottom edges: +1V, left and right edges: -1V")
    print(f"At (5cm, 5cm): {phi_2[10, 10]:.4f}V")
    print(f"At (2.5cm, 2.5cm): {phi_2[5, 5]:.4f}V")
    print(f"At (0.1cm, 2.5cm): {phi_2[1, 5]:.4f}V")
    print(f"At (0.1cm, 0.1cm): {phi_2[1, 1]:.4f}V")
    print()
    print("iii) with BC: Top and left edges: +2V, bottom edge: 0V, right edge: -4V")
    print(f"At (5cm, 5cm): {phi_3[10, 10]:.4f}V")
    print(f"At (2.5cm, 2.5cm): {phi_3[5, 5]:.4f}V")
    print(f"At (0.1cm, 2.5cm): {phi_3[1, 5]:.4f}V")
    print(f"At (0.1cm, 0.1cm): {phi_3[1, 1]:.4f}V")
    print()


# Recording the end time of the code, and taking the difference from the start time to find how
# long the code took to run. This allows for comparison of runtimes for varying number of
# processors. Getting an estimate of the parallel efficiency of the code.
if rank==0:
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"The code took {execution_time} seconds to run for {no_of_ranks} processor")

# If running using 8 processors, it might be beneficial to comment out the MPI.Finalize() command
# below. For an unknown reason, the runtime increases significantly: using a sample size of
# no_of_samples = 100000000/no_of_ranks, the runtime jumps from ~21 seconds to ~80 seconds with an
# uncommented MPI.Finalize().
#MPI.Finalize()
