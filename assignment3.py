#!/bin/python3

"""
This code is suitably licensed:
https://github.com/eamonnmchugh/PH510-ACP/blob/Assignment-3/MIT%20Licence

This program makes use of Monte Carlo simulations to find the estimate for a given function. Monte 
Carlo simulations are used to simplify complex integrations through repeated random sampling. This 
is done by finding the average value of the function across a specified range. This expectation 
value is then multiplied by the range, which approximates the integral of the function. Finally, 
the function's variance can be found to obtain the uncertainty in the integral's estimate.

The two functions the Monte Carlo was used for were the classes 'Points' and 'Gaussian'. 'Points'
finds the region of a d-dimensional ball by generating 'n' random points (with unitary dimensions)
and finding their magnitudes, therefore determining whether each points lies 'within' the ball. The
ratio of points lying within the ball is used to estimate the ball's region. 'Gaussian' expresses 
the d-dimensional normal distribution with a given offset 'x_o' and dist. width 'sigma'.
"""

import time
import numpy as np
from mpi4py import MPI
from points_in_a_box import Points
from monte_carlo import MonteCarlo
from gaussian_distribution import Gaussian

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
comm = MPI.COMM_WORLD
no_of_ranks = comm.Get_size()
rank = comm.Get_rank()

if rank==0:
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(f"{no_of_ranks} Processors:")
    print()
    start_time = time.time()

no_of_samples = np.int32(100000000/no_of_ranks)

if rank==0:
    print("Region of a unitary 2D ball:")

A = Points(2, no_of_samples)
A_MONTE = MonteCarlo(A, A.r_vector, -1, 1)
A_CALC = A_MONTE.parallelisation()

if rank==0:
    print("Region of a unitary 3D ball:")

B = Points(3, no_of_samples)
B_MONTE = MonteCarlo(B, B.r_vector, -1, 1)
B_CALC = B_MONTE.parallelisation()

if rank==0:
    print("Region of a unitary 4D ball:")

C = Points(4, no_of_samples)
C_MONTE = MonteCarlo(C, C.r_vector, -1, 1)
C_CALC = C_MONTE.parallelisation()

if rank==0:
    print("Region of a unitary 5D ball:")

D = Points(5, no_of_samples)
D_MONTE = MonteCarlo(D, D.r_vector, -1, 1)
D_CALC = D_MONTE.parallelisation()

E = Gaussian(1, np.zeros(1), no_of_samples)
if rank==0:
    print(f"Gaussian Dist. in {E.d}D with offset r_o = {E.x_o} and dist. width σ = {E.sigma}:")
E_MONTE = MonteCarlo(E, E.integrand, -1, 1)
E_CALC = E_MONTE.parallelisation()

F = Gaussian(1, np.zeros(6), no_of_samples)
if rank==0:
    print(f"Gaussian Dist. in {F.d}D with offset r_o = {F.x_o} and dist. width σ = {F.sigma}:")
F_MONTE = MonteCarlo(F, F.integrand, -1, 1)
F_CALC = F_MONTE.parallelisation()

G = Gaussian(4, np.zeros(1), no_of_samples)
if rank==0:
    print(f"Gaussian Dist. in {G.d}D with offset r_o = {G.x_o} and dist. width σ = {G.sigma}:")
G_MONTE = MonteCarlo(G, G.integrand, -1, 1)
G_CALC = G_MONTE.parallelisation()

H = Gaussian(4, np.ones(6), no_of_samples)
if rank==0:
    print(f"Gaussian Dist. in {H.d}D with offset r_o = {H.x_o} and dist. width σ = {H.sigma}:")
H_MONTE = MonteCarlo(H, H.integrand, -1, 1)
H_CALC = H_MONTE.parallelisation()

I = Gaussian(2, np.array( [2, 5, 8, 3, 11, 2] ), no_of_samples)
if rank==0:
    print(f"Gaussian Dist. in {I.d}D with offset r_o = {I.x_o} and dist. width σ = {I.sigma}:")
I_MONTE = MonteCarlo(I, I.integrand, -1, 1)
I_CALC = I_MONTE.parallelisation()

if rank==0:
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"The code took {execution_time} seconds to run for {no_of_ranks} processors")
    print()
