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

# The code imports the following 3 classes from other files: 'MonteCarlo', 'Points' and 'Gaussian'.
# This allows for Monte Carlo calculations to be performed on the 'Points' and 'Gaussian' classes
# for specified number of dimensions and samples. Importing the classes rather than defining the
# classes within this file allows for future use of the classes without having to import the
# following code

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
no_of_samples = np.int32(100000000/no_of_ranks)

# The following variables A to D are of the class 'Points' for a differing number of dimensions.
# The first argument sets the dimensions of the randomly generated points, and the second sets the
# number of points generated.
if rank==0:
    print("Region of a unitary 2D ball:")

# Generating 2D (first arg.) points (number of which given by second arg.)
A = Points(2, no_of_samples)

# Running Monte Carlo calculations using the 2D points contained in class A. The function r_vector
# returns an array analysed by the Monte Carlo to estimate the integral between the limits defined
# by the third and fourth arguments.
A_MONTE = MonteCarlo(A, A.r_vector, -1, 1)
A_CALC = A_MONTE.parallelisation()

# Repeating calculations for 3D points
if rank==0:
    print("Region of a unitary 3D ball:")

B = Points(3, no_of_samples)
B_MONTE = MonteCarlo(B, B.r_vector, -1, 1)
B_CALC = B_MONTE.parallelisation()

# Repeating calculations for 4D points
if rank==0:
    print("Region of a unitary 4D ball:")

C = Points(4, no_of_samples)
C_MONTE = MonteCarlo(C, C.r_vector, -1, 1)
C_CALC = C_MONTE.parallelisation()

# Repeating calculations for 5D points
if rank==0:
    print("Region of a unitary 5D ball:")

D = Points(5, no_of_samples)
D_MONTE = MonteCarlo(D, D.r_vector, -1, 1)
D_CALC = D_MONTE.parallelisation()

# Generating the d-dimensional gaussian distributions in classes E to I. The first argument defines
# the distribution width, the second the offset, and third the number of sample points. Dimensions
# are taken from the length of the offset array.

# Generating a 1D gaussian with no offset and dist. width of 1
E = Gaussian(1, np.zeros(1), no_of_samples)
if rank==0:
    print(f"Gaussian Dist. in {E.d}D with offset r_o = {E.x_o} and dist. width σ = {E.sigma}:")

# Running Monte Carlo calculations on the function 'integrand' within the class E. Limits given by
# third and fourth arguments.
E_MONTE = MonteCarlo(E, E.integrand, -1, 1)
E_CALC = E_MONTE.parallelisation()

# Repeating for a 6D gaussian with no offset and dist. width of 1
F = Gaussian(1, np.zeros(6), no_of_samples)
if rank==0:
    print(f"Gaussian Dist. in {F.d}D with offset r_o = {F.x_o} and dist. width σ = {F.sigma}:")
F_MONTE = MonteCarlo(F, F.integrand, -1, 1)
F_CALC = F_MONTE.parallelisation()

# Repeating for a 1D gaussian with no offset and dist. width of 4
G = Gaussian(4, np.zeros(1), no_of_samples)
if rank==0:
    print(f"Gaussian Dist. in {G.d}D with offset r_o = {G.x_o} and dist. width σ = {G.sigma}:")
G_MONTE = MonteCarlo(G, G.integrand, -1, 1)
G_CALC = G_MONTE.parallelisation()

# Repeating for a 6D gaussian with offset 1 in all dimensions and dist. width of 4
H = Gaussian(4, np.ones(6), no_of_samples)
if rank==0:
    print(f"Gaussian Dist. in {H.d}D with offset r_o = {H.x_o} and dist. width σ = {H.sigma}:")
H_MONTE = MonteCarlo(H, H.integrand, -1, 1)
H_CALC = H_MONTE.parallelisation()

# Repeating for a 6D gaussian with various offsets and dist. width of 2
I = Gaussian(2, np.array( [2, 5, 8, 3, 11, 2] ), no_of_samples)
if rank==0:
    print(f"Gaussian Dist. in {I.d}D with offset r_o = {I.x_o} and dist. width σ = {I.sigma}:")
I_MONTE = MonteCarlo(I, I.integrand, -1, 1)
I_CALC = I_MONTE.parallelisation()

# Recording the end time of the code, and taking the difference from the start time to find how
# long the code took to run. This allows for comparison of runtimes for varying number of
# processors. Getting an estimate of the parallel efficiency of the code.
if rank==0:
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"The code took {execution_time} seconds to run for {no_of_ranks} processors")
    print()

# If running using 8 processors, it might be beneficial to comment out the MPI.Finalize() command
# below. For an unknown reason, the runtime increases significantly: using a sample size of
# no_of_samples = 100000000/no_of_ranks, the runtime jumps from ~21 seconds to ~80 seconds with an
# uncommented MPI.Finalize().
#MPI.Finalize()
