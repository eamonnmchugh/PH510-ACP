#!/bin/python3

"""
This program makes use Monte Carlo simulations to find the estimate for a given function. This is 
done by finding the average value of the function across a specified range. This expectation value
is then multiplied by the range, which estimates a value for the integration of the function.
Finally, the function's variance can be found to obtain the error.

The two functions the Monte Carlo was used for were the classes 'Points' and 'Gaussian'. 'Points'
finds the region of a d-dimensional ball by finding the location of 'n' random points (with unitary
dimensions) and calculating the fraction of points found 'within' the ball. 'Gaussian' expresses 
the d-dimensional normal distribution with a given offset 'x_o' and dist. width 'sigma'.
"""

import math
import time
import numpy as np
from mpi4py import MPI

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class Points:
    """
    Generates 'n' random points within a box of 'd' dimensions with constituent coord values
    between the range of -1 and 1
    """
    def __init__(self, d, n):
        self.n = n
        self.d = d
        points = np.random.uniform(-1, 1, self.d*self.n)
        self.points = points.reshape((self.n, self.d))

    def __str__(self):
        """
        Returns the points
        """
        return f"Points:({self.points})"

    def r_vector(self):
        """
        Calculates the square of the magnitude of the d-dimensional vectors by finding the sum of
        the square of the points' coords. Then finds out how many of the random points fall within
        the dimensions of the ball (hit). The points hit if their magnitudes (or more accurately
        the square of their magnitudes) are one or less.
        """
        dim_sum = np.sum(self.points**2, axis=1)
        within = (dim_sum <= 1).astype(int)
        return within

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class MonteCarlo:
    """
    Approximates a given function by finding the expected value (average), then integrates this
    average across the limits of the function 'a' and 'b'. Finally, the function's variance (error)
    is found.
    """
    def __init__(self, class_used, function, a, b):
        self.a = a
        self.b = b
        self.class_used = class_used
        self.function = function
        self.value = function()

    def __str__(self):
        """
        Confirms which function the Monte Carlo simulation is approximating
        """
        return f"Monte Carlo simulation of the function {self.function}"

    def average(self):
        """
        Calculates the average value of a given function, <f> = 1/n * Σn (f(n)).
        """
        value_2 = np.square(self.value)
        average = 1/self.class_used.n * np.sum(self.value)
        average_2 = 1/self.class_used.n * np.sum(value_2)
        return average, average_2

    def parallelisation(self):
        """
        Sums up the averages (and square of the averages) of the given function using the MPI 
        "reduce" command, bringing them into the first processor "rank 0". Using these the
        integral of the given function between limits 'a' and 'b', I = (b - a)<f> were found, as
        well as the variance (error), σ^2 = 1/n * (<f^2> - <f>^2)
        """
        val_mean = comm.reduce(self.average()[0], op=MPI.SUM, root=0)
        val_square_mean = comm.reduce(self.average()[1], op=MPI.SUM, root=0)

        if rank==0:
            integral_term = (self.b - self.a)**self.class_used.d
            integral = (val_mean)*integral_term/no_of_ranks

            variance_1 = 1/no_of_ranks * val_square_mean
            variance_2 = (1/no_of_ranks * val_mean)**2

            variance = 1/self.class_used.n * (variance_1 - variance_2)
            uncertainty = math.sqrt(variance) * integral_term
            print(f"Average = {self.average()[0]}, Integral = {integral},",
            f"Variance = {variance}")
            print(f"Therefore, the integral is {integral:.4f} ± {uncertainty:.4f}",
            f"units^{self.class_used.d}")
            print()
            return integral, uncertainty
        return None

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class Gaussian:
    """
    Finds the normal (gaussian) distribution across a range of values 'x', with a given offset
    'x_o' and distribtuion width 'sigma'
    """
    def __init__(self, sigma, x_o, n):
        self.sigma = sigma
        self.x_o = x_o
        self.d = len(x_o)
        self.n = n
        t = np.random.uniform(-1, 1, self.n*self.d)
        x = t/(1 - t**2)
        self.x = x.reshape((self.n, self.d))
        self.t = t.reshape((self.n, self.d))

    def __str__(self):
        """
        Confirms which function is being used, printing the values for the offset and dist. width.
        """
        return f"Normal distribution with an offset {self.x_o} and dist. width {self.sigma}"

    def normalisation(self, t_val):
        """
        Expresses the term multiplying the integrand after substitution from 'x' to 't'.
        """
        return (1 + t_val**2)/((1 - t_val**2)**2)

    def integrand(self):
        """
        Returns the distribution by rewriting the gaussian function in terms of 't' (integration 
        by substitution).
        """
        exponential = np.exp(np.sum(-(self.x - self.x_o)**2, axis=1)/(2*(self.sigma)**2))
        norm = np.prod(self.normalisation(self.t), axis=1)
        term = 1/(self.sigma * math.sqrt(2*math.pi))
        return term * exponential * norm

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

#J = Gaussian(2, np.array( [2, 5, 8, 3, 11, 2] ), no_of_samples)
#if rank==0:
#    print(f"Gaussian Dist. in {J.d}D with offset r_o = {J.x_o} and dist. width σ = {J.sigma}:")
#J_MONTE = MonteCarlo(J, J.integrand, -1, 1)
#J_CALC = J_MONTE.parallelisation()

if rank==0:
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"The code took {execution_time} seconds to run for {no_of_ranks} processors")
    print()
