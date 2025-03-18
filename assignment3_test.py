#!/usr/bin/env python3

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
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
no_of_ranks = comm.Get_size()
rank = comm.Get_rank()

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
        the square of the points' coords
        """
        j = 0
        dim_sum = np.zeros((self.n, 1))
        while j < self.n:
            dim_sum[j][0] = dim_sum[j][0] + np.sum(self.points[j]**2)
            j = j + 1
        return dim_sum

    def box_location(self):
        """
        Calculates how many of the random points fall within the dimensions of the ball (hit). The
        points hit if their magnitudes (or more accurately the square of their magnitudes) are one 
        or less.
        """
        within = np.zeros(self.n)
        j = 0
        while j < self.n:
            if self.r_vector()[j][0] <= 1:
                within[j] = within[j] + 1
            j = j + 1
        return within

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class MonteCarlo:
    """
    Approximates a given function by finding the expected value (average), then integrates this
    average across the limits of the function 'a' and 'b'. Finally, the function's variance (error)
    is found.
    """
    def __init__(self, function, a, b, d, n):
        self.a = a
        self.b = b
        self.d = d
        self.n = n
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
        value_2 = self.value**2
        average = 1/self.n * np.sum(self.value)
        average_2 = 1/self.n * np.sum(value_2)
        return average, average_2

#    def integral(self):
#        """
#        Calculates the integral of a given function  between limits 'a' and 'b', I = (b - a)<f>.
#        """
#        integral = (self.b - self.a) * self.average()[0]
#        return integral

#    def variance(self):
#        """
#        Calculates the variance (error) of a given function, σ^2 = 1/n * (<f^2> <f>^2).
#        """
#        variance = 1/self.n * (self.average()[1] - self.average()[0]**2)
#        return variance

#    def calculations(self):
#        """
#        Returns all three desired values at once.
#        """
#        return self.average()[0], self.integral(), self.variance()

    def parallelisation(self):
        """
        
        """
        val_mean = comm.reduce(self.average()[0], op=MPI.SUM, root=0)
        val_square_mean = comm.reduce(self.average()[1], op=MPI.SUM, root=0)
        
        if rank==0:
            print("rank", rank)
            integral = (self.b - self.a)**self.d
            integral_global = (val_mean)*integral/no_of_ranks
            total_samples = no_of_ranks * self.n
            
            variance_1 = 1/no_of_ranks * val_square_mean
            variance_2 = (1/no_of_ranks * val_mean)**2
            
            variance_global = 1/self.n * (variance_1 - variance_2)
            uncertainty_global = math.sqrt(variance_global) * integral
            print(f"Integral is {integral_global} ± {uncertainty_global}")
            return integral_global, uncertainty_global

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class Gaussian:
    """
    Finds the normal (gaussian) distribution across a range of values 'x', with a given offset
    'x_o' and distribtuion width 'sigma'
    """
    def __init__(self, sigma, x_o, d, n):
        self.sigma = sigma
        self.d = d
        self.n = n
        t = np.random.uniform(-1, 1, self.n*self.d)
        x = t/(1 - t**2)
        self.x = x.reshape((self.n, self.d))
        self.t = t.reshape((self.n, self.d))
        self.x_o = x_o

    def __str__(self):
        """
        Confirms which function is being used.
        """
        return f"Normal distribution across 'x' with offset {self.x_o} and dist width {self.sigma}"

    def normalisation(self, t_val):
        """
        Expresses the term multiplying the integrand after substitution from 'x' to 't'.
        """
        return (1 + t_val**2)/((1 - t_val**2)**2)

    def normal(self):
        """
        Returns the distribution.
        """
        j = 0
        exponential = np.zeros((self.n, 1))
        while j < self.n:
            exponential[j] = math.exp(np.sum(-((self.x[j] - self.x_o)**2/(2*(self.sigma)**2))))
            j = j + 1
        term = 1/(self.sigma * math.sqrt(2*math.pi))
        return term * exponential

    def integrand(self):
        """
        Rewrites the gaussian function in terms of 't' (integration by substitution).
        """
        j = 0
        norm = np.zeros((self.n, 1))
        while j < self.n:
            norm[j][0] = np.prod(self.normalisation(self.t[j]))
            j = j + 1
        return self.normal() * norm * 2**(self.d - 1)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#comm = MPI.COMM_WORLD
#no_of_ranks = comm.Get_size()
#rank = comm.Get_rank()

no_of_samples = 1000


print("Outside if statement: rank", rank)
print()
A = Points(2, no_of_samples)
A_Monte = MonteCarlo(A.box_location, 0, 1, 2, no_of_samples)
A_Calc = A_Monte.parallelisation()
#print(f"Region of a 2D ball:")
#print(f"Therefore, a 2D ball has a region of {A_Calc[0]:.4f} ± {np.sqrt(A_Calc[1]):.4f}",
#      f"units^{A.d}")
#print()


#A = Points(2, 1000)
#A_Monte = MonteCarlo(A.box_location, 0, 1, 1000)
#A_Calc = A_Monte.calculations()
#print(f"Region of a 2D ball:")
#print(f"Average = {A_Calc[0]:.4f}", f"Integral = {A_Calc[1]:.4f}",
#f"Variance = {A_Calc[2]:.4f}")
#print(f"Therefore, a 2D ball has a region of {A_Calc[1]:.4f} ± {np.sqrt(A_Calc[2]):.4f}",
#      f"units^{A.d}")
#print()

#A = Points(3, 1000)
#A_Monte = MonteCarlo(A.box_location, 0, 1, 1000)
#A_Calc = A_Monte.calculations()
#print(f"Region of a 3D ball:")
#print(f"Average = {A_Calc[0]:.4f}", f"Integral = {A_Calc[1]:.4f}",
#f"Variance = {A_Calc[2]:.4f}")
#print(f"Therefore, a 3D ball has a region of {A_Calc[1]:.4f} ± {np.sqrt(A_Calc[2]):.4f}",
#      f"units^{A.d}")
#print()

#A = Points(4, 1000)
#A_Monte = MonteCarlo(A.box_location, 0, 1, 1000)
#A_Calc = A_Monte.calculations()
#print(f"Region of a 4D ball:")
#print(f"Average = {A_Calc[0]:.4f}", f"Integral = {A_Calc[1]:.4f}",
#f"Variance = {A_Calc[2]:.4f}")
#print(f"Therefore, a 4D ball has a region of {A_Calc[1]:.4f} ± {np.sqrt(A_Calc[2]):.4f}",
#      f"units^{A.d}")
#print()

#A = Points(5, 1000)
#A_Monte = MonteCarlo(A.box_location, 0, 1, 1000)
#A_Calc = A_Monte.calculations()
#print(f"Region of a 5D ball:")
#print(f"Average = {A_Calc[0]:.4f}", f"Integral = {A_Calc[1]:.4f}",
#f"Variance = {A_Calc[2]:.4f}")
#print(f"Therefore, a 5D ball has a region of {A_Calc[1]:.4f} ± {np.sqrt(A_Calc[2]):.4f}",
#      f"units^{A.d}")
#print()


#B = Gaussian(1, 0, 1, 10000)
#B_Monte = MonteCarlo(B.integrand, -1, 1, 10000)
#B_Calc = B_Monte.calculations()
#print(f"For the 1D gaussian distribution, with an offset {B.x_o} and dist. width {B.sigma}")
#print(f"Average = {B_Calc[0]:.4f}", f"Integral = {B_Calc[1]:.4f}",
#f"Variance = {B_Calc[2]:.4f}")

#B = Gaussian(1, 0, 6, 10000)
#B_Monte = MonteCarlo(B.integrand, -1, 1, 10000)
#B_Calc = B_Monte.calculations()
#print(f"For the 6D gaussian distribution, with an offset {B.x_o} and dist. width {B.sigma}")
#print(f"Average = {B_Calc[0]:.4f}", f"Integral = {B_Calc[1]:.4f}",
#f"Variance = {B_Calc[2]:.4f}")


MPI.Finalize()
