#!/usr/bin/env python3

"""
This program makes use of object classes to define d-dimensional points with dimensional components
between the values of -1 and 1 (essentially choosing random points within a box of side length 2,
centred at the origin). From this, the magnitude of the d-dimensional vector can be found to 
determine whether or not the point lies within a d-dimensional ball (circle, sphere, etc.).
"""

import math
import numpy as np

# ------------------------------------------------------------------------------------------------

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

class MonteCarlo:
    """
    Approximates a given function by finding the expected value (average), then integrates this
    average across the limits of the function 'a' and 'b'. Finally, the function's variance (error)
    is found.
    """
    def __init__(self, function, a, b, n):
        self.a = a
        self.b = b
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
        Calculates the average value of a given function.
        """
        value_2 = self.value**2
        average = 1/self.n * np.sum(self.value)
        average_2 = 1/self.n * np.sum(value_2)
        return average, average_2

    def integral(self):
        """
        Calculates the integral of a given function  between limits 'a' and 'b'.   
        """
        integral = (self.b - self.a) * self.average()[0]
        return integral

    def variance(self):
        """
        Calculates the variance (error) of a given function.
        """
        variance = 1/self.n * (self.average()[1] - self.average()[0]**2)
        return variance

    def calculations(self):
        """
        Returns all three desired values at once.
        """
        return self.average()[0], self.integral(), self.variance()

class Gaussian:
    """
    Finds the normal (gaussian) distribution across a range of values 'x', with a given offset
    'x_o' and distribtuion width 'sigma'
    """
    def __init__(self, sigma, x_o, d, n):
        self.sigma = sigma
        self.x_o = x_o
        self.d = d
        self.n = n
        t = np.random.uniform(-1, 1, self.n*self.d)
        x = np.zeros(self.n*self.d)
        j = 0
        while j < self.n*self.d:
            x[j] = t[j]/(1 - t[j]**2)
            j = j + 1
        self.x = x
        self.t = t

    def __str__(self):
        """
        Confirms which function is being used.
        """
        return f"Normal distribution across 'x' with an offset 'x_o' and dist. width 'sigma'"

    def normalisation(self):
        """
        Expresses the term multiplying the integrand after substitution from 'x' to 't'.
        """
        return (1 + self.t**2)/((1 - self.t**2)**2)

    def normal(self):
        """
        Returns the distribution.
        """
        j = 0
        exponential = np.zeros(self.d*self.n)
        while j < self.d*self.n:
            exponential[j]= math.exp(-(np.abs(self.x[j]- self.x_o))**2/(2*(self.sigma)**2))
            j = j + 1
        term = 1/(self.sigma * math.sqrt(2*math.pi))
        return term * exponential

    def integrand(self):
        """
        Rewrites the gaussian function in terms of 't' (integration by substitution).
        """
        return self.normal() * self.normalisation()**self.d

A = Points(2, 1000)
A_Monte = MonteCarlo(A.box_location, 0, 1, 1000)
A_Calc = A_Monte.calculations()
print(A)
print(f"Average = {A_Calc[0]:.4f}", f"Integral = {A_Calc[1]:.4f}",
f"Variance = {A_Calc[2]:.4f}")

B = Gaussian(1, 0, 1, 10000)
B_Monte = MonteCarlo(B.integrand, -1, 1, 10000)
B_Calc = B_Monte.calculations()
print(f"Average = {B_Calc[0]:.4f}", f"Integral = {B_Calc[1]:.4f}",
f"Variance = {B_Calc[2]:.4f}")
