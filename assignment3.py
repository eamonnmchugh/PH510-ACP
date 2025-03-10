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
        j = 0
        points = np.zeros((self.n, self.d))
        while j < self.n:
            i = 0
            point = np.random.uniform(-1, 1, self.d)
            point = point.reshape((1, self.d))
            while i < self.d:
                points[j][i] = point[0][i]
                i = i + 1
            j = j + 1
        self.points = points

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
            i = 0
            while i < self.d:
                dim_sum[j][0] = dim_sum[j][0] + (self.points[j][i])**2
                i = i + 1
            j = j + 1
        return dim_sum

    def box_location(self):
        """
        Calculates how many of the random points fall within the dimensions of the ball (hit). The
        points hit if their magnitudes (or more accurately the square of their magnitudes) are one 
        or less.
        """
        within = 0
        j = 0
        while j < self.n:
            if self.r_vector()[j][0] <= 1:
                within = within + 1
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
    def __init__(self, sigma, x, x_o, d):
        self.sigma = sigma
        self.x = x
        self.x_o = x_o
        self.d = d

    def __str__(self):
        """
        Returns the distribution.
        """
        exponential = math.exp(-(np.abs(self.x - self.x_o))**2/(2*(self.sigma)**2))
        normalisation = 1/(self.sigma * math.sqrt(2*math.pi))
        return normalisation * exponential

# need to rewrite the normal function so it can be integrated over all space
# (look at function given on handout)

# also rewrite in multiple dimensions (by changing x and x_o to r and r_o)

A = Points(2, 1000)
A_Monte = MonteCarlo(A.box_location, 0, 1, 1000)
A_Calc = A_Monte.calculations()
print(A)
print(f"Average = {A_Calc[0]:.4f}", f"Integral = {A_Calc[1]:.4f}",
f"Variance = {A_Calc[2]:.4f}")


