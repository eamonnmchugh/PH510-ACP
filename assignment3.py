#!/usr/bin/env python3

"""
This program makes use of object classes to define d-dimensional points with dimensional components
between the values of -1 and 1 (essentially choosing random points within a box of side length 2,
centred at the origin). From this, the magnitude of the d-dimensional vector can be found to 
determine whether or not the point lies within a d-dimensional ball (circle, sphere, etc.).
"""

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
            A = np.random.uniform(-1, 1, self.d)
            A_reshape = A.reshape((1, self.d))
            while i < self.d:
                points[j][i] = A_reshape[0][i]
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
        Calculates the magnitude of the d-dimensional vectors by finding the Root Sum Square (RSS)
        of the points' coords
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
        
        """
        within = 0
        j = 0
        while j < self.n:
            if self.r_vector()[j][0] <= 1:
                within = within + 1
            j = j + 1
        return within
        
    def MonteCarlo(self, function, a, b):
        """
        
        """
        A = function()
        A2 = A**2
        average = 1/self.n * np.sum(A)
        average_2 = 1/self.n * np.sum(A2)
        integral = (b - a) * average
        variance = 1/self.n * (average_2 - average**2)
        return average, integral, variance

#def normal(sigma, x, x_o):
#    """
#    
#    """
#    return 1/(sigma*math.sqrt(2*math.pi)) * math.exp(-(np.abs(x - x_o))**2/(2*(sigma)**2))

A = Points(2, 1000)
A_Monte = A.MonteCarlo(A.box_location, -1, 1)
print(A)
print(f"Average = {A_Monte[0]:.4f}", f"Integral = {A_Monte[1]:.4f}",
f"Variance = {A_Monte[2]:.4f}")


