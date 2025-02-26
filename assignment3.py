#!/usr/bin/env python3

"""
This program makes use of object classes to define d-dimensional points with dimensional components
between the values of -1 and 1 (essentially choosing random points within a box of side length 2,
centred at the origin). From this, the magnitude of the d-dimensional vector can be found to 
determine whether or not the point lies within a d-dimensional ball (circle, sphere, etc.).
"""

import numpy as np

# ------------------------------------------------------------------------------------------------

class Point:
    """
    Chooses a random point within a box of 'd' dimensions with values between the range of -1 and 1
    """
    def __init__(self, d):
        self.d = d
        self.point = np.random.uniform(-1, 1, d)

    def __str__(self):
        """
        Returns the point 
        """
        return f"Point:({self.point})"

    def r_vector(self):
        """
        Calculates the magnitude of the d-dimensional vector by finding the Root Sum Square (RSS)
        of its constituent components
        """
        i = 0
        dim_sum = 0
        while i < self.d:
            dim_sum = dim_sum + (self.point[i])**2
            i = i + 1
        r = np.sqrt(dim_sum)
        return r

A = Point(2)
print("Example: 2-dimensional point", A)
A_sum = A.r_vector()
print(A_sum)

B = Point(3)
print(B)
B_sum = B.r_vector()
print(B_sum)
