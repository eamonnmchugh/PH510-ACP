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
        r = np.zeros((self.n, 1))
        while j < self.n:
            i = 0
            while i < self.d:
                dim_sum[j][0] = dim_sum[j][0] + (self.points[j][i])**2
                r[j][0] = np.sqrt(dim_sum[j][0])
                i = i + 1
            j = j + 1
        return r

    def box_location(self):
        """
        
        """
        within = 0
        j = 0
        while j < self.n:
            if self.r_vector()[j][0] <= 1:
                within = within + 1
                j = j + 1
            else:
                j = j + 1
        return within, within/self.n * 100
        

A = Points(2, 1)
print("Example: 2D", A)
A_sum = A.r_vector()
print("With magnitude", A_sum)
A_within = A.box_location()
print(A_within)

B = Points(3, 1)
print("Example: 3D", B)
B_sum = B.r_vector()
print("With magnitude", B_sum)

C = Points(2, 1000)
print("Example: 5 2D",C)
C_sum = C.r_vector()
print("With magnitudes", C_sum)
C_within = C.box_location()
print(C_within)
