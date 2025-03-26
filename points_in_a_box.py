#!/bin/python3

"""
This code is suitably licensed:
https://github.com/eamonnmchugh/PH510-ACP/blob/Assignment-3/MIT%20Licence

This code creates a class 'Points' to obtain 'n' number of points of d-dimensions by generating
random values between -1 and 1. This is equivalent to choosing random points within a d-dimensional
box of side length 2 centred at the origin. Finding the magnitudes allows us to determine how far
from the origin the points are. By choosing a condition to test if the magnitudes are less than or
equal to 1, it is determined whether the points lie within a hypothetical unitary ball enclosed by
the box. The array 'within' records whether the points lie inside (outside) the ball by setting 
its corresponding term to a value of 1 (0). 
"""

import numpy as np

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class Points:
    """
    Generates 'n' random points of 'd' dimensions with constituent coordinate values between the 
    range of -1 and 1. This let's us simulate random points being generated in a 'd' dimensional 
    box with side lengths of 2, centred at the origin. The NumPy function 'random.uniform' is used
    to generate random values between -1 and 1 within an array. These values are then taken as d-
    dimensional points by reshaping the array to have d columns. As such, for n points, n*d random
    values were generated, then reshaped. This is executed in the '__init__' function.

    After initialising the points, in the 'r_vector' function, the magnitudes of the individual
    points are found by summing the square of each dimensional coordinate, which are then stored in
    'dim_sum'. By doing this, it can be determined whether the points lie within a hypothetical d-
    dimensional ball of unit radius enclosed by the box. The array 'within' records whether the
    points lie inside (outside) the ball by setting its corresponding index to a value of 1 (0).
    """
    def __init__(self, d, n):
        self.n = n
        self.d = d
        points = np.random.uniform(-1, 1, self.d*self.n)
        self.points = points.reshape((self.n, self.d))

    def __str__(self):
        """
        Prints the individual points that were randomly generated, allowing for a quick check that
        all values are between the range of -1 and 1.
        """
        return f"Points:({self.points})"

    def r_vector(self):
        """
        Calculates the square of the magnitude of the d-dimensional vectors by finding the sum of
        the square of the points' coords. Then finds out how many of the random points fall within
        the dimensions of the ball (hit). As it is a unitary ball, the condition for a 'hit' is
        that the corresponding value of 'dim_sum' is less than or equal to 1. As one is the
        conditional value, the squares of the magnitudes are used, saving on computational power.
        """
        dim_sum = np.sum(self.points**2, axis=1)
        within = (dim_sum <= 1).astype(int)
        return within
