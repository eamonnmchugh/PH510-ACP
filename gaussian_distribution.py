#!/bin/python3

"""
This code is suitably licensed:
https://github.com/eamonnmchugh/PH510-ACP/blob/Assignment-3/MIT%20Licence

This code is used to determine the region contained by a Gaussian distribtuion with a given offset
and distribution width. This is done by generating 'n' random values between -1 and 1, then using
integration by substitution to 'change' these limits to -∞ and ∞. Because of this, an extra term is
added to account for the change in variables.
"""

import numpy as np

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class Gaussian:
    """
    Produces a d-dimensional Gaussian distribution with a given offset 'x_o' and distribution width
    'sigma' over all space by generating random points between 'x' values of -∞ and ∞. To simplify
    integrating over all space, integration by substitution is used by defining a new variable 't'
    such that x = t/(1 - t**2). Then, random values of t are generated between -1 and 1, and 
    converted to x. Both x and t are reshaped to be n x d arrays, where n is the number of samples,
    and d is the number of dimensions. The value for d is taken from the length of the offset 
    array.

    The values of x, x_o and sigma are used to express the Gaussian distribution in the 'integrand'
    function. As integration by substitution has taken place, an additional term is added to the
    Gaussian.
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
        Prints the values for the d-dimensional offset 'x_o' and the distribution width 'sigma'.
        """
        return f"Normal distribution with an offset {self.x_o} and dist. width {self.sigma}"

    def substitution_term(self, t_val):
        """
        Expresses the term multiplying the integrand after substitution from 'x' to 't'.
        """
        return (1 + t_val**2)/((1 - t_val**2)**2)

    def integrand(self):
        """
        Expresses the Gaussian distribution using a given d-dimensional offset 'x_o', a given 
        distribution width 'sigma' and randomly generated points 'x'. The terms 'exponential' and
        'normalisation' combine to give the Gaussian distribution. The substitution term must
        be accounted for in all dimensions, meaning this term must appear in the integrand d-
        times. This is done by taking the product of this term across the dimensions.
        """
        exponential = np.exp(np.sum(-(self.x - self.x_o)**2, axis=1)/(2*(self.sigma)**2))
        sub_term = np.prod(self.substitution_term(self.t), axis=1)
        normalisation = 1/(self.sigma * np.sqrt(2*np.pi))
        return normalisation * exponential * sub_term
