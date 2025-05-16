#!/bin/python3

"""
This code is suitably licensed:
https://github.com/eamonnmchugh/PH510-ACP/blob/Assignment-3/MIT%20Licence

This program makes use of Monte Carlo simulations to find the estimate for a given function. Monte 
Carlo simulations are used to simplify complex integrations through repeated random sampling. This 
is done by finding the average value of the function across a specified range. This expectation 
value is then multiplied by the range, which approximates the integral of the function. Finally, 
the function's variance can be found to obtain the uncertainty in the integral's estimate.
"""

import numpy as np
from mpi4py import MPI

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
comm = MPI.COMM_WORLD
no_of_ranks = comm.Get_size()
rank = comm.Get_rank()

class MonteCarlo:
    """
    Runs the Monte Carlo method for computational analysis on a given function. This analytical
    method makes use of repeated random sampling to obtain numerical results, making integration
    of complex functions easier. Given a sufficently high number of attempts, uniformly 
    distributed inputs are achieved, negating randomness, and allowing for trivial solutions to
    complex calculus.

    Using prevously generated random values, and by being fed the function to be analysed, the
    expectation value of the function is found by summing the function's outputs and dividing by
    the number of points, <f> = 1/n * Σn (f(n)). With this, a value for the integral of the 
    function is found between given limits 'a' and 'b' by multiplying the range of values by the
    expectation value, I = (b - a)<f>. Finally, the variance in the function is found by dividing 
    the diffence between the expectation value of the squared function and the function's 
    expectation value squared by the number of samples, σ^2 = 1/n * (<f^2> - <f>^2).

    The uncertainty in the value obatined for the integral is the square root of the variance.

    The number of smaples 'n' and dimensions 'd' are taken from the class the analysed function
    belongs to.
    """
    def __init__(self, class_used, function, a, b, *args, **kwargs):
        self.a = a
        self.b = b
        self.class_used = class_used
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.value = function(*args, **kwargs)

    def __str__(self):
        """
        Confirms which function the Monte Carlo simulation is approximating.
        """
        return f"Monte Carlo simulation of the function {self.function}"

    def average(self):
        """
        Calculates the average value of a given function, <f> = 1/n * Σn (f(n)), stored in the term
        'average'. The expectation value of the squared values <f^2> is also calculated and stored
        in the term 'average_2'.
        """
        value_2 = np.square(self.value)
        average = 1/self.class_used.n * np.sum(self.value)
        average_2 = 1/self.class_used.n * np.sum(value_2)
        return average, average_2

    def average_array(self):
        """
        Identical definition to the one above, however this one leaves the inputed value as an
        array. As such, both averages 1 and 2 are returned as an array.
        """
        value_2 = np.square(self.value)
        average = 1/self.class_used.n * self.value
        average_2 = 1/self.class_used.n * value_2
        return average, average_2

    def parallelisation(self):
        """
        Makes use of computational parallelisation to sum the averages (and square of the 
        averages) of the given function into terms 'val_mean' (and 'val_square_mean') using the 
        MPI "reduce" command, bringing all values into the same processor "rank 0". Using these 
        the integral of the given function between limits 'a' and 'b', I = (b - a)<f> can be found.
        The range of values (b - a) must be accounted for all dimensions, and as such, is put to a 
        power of 'd', where d is the number of dimensions. The variance (error) is also calculated,
        σ^2 = 1/n * (<f^2> - <f>^2). After printing off these values, the function ends by 
        returning the values for the integral and its uncertainty.

        After the 'comm.reduce' command, all processors that are not rank=0 do nothing.
        """
        val_mean = comm.reduce(self.average()[0], op=MPI.SUM, root=0)
        val_square_mean = comm.reduce(self.average()[1], op=MPI.SUM, root=0)

        if rank==0:
            integral_term = (self.b - self.a)**self.class_used.d
            integral = (val_mean)*integral_term/no_of_ranks

            variance_1 = 1/no_of_ranks * val_square_mean
            variance_2 = (1/no_of_ranks * val_mean)**2

            variance = 1/self.class_used.n**2 * (variance_1 - variance_2)
            uncertainty = np.sqrt(variance) * integral_term
#            print(f"Average = {self.average()[0]}, Integral = {integral},",
#            f"Variance = {variance}")
#            print(f"Therefore, the integral is {integral:.4f} ± {uncertainty:.4f}",
#            f"units^{self.class_used.d}")
#            print()
            return integral, uncertainty
        return None

    def parallelisation_array(self):
        """
        Equivalent parallelisation for the 'average_array' function. Again, the mean, integral,
        variances and uncertainties are left as arrays.        
        """
        avg, avg_sq = self.average_array()

        # Element-wise reduction using MPI
        val_mean = comm.reduce(avg, op=MPI.SUM, root=0)
        val_sq_mean = comm.reduce(avg_sq, op=MPI.SUM, root=0)

        if rank == 0:
            # Element-wise mean across ranks
            mean = val_mean / no_of_ranks

            # Integral term for all elements (scalar multiplier)
            integral_term = (self.b - self.a) ** self.class_used.d
            integral = mean * integral_term

            # Variance: Var = (1/n^2) * (⟨f^2⟩ - ⟨f⟩^2)
            variance = (val_sq_mean / no_of_ranks - np.square(mean)) / self.class_used.n
            # Element-wise uncertainty
            uncertainty = np.sqrt(variance) * integral_term
            return mean, integral, np.mean(uncertainty)
        return None
