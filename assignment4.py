#!/bin/python3

"""
This code is suitably licensed:
https://github.com/eamonnmchugh/PH510-ACP/blob/Assignment-4/MIT%20Licence
"""

import time
import numpy as np
#from mpi4py import MPI
#from monte_carlo import MonteCarlo

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Initialising the MPI environment and drawing key information from it. The number of ranks allows
# us to know how many processors are being used, which allows for equal distribtuion of the
# workload between the processors.
#comm = MPI.COMM_WORLD
#no_of_ranks = comm.Get_size()
#rank = comm.Get_rank()

# Recording the start time of the code
#if rank==0:
#    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
#    print(f"{no_of_ranks} Processors:")
#    print()
start_time = time.time()

# Setting the number of samples to be inversely proportional to the number of ranks. Each rank runs
# the set number of samples, meaning the total number of samples used is equal for all number of
# ranks.
# no_of_samples = np.int32(100000000/no_of_ranks)




class ChargeGridWithSpacing:
    def __init__(self, N, h):
        """
        Initialize the grid with N x N points, separated by a distance h.
        All points start with no charge (charge = 0).
        
        :param N: Size of the grid (NxN)
        :param h: Distance between consecutive points in the grid
        """
        self.N = N
        self.h = h
        # Create an NxN grid of points, initialized with no charge (0).
        # Grid coordinates will be spaced by h.
        self.grid = np.zeros((N, N))
    
    def set_charge(self, x, y, charge):
        """
        Set a specified charge at the grid point (x, y).
        
        :param x: Row index (0 <= x < N)
        :param y: Column index (0 <= y < N)
        :param charge: Charge value to set at (x, y)
        """
        if 0 <= x < self.N and 0 <= y < self.N:
            self.grid[x, y] = charge
        else:
            raise ValueError(f"Invalid coordinates: ({x}, {y}) outside grid bounds.")
    
    def get_charge(self, x, y):
        """
        Get the charge at the grid point (x, y).
        
        :param x: Row index (0 <= x < N)
        :param y: Column index (0 <= y < N)
        :return: Charge at point (x, y)
        """
        if 0 <= x < self.N and 0 <= y < self.N:
            return self.grid[x, y]
        else:
            raise ValueError(f"Invalid coordinates: ({x}, {y}) outside grid bounds.")

    def charge_neighbours(self):
        """
        
        """
        i, j = 0
        while i < self.N:
            while j < self.N:
                neighbours = np.array( [self.grid[i+1, j], self.grid[i-1, j], self.grid[i, j-1],
                self.grid[i, j+1] ] )
                self.grid[i, j] = np.mean(neighbours)

    def get_physical_coordinates(self, x, y):
        """
        Get the physical coordinates of a point in the grid.
        
        :param x: Row index (0 <= x < N)
        :param y: Column index (0 <= y < N)
        :return: Physical coordinates (x_coord, y_coord)
        """
        if 0 <= x < self.N and 0 <= y < self.N:
            # Convert grid index to physical coordinates (with distance h)
            x_coord = x * self.h
            y_coord = y * self.h
            return x_coord, y_coord
        else:
            raise ValueError(f"Invalid coordinates: ({x}, {y}) outside grid bounds.")
    
    def display_grid(self):
        """Display the current charge grid."""
        print(self.grid)
    
    def display_physical_positions(self):
        """Display the grid's physical coordinates and their charges."""
        for x in range(self.N):
            for y in range(self.N):
                x_coord, y_coord = self.get_physical_coordinates(x, y)
                charge = self.grid[x, y]
                print(f"Position ({x_coord:.2f}cm, {y_coord:.2f}cm) has charge: {charge}")

# Example usage
# Create a 5x5 grid with points separated by a distance of 2.0 units
charge_grid = ChargeGridWithSpacing(5, 2.0)

# Set some charges on specific grid points
charge_grid.set_charge(1, 1, 10)  # Set a charge of 10 at (1,1)
charge_grid.set_charge(3, 4, -5)  # Set a charge of -5 at (3,4)
charge_grid.set_charge(0, 0, 20)  # Set a charge of 20 at (0,0)

# Display the grid (charge values)
charge_grid.display_grid()

# Display the physical positions and their charges
print("\nPhysical positions and their charges:")
charge_grid.display_physical_positions()

# Get the charge at a specific point
print(f"\nCharge at (1, 1): {charge_grid.get_charge(1, 1)}")
print(f"Charge at (3, 4): {charge_grid.get_charge(3, 4)}")




# Recording the end time of the code, and taking the difference from the start time to find how
# long the code took to run. This allows for comparison of runtimes for varying number of
# processors. Getting an estimate of the parallel efficiency of the code.
#if rank==0:
end_time = time.time()
execution_time = end_time - start_time
print(f"The code took {execution_time} seconds to run for 1 processor")
print()

# If running using 8 processors, it might be beneficial to comment out the MPI.Finalize() command
# below. For an unknown reason, the runtime increases significantly: using a sample size of
# no_of_samples = 100000000/no_of_ranks, the runtime jumps from ~21 seconds to ~80 seconds with an
# uncommented MPI.Finalize().
#MPI.Finalize()
