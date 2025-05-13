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



class ChargeGridWithSmoothing:
    def __init__(self, N, h, tolerance=1e-100, iterations=10000):
        """
        Initialize the charge grid with N x N points and spacing h.

        :param N: Size of the grid (N x N)
        :param h: Distance between adjacent grid points
        :param tolerance: Tolerance for convergence (equilibrium condition)
        :param max_iter: Maximum number of iterations before stopping
        """
        self.N = N
        self.h = h
        self.tolerance = tolerance
        self.max_iter = iterations
        # Initialize a charge grid with all zeros initially.
        self.grid = np.zeros((N, N))
        # To track fixed charge points (those set by the user)
        self.fixed_charges = set()

    def set_charge(self, x, y, charge):
        """
        Set the charge at grid point (x, y).
        
        :param x: Row index (0 <= x < N)
        :param y: Column index (0 <= y < N)
        :param charge: The charge to set at (x, y)
        """
        if 0 <= x < self.N and 0 <= y < self.N:
            self.grid[x, y] = charge
            self.fixed_charges.add((x, y))
        else:
            raise ValueError(f"Invalid coordinates: ({x}, {y}) outside grid bounds.")

    def update_charges(self):
        """
        Update the charge at each point to be the average of its neighboring points.
        Fixed charges (those set by the user) remain unchanged.
        Boundary points (i.e., points where i=0 or i=N-1 or j=0 or j=N-1) 
        will only average neighboring points within the grid.
        """
        new_grid = np.copy(self.grid)  # Copy the grid to avoid modifying during iteration

        for i in range(self.N):
            for j in range(self.N):
                # Skip if the charge is fixed at this point
                if (i, j) in self.fixed_charges:
                    continue

                # Collect the neighboring points within the grid for averaging
                neighboring_charges = []

                # Check if the neighbor (i+1, j) is within bounds
                if i + 1 < self.N:
                    neighboring_charges.append(self.grid[i+1, j])
                
                # Check if the neighbor (i-1, j) is within bounds
                if i - 1 >= 0:
                    neighboring_charges.append(self.grid[i-1, j])
                
                # Check if the neighbor (i, j+1) is within bounds
                if j + 1 < self.N:
                    neighboring_charges.append(self.grid[i, j+1])
                
                # Check if the neighbor (i, j-1) is within bounds
                if j - 1 >= 0:
                    neighboring_charges.append(self.grid[i, j-1])

                # Calculate the average of valid neighboring charges
                if neighboring_charges:
                    new_grid[i, j] = np.mean(neighboring_charges)

        # Update the grid with the new values
        self.grid = new_grid

    def run_until_equilibrium(self):
        """
        Run the charge update process iteratively until equilibrium is reached.
        The process stops when the maximum change in charges between two consecutive iterations
        is smaller than the specified tolerance, or the maximum number of iterations is reached.
        """
        iteration = 0
        while iteration < self.max_iter:
            iteration += 1

            # Copy the current grid for comparison after update
            old_grid = np.copy(self.grid)

            # Perform charge update
            self.update_charges()

            # Calculate the maximum change between the old and new grid
            max_change = np.max(np.abs(self.grid - old_grid))

            # If the maximum change is less than the tolerance, we're done
            if max_change < self.tolerance:
                print(f"Equilibrium reached after {iteration} iterations.")
                break
        else:
            print(f"Maximum iterations ({self.max_iter}) reached without equilibrium.")

    def get_charge_at(self, x, y):
        """Get the charge at a specific grid point."""
        if 0 <= x < self.N and 0 <= y < self.N:
            return self.grid[x, y]
        else:
            raise ValueError("Invalid grid point.")

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
        print(np.round(self.grid, 2))
    
    def display_physical_positions(self):
        """Display the grid's physical coordinates and their charges."""
        for x in range(self.N):
            for y in range(self.N):
                x_coord, y_coord = self.get_physical_coordinates(x, y)
                charge = self.grid[x, y]
                print(f"Position ({x_coord:.2f}, {y_coord:.2f})cm has charge: {charge}")

# Example usage

# Create a 5x5 grid with spacing h = 1.0
N = 11
h = 1.0
charge_grid = ChargeGridWithSmoothing(N, h)

# Set some charges on specific grid points (e.g., a point charge at (2, 2))
charge_grid.set_charge(2, 2, 10)  # Charge at the center
charge_grid.set_charge(0, 0, 0)   # Charge at (1,1)
charge_grid.set_charge(10, 10, 10)  # Charge at (3,3)

# Display initial grid
print("Initial grid with specified charges:")
charge_grid.display_grid()

# Run the update process until equilibrium
charge_grid.run_until_equilibrium()

# Display updated grid
print("\nFinal grid after equilibrium:")
charge_grid.display_grid()
charge_grid.display_physical_positions()
# Get the charge at a specific point
print(f"\nCharge at (2, 2): {charge_grid.get_charge_at(2, 2)}")
print(f"Charge at (1, 1): {charge_grid.get_charge_at(1, 1)}")  # Fixed charge
print(f"Charge at (3, 3): {charge_grid.get_charge_at(3, 3)}")  # Fixed charge













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
