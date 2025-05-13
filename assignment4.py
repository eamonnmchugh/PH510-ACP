#!/bin/python3

"""
This code is suitably licensed:
https://github.com/eamonnmchugh/PH510-ACP/blob/Assignment-4/MIT%20Licence
"""

import random
import time
import numpy as np
import matplotlib.pyplot as plt
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


class PoissonSolver2D:
    """
    
    """
    def __init__(self, length, number_of_points):
        self.l = length  # physical length in meters
        self.n = number_of_points  # number of grid points
        self.h = self.l / (self.n - 1)
        self.phi = np.random.uniform(-100, 1000, (self.n, self.n))
        self.f = np.zeros((self.n, self.n))
        self.fixed_potentials = set()
        self.fixed_charges = set()

    def set_potential(self, x, y, potential):
        """
        Set the potential at grid point (x, y).
        
        :param x: Row index (0 <= x < N)
        :param y: Column index (0 <= y < N)
        :param potential: The potential to set at (x, y)
        """
        if 0 <= x < self.n and 0 <= y < self.n:
            self.phi[x, y] = potential
            self.fixed_potentials.add((x, y))
        else:
            raise ValueError(f"Invalid coordinates: ({x}, {y}) outside grid bounds.")
        return self.phi

    def set_boundary_conditions(self, bc_type):
        """
        Applying different preset boundary conditions
        """
        if bc_type == 'all_1V':
            self.phi[-1, :] = 1  # Top
            self.phi[0, :] = 1   # Bottom
            self.phi[:, 0] = 1   # Left
            self.phi[:, -1] = 1  # Right
        elif bc_type == 'tb1_lr-1':
            self.phi[-1, :] = 1
            self.phi[0, :] = 1
            self.phi[:, 0] = -1
            self.phi[:, -1] = -1
        elif bc_type == 'tl2_b0_r-4':
            self.phi[-1, :] = 2
            self.phi[0, :] = 0
            self.phi[:, 0] = 2
            self.phi[:, -1] = -4
        else:
            raise ValueError(f"Unknown boundary condition type: {bc_type}")

        # Add boundary points to fixed_potentials set
        for i in range(self.n):
            self.fixed_potentials.add((self.n - 1, i))  # Top
            self.fixed_potentials.add((0, i))           # Bottom
            self.fixed_potentials.add((i, 0))           # Left
            self.fixed_potentials.add((i, self.n - 1))  # Right

        return self.phi

    def overrelax(self, max_iter=10000, tol=1e-10):
        """
        Update the potential at each point to be the average of its neighboring points. Fixed
        potentials (those set by the user) remain unchanged. Boundary points (i.e., points where
        i=0 or i=N-1 or j=0 or j=N-1) will only average neighboring points within the grid. This is
        run iteratively until equilibrium is reached. The process stops when the maximum change in
        potentials between two consecutive iterations is smaller than the specified tolerance, or
        when the maximum number of iterations is reached.
        """
        omega = 2/(1 + np.sin(np.pi/self.n))
        for iteration in range(max_iter):
            max_delta = 0
            for i in range(0, self.n):
                for j in range(0, self.n):
                    if (i, j) in self.fixed_potentials:
                        continue

                    # Collect the neighboring points within the grid for averaging
                    neighboring_potentials = []

                    # Check if the neighbor (i+1, j) is within bounds
                    if i + 1 < self.n:
                        neighboring_potentials.append(self.phi[i+1, j])

                    # Check if the neighbor (i-1, j) is within bounds
                    if i - 1 >= 0:
                        neighboring_potentials.append(self.phi[i-1, j])

                    # Check if the neighbor (i, j+1) is within bounds
                    if j + 1 < self.n:
                        neighboring_potentials.append(self.phi[i, j+1])

                    # Check if the neighbor (i, j-1) is within bounds
                    if j - 1 >= 0:
                        neighboring_potentials.append(self.phi[i, j-1])

                    old_phi = self.phi[i, j]
                    rhs = -(self.h**2 * f[i, j]) + np.mean(neighboring_potentials)
                    self.phi[i,j] = (omega * rhs) + ((1 - omega) * old_phi)
                    max_delta = max(max_delta, abs(self.phi[i,j] - old_phi))
            if max_delta < tol:
                print(f"Converged in {iteration} iterations.")
                print(np.round(self.phi, 2))
                break
        else:
            print(f"Maximum iterations ({max_iter}) reached without equilibrium.")
        return self.phi

    def boundary_check(self, i, j):
        """
        
        """
        return i == 0 or j == 0 or i == self.n - 1 or j == self.n - 1

    def get_potential(self, x, y):
        """
        Get the potential at the grid point (x, y).
        
        :param x: Row index (0 <= x < N)
        :param y: Column index (0 <= y < N)
        :return: potential at point (x, y)
        """
        if 0 <= x < self.n and 0 <= y < self.n:
            return self.phi[x, y]
        else:
            raise ValueError(f"Invalid coordinates: ({x}, {y}) outside grid bounds.")

    def plot_phi(self):
        """
        
        """
        extent = [0, self.l * 100, 0, self.l * 100]  # convert to cm
        plt.imshow(np.round(self.phi, 4), origin='lower', extent=extent, cmap='viridis')
#        plt.imshow(np.round(self.phi, 4), origin='lower', extent=extent, cmap='inferno')
        plt.colorbar(label='Potential (V)')
        plt.title("Potential Distribution")
        plt.xlabel("x (cm)")
        plt.ylabel("y (cm)")
        plt.grid(False)
        plt.show()

# Example usage
example = PoissonSolver2D(0.10, 50)
phi, f = example.phi, example.f
phi = example.set_boundary_conditions('tl2_b0_r-4')
phi = example.set_potential(9, 10, 2)
phi = example.set_potential(20, 30, 2)
phi = example.set_potential(25, 25, 0)
print(phi)
print()
phi = example.overrelax()
example.plot_phi()













# Recording the end time of the code, and taking the difference from the start time to find how
# long the code took to run. This allows for comparison of runtimes for varying number of
# processors. Getting an estimate of the parallel efficiency of the code.
#if rank==0:
end_time = time.time()
execution_time = end_time - start_time
print(f"The code took {execution_time} seconds to run for 1 processor")

# If running using 8 processors, it might be beneficial to comment out the MPI.Finalize() command
# below. For an unknown reason, the runtime increases significantly: using a sample size of
# no_of_samples = 100000000/no_of_ranks, the runtime jumps from ~21 seconds to ~80 seconds with an
# uncommented MPI.Finalize().
#MPI.Finalize()
