#!/bin/python3

"""
This code is suitably licensed:
https://github.com/eamonnmchugh/PH510-ACP/blob/Assignment-4/MIT%20Licence
"""

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



class ChargeGridWithSmoothing:
    def __init__(self, N, h):
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
        self.phi = np.zeros((N, N))
        self.f = np.zeros((N, N))
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
            self.phi[x, y] = charge
            self.fixed_charges.add((x, y))
        else:
            raise ValueError(f"Invalid coordinates: ({x}, {y}) outside grid bounds.")

    def set_boundary_conditions(self, bc_type):
        """
        Applying different preset boundary conditions
        """
        if bc_type == 'all_1V':
            self.phi[0, :] = 1      # Top
            self.phi[-1, :] = 1     # Bottom
            self.phi[:, 0] = 1      # Left
            self.phi[:, -1] = 1     # Right
        elif bc_type == 'tb1_lr-1':
            self.phi[0, :] = 1
            self.phi[-1, :] = 1
            self.phi[:, 0] = -1
            self.phi[:, -1] = -1
        elif bc_type == 'tl2_b0_r-4':
            self.phi[0, :] = 2
            self.phi[-1, :] = 0
            self.phi[:, 0] = 2
            self.phi[:, -1] = -4
        else:
            raise ValueError(f"Unknown boundary condition type: {bc_type}")

        # Add boundary points to fixed_potentials set
        for i in range(self.n):
            self.fixed_charges.add((self.n - 1, i))  # Top
            self.fixed_charges.add((0, i))           # Bottom
            self.fixed_charges.add((i, 0))           # Left
            self.fixed_charges.add((i, self.n - 1))  # Right

        return self.phi

    def relax(self, max_iter=10000, tol=1e-10):
        """
        Update the potential at each point to be the average of its neighboring points. Fixed
        potentials (those set by the user) remain unchanged. Boundary points (i.e., points where
        i=0 or i=N-1 or j=0 or j=N-1) will only average neighboring points within the grid. This is
        run iteratively until equilibrium is reached. The process stops when the maximum change in
        potentials between two consecutive iterations is smaller than the specified tolerance, or
        when the maximum number of iterations is reached.
        """
        omega = 2/(1 + np.sin(np.pi/self.n))
        new_phi = self.phi
        for iteration in range(max_iter):
            max_delta = 0
            for i in range(0, self.n):
                for j in range(0, self.n):
                    if (i, j) in self.fixed_charges:
                        continue

                    # Collect the neighboring points within the grid for averaging
                    neighboring_charges = []

                    # Check if the neighbor (i+1, j) is within bounds
                    if i + 1 < self.n:
                        neighboring_charges.append(self.phi[i+1, j])

                    # Check if the neighbor (i-1, j) is within bounds
                    if i - 1 >= 0:
                        neighboring_charges.append(self.phi[i-1, j])

                    # Check if the neighbor (i, j+1) is within bounds
                    if j + 1 < self.n:
                        neighboring_charges.append(self.phi[i, j+1])

                    # Check if the neighbor (i, j-1) is within bounds
                    if j - 1 >= 0:
                        neighboring_charges.append(self.phi[i, j-1])

                    old_phi = self.phi[i, j]
                    rhs = -(self.h**2 * self.f[i, j]) + np.mean(neighboring_charges)
                    new_phi[i,j] = (omega * rhs) + ((1 - omega) * old_phi)
                    max_delta = max(max_delta, abs(new_phi[i,j] - old_phi))
            self.phi = new_phi
            if max_delta < tol:
                print(f"Converged in {iteration} iterations.")
                print(np.round(self.phi, 2))
                break
        else:
            print(f"Maximum iterations ({max_iter}) reached without equilibrium.")
        return self.phi

    def get_charge_at(self, x, y):
        """Get the charge at a specific grid point."""
        if 0 <= x < self.N and 0 <= y < self.N:
            return self.phi[x, y]
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
        """
        Display the current charge grid.
        """
        print(np.round(self.phi, 2))
    
    def display_physical_positions(self):
        """
        Display the grid's physical coordinates and their charges.
        """
        for x in range(self.N):
            for y in range(self.N):
                x_coord, y_coord = self.get_physical_coordinates(x, y)
                charge = self.phi[x, y]
                print(f"Position ({x_coord:.2f}, {y_coord:.2f})cm has charge: {charge}")

    def plot_phi(self):
        """
        
        """
        extent = [0, self.l * 100, 0, self.l * 100]  # convert to cm
        plt.imshow(np.round(self.phi, 4), origin='lower', extent=extent, cmap='viridis')
        plt.colorbar(label='Potential (V)')
        plt.title("Potential Distribution")
        plt.xlabel("x (cm)")
        plt.ylabel("y (cm)")
        plt.grid(False)
        plt.show()













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
