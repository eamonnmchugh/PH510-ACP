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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
        self.d = 2

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

    def apply_charge_distribution(self, distribution_type):
        """
        
        """
        x = np.linspace(0, self.l, self.n)
        y = np.linspace(0, self.l, self.n)
        x, y = np.meshgrid(x, y, indexing='ij')

        if distribution_type == 'uniform_10C':
            self.f[:, :] = 10

        elif distribution_type == 'linear_gradient_top_to_bottom':
            for i in range(self.n):
                self.f[i, :] = 1 - (i/(self.n - 1))

        elif distribution_type == 'exp_decay':
            x0, y0 = self.l/2
            r = np.sqrt((x - x0)**2 + (y - y0)**2)
            self.f[:, :] = np.exp(-2000 * r)

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
                    rhs = (0.25 * self.h**2 * self.f[i, j]) + np.mean(neighboring_potentials)
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

    def random_walk(self, starting_point_i, starting_point_j, num_walkers=100000):
        """
        Simulates random walkers starting at (starting_point_i, starting_point_j),
        and returns the empirical probabilities of reaching each boundary point.

        :param starting_point_i: Starting row index
        :param starting_point_j: Starting column index
        :param num_walkers: Number of random walkers to simulate
        :return: A dictionary {(x, y): probability} for each boundary point (x, y)
        """
        values = []
        for k in range(num_walkers):
            i, j = starting_point_i, starting_point_j
            while not self.boundary_check(i, j):
                direction = random.choice(['up', 'down', 'left', 'right'])
                if direction == 'up':
                    i += 1
                elif direction == 'down':
                    i -= 1
                elif direction == 'left':
                    j -= 1
                elif direction == 'right':
                    j += 1
            if self.boundary_check(i, j):
                values.append(self.phi[i, j])
        return np.mean(values)

    def random_walk_probabilities(self, starting_point_i, starting_point_j, num_walkers=100000):
        """
        Simulates random walkers starting at (starting_point_i, starting_point_j),
        and returns the empirical probabilities of reaching each boundary point.

        :param starting_point_i: Starting row index
        :param starting_point_j: Starting column index
        :param num_walkers: Number of random walkers to simulate
        :return: A dictionary {(x, y): probability} for each boundary point (x, y)
        """
        prob_grid = np.zeros((self.n, self.n))
        site_visits = np.zeros((self.n, self.n))
        boundary_hits = {}

        # Initialize count for each boundary point
        for i in range(self.n):
            boundary_hits[(0, i)] = 0       # Bottom
            boundary_hits[(self.n - 1, i)] = 0  # Top
            boundary_hits[(i, 0)] = 0       # Left
            boundary_hits[(i, self.n - 1)] = 0  # Right

        for k in range(num_walkers):
            i, j = starting_point_i, starting_point_j
            while not self.boundary_check(i, j):
                direction = random.choice(['up', 'down', 'left', 'right'])
                if direction == 'up':
                    i += 1
                elif direction == 'down':
                    i -= 1
                elif direction == 'left':
                    j -= 1
                elif direction == 'right':
                    j += 1
                site_visits[(i, j)] += 1
            boundary_hits[(i, j)] += 1

        # Fill the 2D probability grid
        for (i, j), count in boundary_hits.items():
            prob_grid[i, j] = count / num_walkers
        return prob_grid, site_visits 

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

    def greens_function(self, starting_point_i, starting_point_j, num_walkers=100000):
        """
        
        """
        green_charge = np.zeros((self.n, self.n))
        i, j = starting_point_i, starting_point_j
        site_visits = self.random_walk_probabilities(i, j)[1]
        for p in range(0, self.n):
            for q in range(0, self.n): 
                green_charge[p, q] = self.h**2/num_walkers * site_visits[p, q]
        return green_charge

    def potential_via_greens(self, starting_point_i, starting_point_j, num_walkers=100000):
        """
        
        """
        i, j = starting_point_i, starting_point_j
        greens_laplace = self.random_walk_probabilities(i, j)[0]
        term1 = np.zeros((self.n, self.n))
        for x_b in range(0, self.n):
            for y_b in range(0, self.n):
                if self.boundary_check(x_b, y_b):
                    term1[x_b, y_b] = greens_laplace[x_b, y_b] * self.phi[x_b, y_b]
        term1_sum = np.sum(term1)
        term2 = np.sum(self.greens_function(i, j) * self.f)
        phi_greens = term1_sum + term2
        return phi_greens, greens_laplace, self.phi, term1, term2

    def plot_value(self, value, title, decimal_places):
        """
        
        """
        plt.figure()
        extent = [0, self.l * 100, 0, self.l * 100]  # convert to cm
        plt.imshow(np.round(value, decimal_places), origin='lower', extent=extent, cmap='viridis')
#        plt.imshow(np.round(self.phi, decimal_places), origin='lower', extent=extent, cmap='inferno')
        plt.colorbar(label='Potential (V)')
        plt.title(title)
        plt.xlabel("x (cm)")
        plt.ylabel("y (cm)")
        plt.grid(False)
        plt.show()

# Question 3
init_grid = PoissonSolver2D(0.10, 101)
phi, f = init_grid.phi, init_grid.f
print("Green's function evaluation for a square grid of side length 10cm:")

# (a)
a = init_grid.random_walk_probabilities(50, 50)
print(f"At centre point (5cm, 5cm):\n{a[0]}")

# (b)
b = init_grid.random_walk_probabilities(25, 25)
print(f"At (2.5cm, 2.5cm):\n{b[0]}")

# (c)
c = init_grid.random_walk_probabilities(1, 25)
print(f"At (0.1cm, 2.5cm):\n{c[0]}")

# (d)
d = init_grid.random_walk_probabilities(1, 1)
print(f"At (0.1cm, 0.1cm):\n{d[0]}")

# 0.05 / init_grid.h
#phi = example.set_boundary_conditions('tb1_lr-1')
#phi = example.set_potential(4, 4, 0)
#phi = example.set_potential(20, 30, 2)
#phi = example.set_potential(25, 25, 0)
#print(phi)
#phi = example.overrelax()
#random_walk = example.random_walk(3, 4)
#random_walk_prob = example.random_walk_probabilities(3, 4)
#print("Random Walk", random_walk)
#print("Prob")
#print(random_walk_prob[0])
#print("site visits")
#print(random_walk_prob[1])
#green = example.greens_function(3, 4)
#print("greens")
#print(green)
#phi_greens = example.potential_via_greens(3, 4)
#print("phi_greens")
#print(phi_greens[0])
#print(phi_greens[1])
#print(phi_greens[2])
#print(phi_greens[3])
#print(example.compute_potential_at_point(24, 24))
#print(example.get_potential(5, 5))
#example.plot_phi()












# Recording the end time of the code, and taking the difference from the start time to find how
# long the code took to run. This allows for comparison of runtimes for varying number of
# processors. Getting an estimate of the parallel efficiency of the code.
#if rank==0:
#end_time = time.time()
#execution_time = end_time - start_time
#print(f"The code took {execution_time} seconds to run for 1 processor")

# If running using 8 processors, it might be beneficial to comment out the MPI.Finalize() command
# below. For an unknown reason, the runtime increases significantly: using a sample size of
# no_of_samples = 100000000/no_of_ranks, the runtime jumps from ~21 seconds to ~80 seconds with an
# uncommented MPI.Finalize().
#MPI.Finalize()
