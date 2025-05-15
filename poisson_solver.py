#!/bin/python3

"""
This code is suitably licensed:
https://github.com/eamonnmchugh/PH510-ACP/blob/Assignment-4/MIT%20Licence

This code creates a class 'PoissonSolver2D' to generate an NxN grid of points. The potentials and
charges of each point can be set (some presets have been built below). After setting these, the
grid is relaxed by 'bleeding' the potentials throughout the grid (i.e. repeated averaging of neighbouring points until an equilibrium is reached. After this, the potential can be evaluated at
any specified point using random walkers. These walkers freely move through the grid from a
starting point (i, j) until they reach a boundary position (x_b, y_b). After repeated use of the
walkers, a probability map (Green's function) is generated, telling us how often each site is
visited. This then can give us an estimate of the potential at the chosen starting point.
"""

import random
import numpy as np
import matplotlib.pyplot as plt

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class PoissonSolver2D:
    """
    Generates a square grid of length 'l' (in metres) spanning 'n' points in both directions. The
    spacing of between each grid point 'h' (in metres) is calculated using these values. The 
    potential 'phi' (in Volts) and charge 'f' (in Coulombs) of each grid point can be manually 
    set, however are initally set as shown below (phi at each point randomly set to a value
    between 0 and 10, and f simply set to 0). The parameters 'fixed_potentials'/'fixed_charges' 
    are automatically updated when a potential/charge is manually set. These store the coordinates
    of the points at which manual change has been made. 'd' is the dimensionality of the grid (in
    this case 2) and 'no_of_samples' tells the code how many times certian tasks should be
    repeated. These final two parameters are present for ease of use when running through a Monte
    Carlo.
    """
    def __init__(self, length, number_of_points, no_of_samples):
        self.l = length  # physical length in meters
        self.n = number_of_points  # number of grid points
        self.h = self.l / (self.n - 1)
        self.phi = np.random.uniform(0, 10, (self.n, self.n))
        self.f = np.zeros((self.n, self.n))
        self.fixed_potentials = set()
        self.fixed_charges = set()
        self.d = 2
        self.no_of_samples = no_of_samples

    def set_potential(self, x, y, potential):
        """
        Manually set the potential 'phi' at a specified grid point (x, y). The first 'if' statement
        simply checks if the desired point lies within the grid. If it does, then the potential at
        this point is updated to the desired value (measured in Volts).
        """
        if 0 <= x < self.n and 0 <= y < self.n:
            self.phi[x, y] = potential
            self.fixed_potentials.add((x, y))
        else:
            raise ValueError(f"Invalid coordinates: ({x}, {y}) lie outside grid bounds.")
        return self.phi

    def get_potential(self, x, y):
        """
        Empirical check of the potential 'phi' at a specified point (x, y). This can be used to
        check the accuracy of some of the other functions. Similar to the function 'set_potential'
        there is a check to see whether the specified point lies within the grid.
        """
        if 0 <= x < self.n and 0 <= y < self.n:
            return self.phi[x, y]
        else:
            raise ValueError(f"Invalid coordinates: ({x}, {y}) outside grid bounds.")

    def set_boundary_conditions(self, bc_type):
        """
        Applys preset boundary conditions. The potential along the boundary of the grid is set
        using one of the three prests below. This first one 'all_1V' sets all edges uniformly to
        1V. The second 'tb1_lr-1' sets the top and bottom edges to +1V, and the left and right
        edges to -1V. For the final preset 'tl2_b0_r-4', the top and left edges are set to +2V, the
        bottom edge to 0V, and the right edge to -4V.

        Upon setting one of these conditions, the boundaries are then added to the list of fixed
        potentials.
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
        Similar to the above definition, the charge thoughout the grid can be set to one of the
        below presets. As the name implies the first preset 'uniform_10C' sets the charge at every
        point to 10C. The second one 'linear_gradient_top_to_bottom' generates a uniform charge
        gradient from top to bottom, starting at 1C, and falling to 0C. The final preset
        'exp_decay' adds an exponentially decaying charge distribution, described by
        exp(-2000 * |r|), placed at the centre of the grid.
        """

        if distribution_type == 'uniform_10C':
            self.f[:, :] = 10

        elif distribution_type == 'linear_gradient_top_to_bottom':
            for i in range(self.n):
                self.f[i, :] = 1 - (i/(self.n - 1))

        elif distribution_type == 'exp_decay':
            x = np.linspace(0, self.l, self.n)
            y = np.linspace(0, self.l, self.n)
            x, y = np.meshgrid(x, y, indexing='ij')
            x0, y0 = self.l/2, self.l/2
            r = np.sqrt((x - x0)**2 + (y - y0)**2)
            self.f[:, :] = np.exp(-2000 * np.abs(r))
        return self.f

    def overrelax(self, max_iter=10000, tol=1e-10):
        """
        Update the potential at each point to be the average of its neighboring points (with 
        additional terms). Fixed potentials (those set by the user) remain unchanged. If
        neighbouring points are found outside the grid, then they are not counted for the average.
        This is run iteratively until equilibrium is reached. The process stops when the maximum
        change in potentials between two consecutive iterations 'max_delta' is smaller than the
        specified tolerance 'tol', or when the maximum number of iterations 'max_iter' is reached.
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
                break
        else:
            print(f"Maximum iterations ({max_iter}) reached without equilibrium.")
        return self.phi

    def boundary_check(self, i, j):
        """
        Simple check to see whether a point (i, j) lies on the boundary of the grid.
        """
        return i == 0 or j == 0 or i == self.n - 1 or j == self.n - 1

    def random_walk(self, starting_point_i, starting_point_j):
        """
        Simulates random walkers starting at (i, j). The walkers have equal probability of moving
        in any of the four directions (by changing i or j by +/-1). This 'walk' is repeated until
        the walker reaches a boundary, at which the potential is recorded. After 'no_of_samples'
        amount of random walkers have reached the boundary, the mean potential is determined,
        giving an estimate for the potential at the starting point (i, j).
        """
        potential = []
        for k in range(self.no_of_samples):
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
                potential.append(self.phi[i, j])
        return np.mean(potential)

    def random_walk_probabilities(self, starting_point_i, starting_point_j):
        """
        Simulates random walkers starting at (i, j), similar to the above function. This time
        however, each time a 'step' is taken, a counter for the number of 'visits' a site receives 
        is updated. Once the walker reaches a boundary, a counter of the number of boundary visits
        'boundary_hits' is updated. After 'no_of_samples' amount of random walkers have reached
        the boundary, the probability each boundary is reached by a random walker is determined
        by dividing the number of boundary hits by the number of walkers. This function therefore
        returns an array of boundary probabilities, i.e. the Laplacian Green's function.
        """
        prob_grid = np.zeros((self.n, self.n))
        self.site_visits = np.zeros((self.n, self.n))
        boundary_hits = {}

        # Initialize count for each boundary point
        for i in range(self.n):
            boundary_hits[(0, i)] = 0           # Bottom
            boundary_hits[(self.n - 1, i)] = 0  # Top
            boundary_hits[(i, 0)] = 0           # Left
            boundary_hits[(i, self.n - 1)] = 0  # Right

        for k in range(self.no_of_samples):
            i, j = starting_point_i, starting_point_j
            while not self.boundary_check(i, j):
                self.site_visits[(i, j)] += 1
                direction = random.choice(['up', 'down', 'left', 'right'])
                if direction == 'up':
                    i += 1
                elif direction == 'down':
                    i -= 1
                elif direction == 'left':
                    j -= 1
                elif direction == 'right':
                    j += 1
            boundary_hits[(i, j)] += 1

        # Fill the 2D probability grid
        for (i, j), count in boundary_hits.items():
            prob_grid[i, j] = count / self.no_of_samples
        return prob_grid

    def greens_charge(self):
        """
        Using the number of site_visits generated in the function 'random_walk_probabilities', the
        Green's function's contribution from the charge density is calculated. As the Green's
        function specifies the probability that a walker reaches a point (p, q) from a starting
        position (i, j), naturally it to proportional to the number of times a site is visited.
        """
        green_charge = np.zeros((self.n, self.n))
        for p in range(0, self.n):
            for q in range(0, self.n):
                green_charge[p, q] = self.h**2/self.no_of_samples * self.site_visits[p, q]
        return green_charge

    def greens_function(self, starting_point_i, starting_point_j):
        """
        This function combines the results of the functions 'random_walk_probabilities' and
        'greens_charge' to evaluate the total Green's function of the grid from a starting
        position (i, j).
        """
        i, j = starting_point_i, starting_point_j
        return self.random_walk_probabilities(i, j) + self.greens_charge

    def potential_via_greens(self, starting_point_i, starting_point_j):
        """
        Calculates the potential 'phi_greens' at a point (i, j) via the Green's function, combining
        the previous three functions. This is the culmination of the entire class, and the main
        test of the code by comparing the value returned from this function with the that
        generated from the over-relaxation method.
        """
        i, j = starting_point_i, starting_point_j
        greens_laplace = self.random_walk_probabilities(i, j)
        term1 = np.zeros((self.n, self.n))
        for x_b in range(0, self.n):
            for y_b in range(0, self.n):
                if self.boundary_check(x_b, y_b):
                    term1[x_b, y_b] = greens_laplace[x_b, y_b] * self.phi[x_b, y_b]
        term1_sum = np.sum(term1)
        term2 = np.sum(self.greens_charge(i, j) * self.f)
        phi_greens = term1_sum + term2
        return phi_greens

    def plot_value(self, value, title, decimal_places, cbar_label):
        """
        This final function plots a given array 'value' via a colour-map, with array-elements
        rounded to a set number of decimal places. The 'origin' selected plots the map such that
        the origin is at the bottom of the plot. 'extent' plots the grid in centimetres.
        """
        plt.figure()
        extent = [0, self.l * 100, 0, self.l * 100]  # convert to cm
        plt.imshow(np.round(value, decimal_places), origin='lower', extent=extent, cmap='viridis')
        plt.colorbar(label=cbar_label)
        plt.title(title)
        plt.xlabel("x (cm)")
        plt.ylabel("y (cm)")
        plt.grid(False)
        plt.show()
