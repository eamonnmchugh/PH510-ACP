#!/usr/bin/env python3

"""
This program
"""

import math
import numpy as np

# ------------------------------------------------------------------------------------------------
# Task 1a: Initialise a vector object with the components of the vector

class Vector:
    """
    Vector class for 2D quantities
    """
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        """
        Assumes floating point when printing
        """
        return f"Vector:({self.x:.3f}, {self.y:.3f}, {self.z:.3f})"

    def __add__(self, other):
        """
        Overloads addition for the elements of two instances
        """
        return Vector(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        """
        Overloads subtraction for the elements of two instances
        """
        return Vector(self.x - other.x, self.y - other.y, self.z - other.z)

    def norm(self):
        """
        Computes the magnitude of a given vector
        """
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def dot(self, other):
        """
        Computes the dot (scalar) product of two vectors
        """
        return (self.x * other.x) + (self.y * other.y) + (self.z * other.z)

    def cross(self, other):
        """
        Computes the cross (vector) product of two vectors
        """
        sx = (self.y * other.z) - (self.z * other.y)
        sy = (self.z * other.x) - (self.x * other.z)
        sz = (self.x * other.y) - (self.y * other.x)
        return Vector(sx, sy, sz)

    def cart_to_sphere(self):
        """
        Converting cartesian coordinates to spherical
        """
        r = math.sqrt(self.x**2 + self.y**2 + self.z**2)
        theta = math.atan(self.y/self.x)
        phi = math.acos(self.z/r)
        return Spherical(r, theta, phi)

# ------------------------------------------------------------------------------------------------
# Task 1b: Print the vector an object holds
print("Task 1b")

A = Vector(2, 5, 7)

print("Initial vector A is", A)
print()

# ------------------------------------------------------------------------------------------------
# Task 1c: Calculate and return the magnitude of the vector the object holds
print("Task 1c")

A_mag = A.norm()

print("The magnitude of vector A is", A_mag)
print()

# ------------------------------------------------------------------------------------------------
# Task 1d: Add and subtract vectors, returning the resulting vector as an instance of the object
print("Task 1d")

B = Vector(3, 4, 2)

print("Secondary vector B is", B)

C = A + B
D = A - B
E = B - A
print("Addition of vectors A and B gives", C)
print("Subtraction of vector B from A returns", D)
print("Subtraction of vector A from B returns", E)
print()

# ------------------------------------------------------------------------------------------------
# Task 1e: Calculate the dot (scalar) and cross (vector) products between two vectors
print("Task 1e")

F = A.dot(B)
print("Dot product of A and B is", F)

G = A.cross(B)
print("Cross product of A and B is", G)
print()

# ------------------------------------------------------------------------------------------------
# Task 2: Inherit from Task 1 to make a new class that handle vectors in spherical-polar coords
print("Task 2")

class Spherical(Vector):
    """
    Inherits the Vector class to define a vector using polar-spherical coordinates
    """
    def __init__(self, r, theta, phi):
        self.r = r
        self.theta = theta
        self.phi = phi
        self.x = self.r * math.sin(self.theta) * math.cos(self.phi)
        self.y = self.r * math.sin(self.theta) * math.sin(self.phi)
        self.z = self.r * math.cos(self.theta)

    def __str__(self):
        """
        Assumes floating point when printing
        """
        return f"Spherical Vector:({self.r:.3f}, {self.theta:.3f}, {self.phi:.3f})"

    def __add__(self, other):
        """
        Addition of vectors in spherical coords via temporary convertion to cartesian
        """
        Cartesian_Sum = Vector(self.x + other.x, self.y + other.y, self.z + other.z)
        return Cartesian_Sum.cart_to_sphere()

    def __sub__(self, other):
        """
        Addition of vectors in spherical coords via temporary convertion to cartesian
        """
        Cartesian_Diff = Vector(self.x - other.x, self.y - other.y, self.z - other.z)
        return Cartesian_Diff.cart_to_sphere()

    def cross(self, other):
        """
        Computes the cross (vector) product of two vectors
        """
        sx = (self.y * other.z) - (self.z * other.y)
        sy = (self.z * other.x) - (self.x * other.z)
        sz = (self.x * other.y) - (self.y * other.x)
        Cartesian_Cross = Vector(sx, sy, sz)
        return Cartesian_Cross.cart_to_sphere()

    def magnitude(self):
        """
        Magnitude of a vector in spherical coordinates
        """
        return self.r

A2 = Spherical(1, 25*np.pi/180, 43*np.pi/180)
print("Initial spherical coord vector A2 is", A2)

B2 = Spherical(5, 60*np.pi/180, 9*np.pi/180)
print("Secondary vector B2 is", B2)

C2 = A2 + B2
D2 = A2 - B2
E2 = B2 - A2
F2 = A2.dot(B2)
G2 = A2.cross(B2)
print("Addition of vectors A2 and B2 gives", C2)
print("Subtraction of vector B2 from A2 returns", D2)
print("Subtraction of vector A2 from B2 returns", E2)
print("Dot product of A2 and B2 is", F2)
print("Cross product of A2 and B2 is", G2)
print()





