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







