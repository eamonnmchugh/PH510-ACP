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

    def angle(self, other):
        """
        Calculates the angle between two vectors
        """
        return math.acos(self.dot(other)/(self.norm() * other.norm()))

    def negative(self):
        """
        Returns a vector that is anti-parallel to the given vector
        """
        return Vector(-self.x, -self.y, -self.z)  


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

#-------------------------------------------------------------------------------------------------
# Task 3a: Evaluating the area of four triangles given vertices in Cartesian points
print("Task 3a")

# Triangle 1 (with vertices at A = (0, 0, 0), B = (1, 0, 0) and C = (0, 1, 0))
Vert_A1 = Vector(0, 0, 0)
Vert_B1 = Vector(1, 0, 0)
Vert_C1 = Vector(0, 1, 0)

AB = Vert_B1 - Vert_A1
AC = Vert_C1 - Vert_A1
BC = Vert_C1 - Vert_B1
BAC_Angle = AB.angle(AC)

Area_1 = 0.5 * AB.norm() * AC.norm() * math.sin(BAC_Angle)
print("The area of the first triangle is", Area_1)

# Triangle 2 (with vertices at A = (-1, -1, -1), B = (0, -1, -1) and C = (-1, 0, -1))
Vert_A2 = Vector(-1, -1, -1)
Vert_B2 = Vector(0, -1, -1)
Vert_C2 = Vector(-1, 0, -1)

AB_2 = Vert_B2 - Vert_A2
AC_2 = Vert_C2 - Vert_A2
BC_2 = Vert_C2 - Vert_B2
BAC_Angle_2 = AB_2.angle(AC_2)

Area_2 = 0.5 * AB_2.norm() * AC_2.norm() * math.sin(BAC_Angle_2)
print("The area of the second triangle is", Area_2)

# Triangle 3 (with vertices at A = (1, 0, 0), B = (0, 0, 1) and C = (0, 0, 0))
Vert_A3 = Vector(1, 0, 0)
Vert_B3 = Vector(0, 0, 1)
Vert_C3 = Vector(0, 0, 0)

AB_3 = Vert_B3 - Vert_A3
AC_3 = Vert_C3 - Vert_A3
BC_3 = Vert_C3 - Vert_B3
BAC_Angle_3 = AB_3.angle(AC_3)

Area_3 = 0.5 * AB_3.norm() * AC_3.norm() * math.sin(BAC_Angle_3)
print("The area of the third triangle is", Area_3)

# Triangle 4 (with vertices at A = (0, 0, 0), B = (1, -1, 0) and C = (0, 0, 1))
Vert_A4 = Vector(0, 0, 0)
Vert_B4 = Vector(1, -1, 0)
Vert_C4 = Vector(0, 0, 1)

AB_4 = Vert_B4 - Vert_A4
AC_4 = Vert_C4 - Vert_A4
BC_4 = Vert_C4 - Vert_B4
BAC_Angle_4 = AB_4.angle(AC_4)

Area_4 = 0.5 * AB_4.norm() * AC_4.norm() * math.sin(BAC_Angle_4)
print("The area of the fourth triangle is", Area_4)
print()

#-------------------------------------------------------------------------------------------------
# Task 3b: Evaluating the internal angles of the previous four triangles
print("Task 3b")

# Triangle 1
BA = AB.negative()
CA = AC.negative()
CB = BC.negative()

ABC_Angle = BA.angle(BC)
ACB_Angle = CA.angle(CB)
print("The internal angles of Triangle 1 were BAC =", BAC_Angle, "rad, ABC =", ABC_Angle, "rad and ACB =", ACB_Angle, "rad")

# Triangle 2
BA_2 = AB_2.negative()
CA_2 = AC_2.negative()
CB_2 = BC_2.negative()

ABC_Angle_2 = BA_2.angle(BC_2)
ACB_Angle_2 = CA_2.angle(CB_2)
print("The internal angles of Triangle 2 were BAC =", BAC_Angle_2, "rad, ABC =", ABC_Angle_2, "rad and ACB =", ACB_Angle_2, "rad")

# Triangle 3
BA_3 = AB_3.negative()
CA_3 = AC_3.negative()
CB_3 = BC_3.negative()

ABC_Angle_3 = BA_3.angle(BC_3)
ACB_Angle_3 = CA_3.angle(CB_3)
print("The internal angles of Triangle 3 were BAC =", BAC_Angle_3, "rad, ABC =", ABC_Angle_3, "rad and ACB =", ACB_Angle_3, "rad")

# Triangle 4
BA_4 = AB_4.negative()
CA_4 = AC_4.negative()
CB_4 = BC_4.negative()

ABC_Angle_4 = BA_4.angle(BC_4)
ACB_Angle_4 = CA_4.angle(CB_4)
print("The internal angles of Triangle 4 were BAC =", BAC_Angle_4, "rad, ABC =", ABC_Angle_4, "rad and ACB =", ACB_Angle_4, "rad")
print()

#-------------------------------------------------------------------------------------------------
# Task 3c: Perform similar tasks to parts 3a and 3b, for points specified in spherical polar coordinates, with angles in degrees
print("Task 3c")

# Triangle 1, with vertices specified by polar coordinates A = (0, 0, 0), B = (1, 0, 0) and C = (1, 90, 0)

Sph_A1 = Spherical(0, 0, 0)
Sph_B1 = Spherical(1, 0, 0)
Sph_C1 = Spherical(1, 90*np.pi/180, 0)




