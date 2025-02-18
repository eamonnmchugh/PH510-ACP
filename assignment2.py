#!/usr/bin/env python3

"""
This program makes use of object classes to define a set of functions for use in vector
calculations. By defining a set of rules for vectors in both Cartesian and Spherical-Polar
coordinates, and being given multiple trios of vertices that each describe a triangle in 3D space,
the area and internal angles of said triangle were determined.
"""

import math
import numpy as np

# ------------------------------------------------------------------------------------------------
# Task 1a: Initialise a vector object with the components of the vector

class Vector:
    """
    Vector class for 3D quantities
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
        if self.x == 0:
            theta = math.pi/2
        else:
            theta = math.atan(self.y/self.x)
        phi = math.acos(self.z/r)
        return Spherical(r, theta, phi)

    def angle(self, vert_b, vert_c):
        """
        Calculates the internal angles of a triangle defined by three vertices
        """
        vec_ab = Vector(vert_b.x - self.x, vert_b.y - self.y, vert_b.z - self.z)
        vec_ac = Vector(vert_c.x - self.x, vert_c.y - self.y, vert_c.z - self.z)
        vec_bc = Vector(vert_c.x - vert_b.x, vert_c.y - vert_b.y, vert_c.z - vert_b.z)
        vec_ba = Vector(self.x - vert_b.x, self.y - vert_b.y, self.z - vert_b.z)
        vec_ca = Vector(self.x - vert_c.x, self.y - vert_c.y, self.z - vert_c.z)
        vec_cb = Vector(vert_b.x - vert_c.x, vert_b.y - vert_c.y, vert_b.z - vert_c.z)
        angle_abc = math.acos(vec_ba.dot(vec_bc)/(vec_ba.norm() * vec_bc.norm()))
        angle_bac = math.acos(vec_ab.dot(vec_ac)/(vec_ab.norm() * vec_ac.norm()))
        angle_bca = math.acos(vec_cb.dot(vec_ca)/(vec_cb.norm() * vec_ca.norm()))
        return angle_abc * 180/math.pi, angle_bac * 180/math.pi, angle_bca * 180/math.pi

    def tri_area(self, vert_b, vert_c):
        """
        Calculates the area of a triangle with given vertices
        """
        vec_ab = Vector(vert_b.x - self.x, vert_b.y - self.y, vert_b.z - self.z)
        vec_ac = Vector(vert_c.x - self.x, vert_c.y - self.y, vert_c.z - self.z)
        cross = vec_ab.cross(vec_ac)
        return 0.5 * cross.norm()

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
        if phi > math.pi:
            self.theta = theta + math.pi
            self.phi = phi - math.pi
        else:
            self.theta = theta
            self.phi = phi
        self.x = self.r * math.sin(self.theta) * math.cos(self.phi)
        self.y = self.r * math.sin(self.theta) * math.sin(self.phi)
        self.z = self.r * math.cos(self.theta)
        if np.abs(self.x) < 0.0001:
            self.x = 0
        if np.abs(self.y) < 0.0001:
            self.y = 0
        if np.abs(self.z) < 0.0001:
            self.z = 0
        super().__init__(self.x, self.y, self.z)

    def __str__(self):
        """
        Assumes floating point when printing
        """
        return f"Spherical Vector:({self.r:.3f}, {self.theta:.3f}, {self.phi:.3f})"

    def __add__(self, other):
        """
        Addition of vectors in spherical coords via temporary convertion to cartesian
        """
        cartesian_sum = Vector(self.x + other.x, self.y + other.y, self.z + other.z)
        return cartesian_sum.cart_to_sphere()

    def __sub__(self, other):
        """
        Addition of vectors in spherical coords via temporary convertion to cartesian
        """
        cartesian_diff = Vector(self.x - other.x, self.y - other.y, self.z - other.z)
        return cartesian_diff.cart_to_sphere()

    def cross(self, other):
        """
        Computes the cross (vector) product of two vectors
        """
        sx = (self.y * other.z) - (self.z * other.y)
        sy = (self.z * other.x) - (self.x * other.z)
        sz = (self.x * other.y) - (self.y * other.x)
        cartesian_cross = Vector(sx, sy, sz)
        return cartesian_cross.cart_to_sphere()

    def magnitude(self):
        """
        Magnitude of a vector in spherical coordinates
        """
        return self.r

A2 = Spherical(1, 25*math.pi/180, 43*math.pi/180)
print("Initial spherical coord vector A2 is", A2)

B2 = Spherical(5, 60*math.pi/180, 9*math.pi/180)
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

Area_1 = Vert_A1.tri_area(Vert_B1, Vert_C1)
print("The area of the first triangle is", Area_1)

# Triangle 2 (with vertices at A = (-1, -1, -1), B = (0, -1, -1) and C = (-1, 0, -1))
Vert_A2 = Vector(-1, -1, -1)
Vert_B2 = Vector(0, -1, -1)
Vert_C2 = Vector(-1, 0, -1)

Area_2 = Vert_A2.tri_area(Vert_B2, Vert_C2)
print("The area of the second triangle is", Area_2)

# Triangle 3 (with vertices at A = (1, 0, 0), B = (0, 0, 1) and C = (0, 0, 0))
Vert_A3 = Vector(1, 0, 0)
Vert_B3 = Vector(0, 0, 1)
Vert_C3 = Vector(0, 0, 0)

Area_3 = Vert_A3.tri_area(Vert_B3, Vert_C3)
print("The area of the third triangle is", Area_3)

# Triangle 4 (with vertices at A = (0, 0, 0), B = (1, -1, 0) and C = (0, 0, 1))
Vert_A4 = Vector(0, 0, 0)
Vert_B4 = Vector(1, -1, 0)
Vert_C4 = Vector(0, 0, 1)

Area_4 = Vert_A4.tri_area(Vert_B4, Vert_C4)
print(f"The area of the fourth triangle is {Area_4:0.3f}")
print()

#-------------------------------------------------------------------------------------------------
# Task 3b: Evaluating the internal angles of the previous four triangles
print("Task 3b")

# Triangle 1
Angle_1 = Vert_A1.angle(Vert_B1, Vert_C1)
print(f"The internal angles of Triangle 1 were ABC = {Angle_1[0]:2.1f}°,",
f"BAC = {Angle_1[1]:2.1f}°", f"and BCA = {Angle_1[2]:2.1f}°")

# Triangle 2
Angle_2 = Vert_A2.angle(Vert_B2, Vert_C2)
print(f"The internal angles of Triangle 2 were ABC = {Angle_2[0]:2.1f}°,",
f"BAC = {Angle_2[1]:2.1f}°", f"and BCA = {Angle_2[2]:2.1f}°")

# Triangle 3
Angle_3 = Vert_A3.angle(Vert_B3, Vert_C3)
print(f"The internal angles of Triangle 3 were ABC = {Angle_3[0]:2.1f}°,",
f"BAC = {Angle_3[1]:2.1f}°", f"and BCA = {Angle_3[2]:2.1f}°")

# Triangle 4
Angle_4 = Vert_A4.angle(Vert_B4, Vert_C4)
print(f"The internal angles of Triangle 4 were ABC = {Angle_4[0]:2.1f}°,",
f"BAC = {Angle_4[1]:2.1f}°", f"and BCA = {Angle_4[2]:2.1f}°")
print()

#-------------------------------------------------------------------------------------------------
# Task 3c: Perform similar tasks to parts 3a and 3b, for points specified in spherical polar
# coordinates (angles in degrees)
print("Task 3c")

# Triangle 1, with vertices specified by spherical polar coordinates A = (0, 0, 0),
# B = (1, 0, 0) and C = (1, 90, 0)

Sph_A1 = Spherical(0, 0, 0)
Sph_B1 = Spherical(1, 0, 0)
Sph_C1 = Spherical(1, 90*math.pi/180, 0)

Sph_Area_1 = Sph_A1.tri_area(Sph_B1, Sph_C1)
Sph_Angle_1 = Sph_A1.angle(Sph_B1, Sph_C1)
print("The area of the first triangle is", Sph_Area_1)
print(f"With internal angles ABC = {Sph_Angle_1[0]:2.1f}°,", f"BAC = {Sph_Angle_1[1]:2.1f}°",
f"and BCA = {Sph_Angle_1[2]:2.1f}°")
print()

# Triangle 2, with vertices specified by spherical polar coordinates A = (1, 0, 0),
# B = (1, 90, 0) and C = (1, 90, 180)

Sph_A2 = Spherical(1, 0, 0)
Sph_B2 = Spherical(1, 90*math.pi/180, 0)
Sph_C2 = Spherical(1, 90*math.pi/180, math.pi)

Sph_Area_2 = Sph_A2.tri_area(Sph_B2, Sph_C2)
Sph_Angle_2 = Sph_A2.angle(Sph_B2, Sph_C2)
print("The area of the second triangle is", Sph_Area_2)
print(f"With internal angles ABC = {Sph_Angle_2[0]:2.1f}°,", f"BAC = {Sph_Angle_2[1]:2.1f}°",
f"and BCA = {Sph_Angle_2[2]:2.1f}°")
print()

# Triangle 3, with vertices specified by spherical polar coordinates A = (0, 0, 0),
# B = (2, 0, 0) and C = (2, 90, 0)

Sph_A3 = Spherical(0, 0, 0)
Sph_B3 = Spherical(2, 0, 0)
Sph_C3 = Spherical(2, 90*math.pi/180, 0)

Sph_Area_3 = Sph_A3.tri_area(Sph_B3, Sph_C3)
Sph_Angle_3 = Sph_A3.angle(Sph_B3, Sph_C3)
print("The area of the third triangle is", Sph_Area_3)
print(f"With internal angles ABC = {Sph_Angle_3[0]:2.1f}°,", f"BAC = {Sph_Angle_3[1]:2.1f}°",
f"and BCA = {Sph_Angle_3[2]:2.1f}°")
print()

# Triangle 4, with vertices specified by spherical polar coordinates A = (1, 90, 0),
# B = (1, 90, 180) and C = (1, 90, 270)

Sph_A4 = Spherical(1, 90*math.pi/180, 0)
Sph_B4 = Spherical(1, 90*math.pi/180, math.pi)
Sph_C4 = Spherical(1, 90*math.pi/180, 3*math.pi/2)

Sph_Area_4 = Sph_A4.tri_area(Sph_B4, Sph_C4)
Sph_Angle_4 = Sph_A4.angle(Sph_B4, Sph_C4)
print("The area of the fourth triangle is", Sph_Area_4)
print(f"With internal angles ABC = {Sph_Angle_4[0]:2.1f}°,",f"BAC = {Sph_Angle_4[1]:2.1f}°",
f"and BCA = {Sph_Angle_4[2]:2.1f}°")
