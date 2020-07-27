#!/usr/bin/env python3
# Vector2 Class for games
# Original version by Will McGugan, modified extensively by CoolCat467

from math import sqrt

__version__ = '0.0.3'
NAME = 'Vector2 Module'

class Vector2(object):
    def __init__(self, x=0, y=0):
        if str(type(x)) in ("<class 'tuple'>", "<class 'list'>"):
            x, y = x
        self.x = x
        self.y = y
    
    def __str__(self):
        return "(%s, %s)" % (self.x, self.y)
    
    def __repr__(self):
        x, y = self.x, self.y
        return "Vector2(%s, %s)" % (x, y)
    
    @staticmethod
    def from_points(frompoint, topoint):
        """Return a vector with the direction of frompoint to topoint"""
        P1, P2 = list(frompoint), list(topoint)
        return Vector2(P2[0] - P1[0], P2[1] - P1[1])
    
    def get_magnitude(self):
        """Return the magnitude (length) of the vector"""
        return sqrt(self.x**2 + self.y**2)
    
    def get_distance_to(self, point):
        """Get the magnitude to a point"""
##        px, py = list(point)
##        sx, sy = list(self)
##        try:
##            px, py, sx, sy = float(px), float(py), float(sx), float(sy)
##        except TypeError as e:
##            print(type(point))
##            print(list(point))
##            raise TypeError(str(e))
##        return Vector2(px - sx, py - sy).get_magnitude()
        return Vector2.from_points(point, self).get_magnitude()
    
    def normalize(self):
        """Normalize self (make into a unit vector)"""
        magnitude = self.get_magnitude()
        if not magnitude == 0:
            self.x /= magnitude
            self.y /= magnitude
    
    def copy(self):
        """Make a copy of self"""
        return Vector2(self.x, self.y)
    
    def __copy__(self):
        return self.copy()
    
    def get_normalized(self):
        """Return a normalized vector (heading)"""
        vec = self.copy()
        vec.normalize()
        return vec
    
    #rhs is Right Hand Side
    def __add__(self, rhs):
        return Vector2(self.x + rhs.x, self.y + rhs.y)
    
    def __sub__(self, rhs):
        return Vector2(self.x - rhs.x, self.y - rhs.y)
    
    def __neg__(self):
        return Vector2(-self.x, -self.y)
    
    def __mul__(self, scalar):
        return Vector2(self.x * scalar, self.y * scalar)
    
    def __truediv__(self, scalar):
        try:
            x, y = self.x / scalar, self.y / scalar
        except ZeroDivisionError:
            x, y = self.x, self.y
        return Vector2(x, y)
    
    def __len__(self):
        return 2
    
    def __iter__(self):
        self.iterc = 0
        return self
    
    def __next__(self):
        if self.iterc <= 1:
            val = [self.x, self.y][self.iterc]
            self.iterc += 1
            return val
        else:
            raise StopIteration
    
    def __getitem__(self, x):
        return [self.x, self.y][x]
    pass
