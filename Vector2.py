#!/usr/bin/env python3
# Vector2 Class for games
# Original version by Will McGugan, modified extensively by CoolCat467

from math import sqrt

__title__ = 'Vector2'
__author__ = 'CoolCat467 & Will McGugan'
__version__ = '0.1.0'
__ver_major__ = 0
__ver_minor__ = 1
__ver_patch__ = 0

class Vector2(object):
    __slots__ 'x', 'y'
    def __init__(self, x=0, y=0):
        if isinstance(x, (list, tuple)):
            x, y = x
        self.x = x
        self.y = y
    
    def __str__(self):
        return f'({self.x}, {self.y})'
    
    def __repr__(self):
        x, y = self.x, self.y
        return f'Vector2({self.x}, {self.y})'
    
    @staticmethod
    def from_points(frompoint, topoint):
        """Return a vector with the direction of frompoint to topoint"""
        P1, P2 = list(frompoint), list(topoint)
        return self.__class__(P2[0] - P1[0], P2[1] - P1[1])
    
    def get_magnitude(self):
        """Return the magnitude (length) of the vector"""
        return sqrt(self.x**2 + self.y**2)
    
    def get_distance_to(self, point):
        """Get the magnitude to a point"""
        return self.__class__.from_points(point, self).get_magnitude()
    
    def normalize(self):
        """Normalize self (make into a unit vector)"""
        magnitude = self.get_magnitude()
        if not magnitude == 0:
            self.x /= magnitude
            self.y /= magnitude
    
    def copy(self):
        """Make a copy of self"""
        return self.__class__(self.x, self.y)
    
    def __copy__(self):
        return self.copy()
    
    def get_normalized(self):
        """Return a normalized vector (heading)"""
        vec = self.copy()
        vec.normalize()
        return vec
    
    #rhs is Right Hand Side
    def __add__(self, rhs):
        return self.__class__(self.x + rhs.x, self.y + rhs.y)
    
    def __sub__(self, rhs):
        return self.__class__(self.x - rhs.x, self.y - rhs.y)
    
    def __neg__(self):
        return self.__class__(-self.x, -self.y)
    
    def __mul__(self, scalar):
        return self.__class__(self.x * scalar, self.y * scalar)
    
    def __truediv__(self, scalar):
        try:
            x, y = self.x / scalar, self.y / scalar
        except ZeroDivisionError:
            x, y = self.x, self.y
        return self.__class__(x, y)
    
    def __len__(self):
        return 2
    
    def __iter__(self):
        return iter([self.x, self.y])
    
    def __getitem__(self, idx):
        return [self.x, self.y][idx]
    pass
