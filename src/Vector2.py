#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Vector2 Class for games

# Created by Will McGugan, modified extensively by CoolCat467

from __future__ import annotations

__title__ = "Vector2"
__author__ = "CoolCat467 & Will McGugan"
__version__ = "0.2.0"
__ver_major__ = 0
__ver_minor__ = 2
__ver_patch__ = 0

from math import sqrt
from typing import Any, Iterable, Iterator


class Vector2:
    """Vector with X and Y component"""

    __slots__ = ("x", "y")

    def __init__(
        self,
        x: int
        | float
        | list[int | float]
        | tuple[int | float, int | float] = 0,
        y: int | float = 0,
    ) -> None:
        if isinstance(x, (list, tuple)):
            x, y = x
        self.x = x
        self.y = y

    def __str__(self) -> str:
        return f"({self.x}, {self.y})"

    def __repr__(self) -> str:
        return f"Vector2({self.x}, {self.y})"

    @classmethod
    def from_points(
        cls, frompoint: Iterable[int | float] | "Vector2", topoint: "Vector2"
    ) -> Any:
        """Return a vector with the direction of from point to to point"""
        p1, p2 = list(frompoint), list(topoint)
        return cls(p2[0] - p1[0], p2[1] - p1[1])

    def get_magnitude(self) -> float:
        """Return the magnitude (length) of the vector"""
        return sqrt(self.x**2 + self.y**2)

    def get_distance_to(
        self, point: Iterable[int | float] | "Vector2"
    ) -> float:
        """Get the magnitude to a point"""
        vec: "Vector2" = self.__class__.from_points(point, self)
        return vec.get_magnitude()

    def normalize(self) -> None:
        """Normalize self (make into a unit vector) **IN PLACE**"""
        magnitude = self.get_magnitude()
        if not magnitude == 0:
            self.x /= magnitude
            self.y /= magnitude

    def __copy__(self) -> "Vector2":
        """Make a copy of self"""
        return self.__class__(self.x, self.y)

    copy = __copy__

    def get_normalized(self) -> "Vector2":
        """Return a normalized vector (heading)"""
        vec = self.copy()
        vec.normalize()
        return vec

    # rhs is Right Hand Side
    def __add__(self, rhs: "Vector2") -> "Vector2":
        return self.__class__(self.x + rhs.x, self.y + rhs.y)

    def __sub__(self, rhs: "Vector2") -> "Vector2":
        return self.__class__(self.x - rhs.x, self.y - rhs.y)

    def __neg__(self: "Vector2") -> "Vector2":
        return self.__class__(-self.x, -self.y)

    def __mul__(self, scalar: int | float) -> "Vector2":
        return self.__class__(self.x * scalar, self.y * scalar)

    def __truediv__(self, scalar: int | float) -> "Vector2":
        try:
            x, y = self.x / scalar, self.y / scalar
        except ZeroDivisionError:
            x, y = self.x, self.y
        return self.__class__(x, y)

    def __len__(self) -> int:
        return 2

    def __iter__(self) -> Iterator[int | float]:
        return iter([self.x, self.y])

    def __getitem__(self, idx: int) -> int | float:
        return (self.x, self.y)[idx]
