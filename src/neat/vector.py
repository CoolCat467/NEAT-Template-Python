"""Vector2 Class for games."""

# Original version by Will McGugan, modified extensively by CoolCat467
# Programmed by CoolCat467

from __future__ import annotations

__title__ = "Vector2 Module"
__author__ = "CoolCat467"
__version__ = "1.0.7"
__ver_major__ = 1
__ver_minor__ = 0
__ver_patch__ = 7

import math
import sys
from typing import TYPE_CHECKING, NamedTuple, Self, override

if TYPE_CHECKING:  # pragma: nocover
    from collections.abc import Iterable


class Vector2(NamedTuple):
    """Vector2 Object. Takes an x and a y choordinate."""

    x: float
    y: float

    @classmethod
    def from_iter(cls, iterable: Iterable[float]) -> Self:
        """Return new vector from iterable."""
        return cls(*iter(iterable))

    @classmethod
    def from_points(
        cls,
        from_point: Iterable[float],
        to_point: Iterable[float],
    ) -> Self:
        """Return a vector with the direction of frompoint to topoint."""
        return cls.from_iter(to_point) - from_point

    @classmethod
    def from_radians(
        cls,
        radians: float,
        distance: float = 1,
    ) -> Self:
        """Return vector from angle in radians."""
        return cls(math.cos(radians), math.sin(radians)) * distance

    @classmethod
    def from_degrees(
        cls,
        degrees: float,
        distance: float = 1,
    ) -> Self:
        """Return vector from angle in degrees.

        Angle is measured from the positive X axis counterclockwise.
        """
        return cls.from_radians(math.radians(degrees), distance)

    def magnitude(self) -> float:
        """Return the magnitude (length) of self."""
        return math.hypot(*self)

    def get_distance_to(self, point: Iterable[float]) -> float:
        """Return the magnitude (distance) to a given point."""
        # return self.from_points(point, self).magnitude()
        return math.dist(point, self)

    def normalized(self) -> Self:
        """Return a normalized (unit) vector."""
        return self / self.magnitude()

    def heading_radians(self) -> float:
        """Returns the arc tangent (measured in radians) of self.y/self.x."""
        return math.atan2(self.y, self.x)

    def heading(self) -> float:
        """Returns the arc tangent (measured in degrees) of self.y/self.x.

        Angle is measured from the positive X axis counterclockwise.
        """
        return math.degrees(self.heading_radians())

    def rotate_radians(self, radians: float) -> Self:
        """Returns a new vector by rotating self around (0, 0) by radians."""
        new_heading = self.heading_radians() + radians
        return self.from_radians(new_heading, self.magnitude())

    def rotate(self, degrees: float) -> Self:
        """Returns a new vector by rotating self around (0, 0) by degrees.

        Angle is measured from the positive X axis counterclockwise.
        """
        return self.rotate_radians(math.radians(degrees))

    # rhs is Right Hand Side
    @override
    def __add__(  # type: ignore[override]
        self,
        rhs: Iterable[float],
    ) -> Self:
        """Return result of adding self components to rhs components."""
        return self.from_iter(a + b for a, b in zip(self, rhs, strict=True))

    def __sub__(self, rhs: Iterable[float]) -> Self:
        """Return result of subtracting self components from rhs components."""
        return self.from_iter(a - b for a, b in zip(self, rhs, strict=True))

    def __neg__(self) -> Self:
        """Return result of negating self components."""
        return self.from_iter(-c for c in self)

    @override
    def __mul__(self, scalar: float) -> Self:  # type: ignore[override]
        """Return result of multiplying self components by rhs scalar."""
        return self.from_iter(c * scalar for c in self)

    def __truediv__(self, scalar: float) -> Self:
        """Return result of dividing self components by rhs scalar."""
        return self.from_iter(c / scalar for c in self)

    def __floordiv__(self, scalar: float) -> Self:
        """Return result of floor division of self components by rhs scalar."""
        return self.from_iter(c // scalar for c in self)

    def __round__(self, ndigits: int | None = None) -> Self:
        """Return result of rounding self components to given number of digits."""
        return self.from_iter(round(c, ndigits) for c in self)

    def __abs__(self) -> Self:
        """Return result of absolute value of self components."""
        return self.from_iter(abs(c) for c in self)

    def __mod__(self, scalar: float) -> Self:
        """Return result of modulus of self components by rhs scalar."""
        return self.from_iter(c % scalar for c in self)

    def __divmod__(self, rhs: float) -> tuple[Self, Self]:
        """Return tuple of (self // rhs, self % rhs)."""
        x_div, x_mod = divmod(self.x, rhs)
        y_div, y_mod = divmod(self.y, rhs)
        return self.from_iter((x_div, y_div)), self.from_iter((x_mod, y_mod))

    def __matmul__(self, vec: Iterable[float]) -> float:
        """Return the dot product of this vector and another."""
        if sys.version_info >= (3, 12):
            # math.sumprod is new in python 3.12
            return math.sumprod(self, vec)
        return sum(a * b for a, b in zip(self, vec, strict=True))  # type: ignore  # pragma: nocover

    def dot(self, vec: Iterable[float]) -> float:
        """Return the dot product of this vector and another (same as @)."""
        return self @ vec


def get_angle_between_vectors(vec_a: Vector2, vec_b: Vector2) -> float:
    """Return the angle between two vectors (measured in radians)."""
    return math.acos((vec_a @ vec_b) / (vec_a.magnitude() * vec_b.magnitude()))


def project_v_onto_w(vec_v: Vector2, vec_w: Vector2) -> Vector2:
    """Return the projection of v onto w."""
    return vec_w * ((vec_v @ vec_w) / (vec_w.magnitude() ** 2))
