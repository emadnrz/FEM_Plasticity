"""Numerical integration rules, kept independent of the FE assembler."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


Array = NDArray[np.float64]


@dataclass(frozen=True)
class GaussPoint:
    """Natural coordinates and integration weight of one quadrature point."""

    natural_coordinates: tuple[float, float]
    weight: float

    @property
    def coordinates(self) -> Array:
        return np.asarray(self.natural_coordinates, dtype=float)


def gauss_legendre_2x2() -> tuple[GaussPoint, ...]:
    """Return the tensor-product two-point Gauss rule for a quadrilateral."""

    a = 1.0 / np.sqrt(3.0)
    return (
        GaussPoint((-a, -a), 1.0),
        GaussPoint((a, -a), 1.0),
        GaussPoint((a, a), 1.0),
        GaussPoint((-a, a), 1.0),
    )


QUAD4_GAUSS_POINTS = gauss_legendre_2x2()