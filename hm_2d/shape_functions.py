"""Interpolation functions with no mesh or solver dependencies."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


Array = NDArray[np.float64]


def quad4_shape(natural_coordinates: Array) -> tuple[Array, Array]:
    """Return Quad4 values and natural gradients.

    The node ordering is bottom-left, bottom-right, top-right, top-left.  The
    gradient columns correspond to derivatives with respect to ``xi`` and
    ``eta``.
    """

    xi, eta = map(float, np.asarray(natural_coordinates, dtype=float).reshape(2))
    values = 0.25 * np.array(
        [
            (1.0 - xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 + eta),
            (1.0 - xi) * (1.0 + eta),
        ],
        dtype=float,
    )
    natural_gradients = 0.25 * np.array(
        [
            [-(1.0 - eta), -(1.0 - xi)],
            [1.0 - eta, -(1.0 + xi)],
            [1.0 + eta, 1.0 + xi],
            [-(1.0 + eta), 1.0 - xi],
        ],
        dtype=float,
    )
    return values, natural_gradients
