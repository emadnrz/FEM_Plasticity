"""Model-independent effective-stress integration helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


Array = NDArray[np.float64]
AXISYMMETRIC_COLUMNS = (0, 1, 2, 4)


def state_key(state: Any) -> tuple[Any, ...]:
    """Create a stable cache key for a dataclass-like material state."""

    values: list[Any] = []
    for name, value in vars(state).items():
        if isinstance(value, np.ndarray):
            values.append((name, value.shape, np.round(value, 12).tobytes()))
        else:
            values.append((name, float(value)))
    return tuple(values)


def integrate_axisymmetric(
    model: Any,
    committed_state: Any,
    soil_strain_increment: Array,
    dt: float,
    compute_tangent: bool = True,
):
    """Integrate effective stress and calculate the needed tangent columns."""

    strain = np.asarray(soil_strain_increment, dtype=float).reshape(6)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        response = model.integrate_raw(committed_state, strain, dt)
    tangent = np.zeros((6, 6), dtype=float)
    if compute_tangent:
        scale = max(float(np.linalg.norm(strain)), 1.0e-5)
        h = max(1.0e-9, 1.0e-5 * scale)
        for column in AXISYMMETRIC_COLUMNS:
            perturbed = strain.copy()
            perturbed[column] += h
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                plus = model.integrate_raw(committed_state, perturbed, dt)
            tangent[:, column] = (plus.stress - response.stress) / h
    response.tangent = tangent
    return response
