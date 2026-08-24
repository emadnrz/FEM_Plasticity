"""Axisymmetric Quad4 element kinematics and degree-of-freedom maps."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi

import numpy as np
from numpy.typing import NDArray

from .mesh import AxisymmetricMesh, Quad4Element
from .shape_functions import quad4_shape


Array = NDArray[np.float64]


@dataclass(frozen=True)
class AxisymmetricKinematics:
    shape: Array
    gradients: Array
    strain_displacement: Array
    determinant: float
    radius: float
    integration_weight: float


def axisymmetric_kinematics(
    coordinates: Array,
    natural_coordinates: Array,
    quadrature_weight: float = 1.0,
) -> AxisymmetricKinematics:
    """Evaluate geometry, gradients, and the six-component axisymmetric B."""

    coordinates = np.asarray(coordinates, dtype=float).reshape(4, 2)
    shape, natural_gradients = quad4_shape(natural_coordinates)
    jacobian = coordinates.T @ natural_gradients
    determinant = float(np.linalg.det(jacobian))
    if determinant <= 0.0:
        raise ValueError("Quad4 element has a nonpositive Jacobian")
    gradients = natural_gradients @ np.linalg.inv(jacobian)
    radius = float(shape @ coordinates[:, 0])
    if radius <= 0.0:
        raise ValueError("axisymmetric Gauss radius must be positive")

    # Material ordering: [rr, hoop, zz, r-hoop, rz, hoop-z].
    B = np.zeros((6, 8), dtype=float)
    for local_node, ((dr, dz), value) in enumerate(zip(gradients, shape, strict=True)):
        radial_dof = 2 * local_node
        axial_dof = radial_dof + 1
        B[0, radial_dof] = dr
        B[1, radial_dof] = value / radius
        B[2, axial_dof] = dz
        B[4, radial_dof] = dz
        B[4, axial_dof] = dr

    weight = 2.0 * pi * radius * determinant * float(quadrature_weight)
    return AxisymmetricKinematics(shape, gradients, B, determinant, radius, weight)


def displacement_dofs(element: Quad4Element) -> Array:
    return np.asarray(
        [2 * node + component for node in element.node_ids for component in range(2)],
        dtype=int,
    )


def pressure_dofs(mesh: AxisymmetricMesh, element: Quad4Element) -> Array:
    return mesh.displacement_dofs + np.asarray(element.node_ids, dtype=int)


def characteristic_length(coordinates: Array) -> float:
    """Return a conservative element length for pressure stabilization."""

    coordinates = np.asarray(coordinates, dtype=float).reshape(4, 2)
    edges = np.roll(coordinates, -1, axis=0) - coordinates
    return float(np.min(np.linalg.norm(edges, axis=1)))
