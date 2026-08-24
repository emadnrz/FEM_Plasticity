"""Volume-averaged fields for axisymmetric hydro-mechanical analyses."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .elements import axisymmetric_kinematics, displacement_dofs
from .mesh import AxisymmetricMesh
from .quadrature import QUAD4_GAUSS_POINTS


Array = NDArray[np.float64]


def _weighted_points(mesh: AxisymmetricMesh):
    for element_index, element in enumerate(mesh.elements):
        coordinates = mesh.element_coordinates(element)
        for point_index, point in enumerate(QUAD4_GAUSS_POINTS):
            kin = axisymmetric_kinematics(
                coordinates, point.coordinates, point.weight
            )
            yield element_index, point_index, kin


def volume_average_effective_stress(
    mesh: AxisymmetricMesh, states: list[list[Any]]
) -> Array:
    total = np.zeros(6, dtype=float)
    volume = 0.0
    for element_index, point_index, kin in _weighted_points(mesh):
        total += states[element_index][point_index].stress * kin.integration_weight
        volume += kin.integration_weight
    return total / volume


def volume_average_pressure(mesh: AxisymmetricMesh, nodal_pressure: Array) -> float:
    pressure = np.asarray(nodal_pressure, dtype=float).reshape(mesh.n_nodes)
    total = 0.0
    volume = 0.0
    for element_index, _, kin in _weighted_points(mesh):
        element = mesh.elements[element_index]
        value = float(kin.shape @ pressure[np.asarray(element.node_ids, dtype=int)])
        total += value * kin.integration_weight
        volume += kin.integration_weight
    return total / volume


def volume_average_strain(mesh: AxisymmetricMesh, displacement: Array) -> Array:
    displacement = np.asarray(displacement, dtype=float).reshape(mesh.displacement_dofs)
    total = np.zeros(6, dtype=float)
    volume = 0.0
    for element_index, _, kin in _weighted_points(mesh):
        element = mesh.elements[element_index]
        strain = kin.strain_displacement @ displacement[displacement_dofs(element)]
        total += strain * kin.integration_weight
        volume += kin.integration_weight
    return total / volume


def total_stress(effective_stress: Array, pressure: float, biot: float = 1.0) -> Array:
    """Return compression-positive total stress from effective stress."""

    result = np.asarray(effective_stress, dtype=float).reshape(6).copy()
    result[:3] += biot * float(pressure)
    return result
