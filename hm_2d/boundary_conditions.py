"""Boundary-condition helpers for axisymmetric laboratory specimens."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .mesh import AxisymmetricMesh


Array = NDArray[np.float64]


def undrained_triaxial_displacements(
    mesh: AxisymmetricMesh, axial_compression_increment: float
) -> dict[int, float]:
    """Axis, base, and moving-platen displacement increments."""

    if axial_compression_increment <= 0.0:
        raise ValueError("axial compression increment must be positive")
    coordinates = mesh.coordinates
    height = float(np.max(coordinates[:, 1]))
    tolerance = 1.0e-12 * max(float(np.max(coordinates)), 1.0)
    prescribed: dict[int, float] = {}
    for node, (radius, axial) in enumerate(coordinates):
        if abs(radius) <= tolerance:
            prescribed[2 * node] = 0.0
        if abs(axial) <= tolerance:
            prescribed[2 * node + 1] = 0.0
        if abs(axial - height) <= tolerance:
            prescribed[2 * node + 1] = -axial_compression_increment * height
    return prescribed


def outer_radial_tie(mesh: AxisymmetricMesh) -> tuple[tuple[int, ...], ...]:
    """Tie the membrane radial DOFs for a homogeneous triaxial patch."""

    coordinates = mesh.coordinates
    outer = float(np.max(coordinates[:, 0]))
    tolerance = 1.0e-12 * max(outer, 1.0)
    dofs = tuple(
        2 * node
        for node, (radius, _) in enumerate(coordinates)
        if abs(radius - outer) <= tolerance
    )
    return (dofs,) if len(dofs) > 1 else ()


def uniform_pressure_tie(mesh: AxisymmetricMesh) -> tuple[tuple[int, ...], ...]:
    """Tie nodal pressure increments for a homogeneous impermeable test."""

    nodes = tuple(range(mesh.n_nodes))
    return (nodes,) if len(nodes) > 1 else ()


def affine_undrained_guess(
    mesh: AxisymmetricMesh, axial_compression_increment: float
) -> Array:
    """Isochoric affine displacement predictor for Newton iteration."""

    guess = np.zeros(mesh.displacement_dofs, dtype=float)
    for node, (radius, axial) in enumerate(mesh.coordinates):
        guess[2 * node] = 0.5 * axial_compression_increment * radius
        guess[2 * node + 1] = -axial_compression_increment * axial
    return guess
