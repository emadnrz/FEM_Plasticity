"""Node, element, and structured axisymmetric mesh creation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


Array = NDArray[np.float64]


@dataclass(frozen=True)
class Node:
    """A node in the radius--axial (r-z) meridian plane."""

    id: int
    radius: float
    axial: float

    @property
    def coordinates(self) -> tuple[float, float]:
        return self.radius, self.axial


@dataclass(frozen=True)
class Quad4Element:
    """Four-node bilinear quadrilateral connectivity."""

    id: int
    node_ids: tuple[int, int, int, int]


@dataclass(frozen=True)
class AxisymmetricMesh:
    """Two-dimensional meridian mesh representing a body of revolution."""

    nodes: tuple[Node, ...]
    elements: tuple[Quad4Element, ...]

    @property
    def coordinates(self) -> Array:
        return np.asarray([node.coordinates for node in self.nodes], dtype=float)

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def displacement_dofs(self) -> int:
        return 2 * self.n_nodes

    @property
    def pressure_dofs(self) -> int:
        return self.n_nodes

    @property
    def total_dofs(self) -> int:
        return self.displacement_dofs + self.pressure_dofs

    def element_coordinates(self, element: Quad4Element) -> Array:
        return self.coordinates[np.asarray(element.node_ids, dtype=int)]


def structured_cylinder_mesh(
    radius: float = 1.0,
    height: float = 2.0,
    radial_elements: int = 1,
    axial_elements: int = 1,
) -> AxisymmetricMesh:
    """Create a structured Quad4 r-z mesh of a cylindrical specimen."""

    if radius <= 0.0 or height <= 0.0:
        raise ValueError("radius and height must be positive")
    if radial_elements < 1 or axial_elements < 1:
        raise ValueError("at least one element is required in each direction")

    radial = np.linspace(0.0, radius, radial_elements + 1)
    axial = np.linspace(0.0, height, axial_elements + 1)
    nodes = tuple(
        Node(iz * (radial_elements + 1) + ir, float(r), float(z))
        for iz, z in enumerate(axial)
        for ir, r in enumerate(radial)
    )

    def node_id(ir: int, iz: int) -> int:
        return iz * (radial_elements + 1) + ir

    elements: list[Quad4Element] = []
    for iz in range(axial_elements):
        for ir in range(radial_elements):
            elements.append(
                Quad4Element(
                    len(elements),
                    (
                        node_id(ir, iz),
                        node_id(ir + 1, iz),
                        node_id(ir + 1, iz + 1),
                        node_id(ir, iz + 1),
                    ),
                )
            )
    return AxisymmetricMesh(nodes, tuple(elements))
