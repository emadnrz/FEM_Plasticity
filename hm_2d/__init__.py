"""Modular two-dimensional axisymmetric hydro-mechanical finite elements.

The package uses displacement--pore-pressure (u-p) unknowns.  Soil models
remain effective-stress constitutive laws and are connected to the coupled
solver through the common interface in :mod:`plasticity`.
"""

from .hydraulics import HydraulicProperties
from .mesh import AxisymmetricMesh, Node, Quad4Element, structured_cylinder_mesh
from .solver import HMStepResult, initialize_gauss_states, solve_coupled_increment

__all__ = [
    "AxisymmetricMesh",
    "HMStepResult",
    "HydraulicProperties",
    "Node",
    "Quad4Element",
    "initialize_gauss_states",
    "solve_coupled_increment",
    "structured_cylinder_mesh",
]
