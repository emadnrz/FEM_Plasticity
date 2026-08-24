"""Coupled 2-D axisymmetric undrained triaxial driver."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from plasticity import invariants

from .boundary_conditions import (
    affine_undrained_guess,
    outer_radial_tie,
    undrained_triaxial_displacements,
    uniform_pressure_tie,
)
from .hydraulics import HydraulicProperties
from .materials.common import MaterialCase
from .mesh import AxisymmetricMesh
from .postprocessing import (
    total_stress,
    volume_average_effective_stress,
    volume_average_pressure,
    volume_average_strain,
)
from .solver import initialize_gauss_states, solve_coupled_increment


Array = NDArray[np.float64]


@dataclass
class HMTriaxialHistory:
    data: dict[str, Array]

    def __getitem__(self, name: str) -> Array:
        return self.data[name]


def _initial_porosity(case: MaterialCase) -> float:
    state = case.initial_state
    specific_volume = float(state.V) if hasattr(state, "V") else 1.0 + float(state.e)
    return (specific_volume - 1.0) / specific_volume


def run_undrained_triaxial(
    mesh: AxisymmetricMesh,
    case: MaterialCase,
    axial_strain_end: float = 0.15,
    increments: int = 100,
    strain_rate: float = 1.0e-3,
    hydraulics: HydraulicProperties | None = None,
    tolerance: float | None = None,
) -> HMTriaxialHistory:
    """Run a closed-boundary saturated triaxial compression test."""

    if axial_strain_end <= 0.0 or increments < 1 or strain_rate <= 0.0:
        raise ValueError("positive strain, increments, and strain rate required")
    hydraulics = hydraulics or HydraulicProperties.saturated_soil(
        _initial_porosity(case)
    )
    if tolerance is None:
        # The explicit NorSand return map has small stress jumps at yield
        # corners; this equilibrium tolerance keeps those below about 1 kPa.
        tolerance = 1.0e-3 if case.name == "NorSand" else 1.0e-5
    states = initialize_gauss_states(mesh, case.initial_state)
    pressure = np.zeros(mesh.n_nodes, dtype=float)
    displacement = np.zeros(mesh.displacement_dofs, dtype=float)
    step = axial_strain_end / increments
    dt = step / strain_rate
    names = (
        "eps_axial",
        "eps_radial",
        "eps_v",
        "p_effective",
        "q",
        "eta",
        "pore_pressure",
        "mean_total_stress",
        "radial_total_stress",
        "axial_total_stress",
        "newton_iterations",
        "equilibrium_residual",
        "mass_residual",
    )
    data = {name: np.zeros(increments + 1, dtype=float) for name in names}
    initial_effective = volume_average_effective_stress(mesh, states)
    p0, q0, *_ = invariants(initial_effective)
    data["p_effective"][0] = p0
    data["q"][0] = q0
    data["eta"][0] = q0 / p0
    initial_total = total_stress(initial_effective, 0.0, hydraulics.biot_coefficient)
    data["mean_total_stress"][0] = float(np.mean(initial_total[:3]))
    data["radial_total_stress"][0] = 0.5 * float(initial_total[0] + initial_total[1])
    data["axial_total_stress"][0] = float(initial_total[2])

    for increment in range(1, increments + 1):
        displacement_guess = affine_undrained_guess(mesh, step)
        elastic_shear = float(case.model.elastic_matrix(states[0][0])[4, 4])
        pressure_guess = np.full(mesh.n_nodes, elastic_shear * step, dtype=float)
        result = solve_coupled_increment(
            mesh,
            case.model,
            states,
            pressure,
            hydraulics,
            undrained_triaxial_displacements(mesh, step),
            dt=dt,
            tolerance=tolerance,
            max_iterations=60,
            displacement_ties=outer_radial_tie(mesh),
            pressure_ties=uniform_pressure_tie(mesh),
            initial_displacement_guess=displacement_guess,
            initial_pressure_guess=pressure_guess,
            homogeneous_bracketing=case.name == "NorSand",
        )
        states = result.states
        pressure += result.pore_pressure_increment
        displacement += result.displacement_increment
        effective = volume_average_effective_stress(mesh, states)
        pore = volume_average_pressure(mesh, pressure)
        strain_tension = volume_average_strain(mesh, displacement)
        total = total_stress(effective, pore, hydraulics.biot_coefficient)
        p_eff, q, *_ = invariants(effective)

        data["eps_axial"][increment] = increment * step
        data["eps_radial"][increment] = -0.5 * float(
            strain_tension[0] + strain_tension[1]
        )
        data["eps_v"][increment] = -float(np.sum(strain_tension[:3]))
        data["p_effective"][increment] = p_eff
        data["q"][increment] = q
        data["eta"][increment] = q / p_eff
        data["pore_pressure"][increment] = pore
        data["mean_total_stress"][increment] = float(np.mean(total[:3]))
        data["radial_total_stress"][increment] = 0.5 * float(total[0] + total[1])
        data["axial_total_stress"][increment] = float(total[2])
        data["newton_iterations"][increment] = result.iterations
        data["equilibrium_residual"][increment] = result.momentum_residual_norm
        data["mass_residual"][increment] = result.mass_residual_norm
    return HMTriaxialHistory(data)
