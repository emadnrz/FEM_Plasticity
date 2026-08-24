"""Independent numerical checks for interpolation, integration, and u-p coupling."""

from __future__ import annotations

from math import pi

import numpy as np

from modified_cam_clay import MCCParameters, ModifiedCamClay
from plasticity import invariants

from .boundary_conditions import (
    affine_undrained_guess,
    outer_radial_tie,
    undrained_triaxial_displacements,
    uniform_pressure_tie,
)
from .elements import axisymmetric_kinematics
from .hydraulics import HydraulicProperties
from .mesh import structured_cylinder_mesh
from .postprocessing import volume_average_effective_stress, volume_average_pressure
from .quadrature import QUAD4_GAUSS_POINTS
from .shape_functions import quad4_shape
from .solver import initialize_gauss_states, solve_coupled_increment


def verify_shape_and_axisymmetric_volume() -> dict[str, float]:
    """Check partition of unity, zero gradient sum, and cylinder volume."""

    mesh = structured_cylinder_mesh(
        radius=1.7, height=3.2, radial_elements=3, axial_elements=4
    )
    partition_error = 0.0
    gradient_error = 0.0
    integrated_volume = 0.0
    for element in mesh.elements:
        coordinates = mesh.element_coordinates(element)
        for point in QUAD4_GAUSS_POINTS:
            shape, gradients = quad4_shape(point.coordinates)
            partition_error = max(partition_error, abs(float(np.sum(shape)) - 1.0))
            gradient_error = max(
                gradient_error, float(np.linalg.norm(np.sum(gradients, axis=0)))
            )
            kin = axisymmetric_kinematics(
                coordinates, point.coordinates, point.weight
            )
            integrated_volume += kin.integration_weight
    exact_volume = pi * 1.7**2 * 3.2
    return {
        "partition_of_unity_error": partition_error,
        "natural_gradient_sum_error": gradient_error,
        "axisymmetric_volume_relative_error": abs(integrated_volume - exact_volume)
        / exact_volume,
    }


def verify_elastic_undrained_patch() -> dict[str, float]:
    """Compare one u-p element with a closed-form poroelastic increment."""

    mesh = structured_cylinder_mesh(radius=1.0, height=2.0)
    model = ModifiedCamClay(
        MCCParameters(max_strain_step=1.0e-4, p_ref=100.0)
    )
    state = model.initialize_isotropic(100.0, 1000.0)
    states = initialize_gauss_states(mesh, state)
    storage = 2.0e-7
    hydraulics = HydraulicProperties(
        biot_coefficient=1.0,
        storage=storage,
        mobility=0.0,
        stabilization_factor=0.05,
    )
    axial_increment = 1.0e-6
    shear = float(model.elastic_matrix(state)[4, 4])
    bulk = float(
        (
            model.elastic_matrix(state)[0, 0]
            + 2.0 * model.elastic_matrix(state)[0, 1]
        )
        / 3.0
    )
    result = solve_coupled_increment(
        mesh,
        model,
        states,
        np.zeros(mesh.n_nodes),
        hydraulics,
        undrained_triaxial_displacements(mesh, axial_increment),
        dt=1.0,
        tolerance=1.0e-10,
        displacement_ties=outer_radial_tie(mesh),
        pressure_ties=uniform_pressure_tie(mesh),
        initial_displacement_guess=affine_undrained_guess(mesh, axial_increment),
    )
    stress = volume_average_effective_stress(mesh, result.states)
    pressure = volume_average_pressure(mesh, result.pore_pressure_increment)
    _, q_fe, *_ = invariants(stress)

    volumetric_compression = shear * axial_increment / (
        bulk + shear / 3.0 + 1.0 / storage
    )
    pressure_exact = volumetric_compression / storage
    q_exact = shear * (3.0 * axial_increment - volumetric_compression)
    radial_total = 0.5 * float(stress[0] + stress[1]) + pressure
    return {
        "pore_pressure_relative_error": abs(pressure - pressure_exact)
        / pressure_exact,
        "deviator_stress_relative_error": abs(q_fe - q_exact) / q_exact,
        "radial_total_stress_error_kpa": abs(radial_total - 100.0),
        "mass_residual_norm": result.mass_residual_norm,
        "momentum_residual_norm": result.momentum_residual_norm,
    }


def run_basic_verifications() -> dict[str, dict[str, float]]:
    return {
        "interpolation_and_quadrature": verify_shape_and_axisymmetric_volume(),
        "elastic_undrained_patch": verify_elastic_undrained_patch(),
    }
