"""Nonlinear monolithic u-p solver for 2-D axisymmetric saturated soil."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .constitutive import integrate_axisymmetric, state_key
from .elements import (
    axisymmetric_kinematics,
    characteristic_length,
    displacement_dofs,
)
from .hydraulics import HydraulicProperties
from .mesh import AxisymmetricMesh
from .quadrature import QUAD4_GAUSS_POINTS


Array = NDArray[np.float64]
VOLUMETRIC_VECTOR = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])


@dataclass
class HMStepResult:
    """Converged increments and states from one backward-Euler HM step."""

    displacement_increment: Array
    pore_pressure_increment: Array
    states: list[list[Any]]
    displacement_reactions: Array
    fluid_residual: Array
    iterations: int
    residual_norm: float
    momentum_residual_norm: float
    mass_residual_norm: float


def initialize_gauss_states(
    mesh: AxisymmetricMesh, initial_state: Any
) -> list[list[Any]]:
    """Copy one constitutive state to every element Gauss point."""

    return [
        [initial_state.copy() for _ in QUAD4_GAUSS_POINTS]
        for _ in mesh.elements
    ]


def _constraint_transform(
    size: int,
    prescribed: set[int],
    tie_groups: tuple[tuple[int, ...], ...],
) -> Array:
    free = set(range(size)) - prescribed
    handled: set[int] = set()
    columns: list[list[int]] = []
    for group in tie_groups:
        dofs = list(map(int, group))
        if len(dofs) < 2 or not set(dofs) <= free:
            raise ValueError("tie groups must contain at least two free field DOFs")
        if handled.intersection(dofs):
            raise ValueError("tie groups may not overlap")
        handled.update(dofs)
        columns.append(dofs)
    columns.extend([[dof] for dof in sorted(free - handled)])
    transform = np.zeros((size, len(columns)), dtype=float)
    for column, dofs in enumerate(columns):
        transform[dofs, column] = 1.0
    return transform


def _specimen_volume(mesh: AxisymmetricMesh) -> float:
    radius = float(np.max(mesh.coordinates[:, 0]))
    height = float(np.max(mesh.coordinates[:, 1]) - np.min(mesh.coordinates[:, 1]))
    return pi * radius * radius * height


def solve_coupled_increment(
    mesh: AxisymmetricMesh,
    model: Any,
    committed_states: list[list[Any]],
    committed_pore_pressure: Array,
    hydraulics: HydraulicProperties,
    prescribed_displacement_increment: dict[int, float],
    prescribed_pressure_increment: dict[int, float] | None = None,
    displacement_force_increment: Array | None = None,
    fluid_source_increment: Array | None = None,
    dt: float = 1.0,
    tolerance: float = 1.0e-6,
    max_iterations: int = 25,
    displacement_ties: tuple[tuple[int, ...], ...] = (),
    pressure_ties: tuple[tuple[int, ...], ...] = (),
    initial_displacement_guess: Array | None = None,
    initial_pressure_guess: Array | None = None,
    homogeneous_bracketing: bool = False,
) -> HMStepResult:
    """Solve one nonlinear incremental momentum/mass-balance problem.

    Displacements use the conventional tension-positive FE sign, constitutive
    strains and effective stresses are compression-positive, and pore pressure
    is positive in compression.  Natural pressure boundaries are no-flow.
    """

    if dt <= 0.0:
        raise ValueError("dt must be positive")
    n_u, n_p = mesh.displacement_dofs, mesh.pressure_dofs
    old_pressure = np.asarray(committed_pore_pressure, dtype=float).reshape(n_p)
    pressure_bc = prescribed_pressure_increment or {}
    external_u = (
        np.zeros(n_u, dtype=float)
        if displacement_force_increment is None
        else np.asarray(displacement_force_increment, dtype=float).reshape(n_u)
    )
    source_p = (
        np.zeros(n_p, dtype=float)
        if fluid_source_increment is None
        else np.asarray(fluid_source_increment, dtype=float).reshape(n_p)
    )

    transform_u = _constraint_transform(
        n_u, set(prescribed_displacement_increment), displacement_ties
    )
    transform_p = _constraint_transform(n_p, set(pressure_bc), pressure_ties)
    du = (
        np.zeros(n_u, dtype=float)
        if initial_displacement_guess is None
        else np.asarray(initial_displacement_guess, dtype=float).reshape(n_u).copy()
    )
    dp = (
        np.zeros(n_p, dtype=float)
        if initial_pressure_guess is None
        else np.asarray(initial_pressure_guess, dtype=float).reshape(n_p).copy()
    )
    for dof, value in prescribed_displacement_increment.items():
        du[int(dof)] = float(value)
    for node, value in pressure_bc.items():
        dp[int(node)] = float(value)

    volume = _specimen_volume(mesh)
    initial_stress_scale = max(
        float(
            np.mean(
                [
                    np.linalg.norm(state.stress[:3]) / np.sqrt(3.0)
                    for element_states in committed_states
                    for state in element_states
                ]
            )
        ),
        1.0,
    )
    radius = float(np.max(mesh.coordinates[:, 0]))
    height = float(np.max(mesh.coordinates[:, 1]) - np.min(mesh.coordinates[:, 1]))
    force_scale = initial_stress_scale * 2.0 * pi * radius * height
    increment_strain_scale = max(
        max((abs(value) for value in prescribed_displacement_increment.values()), default=0.0)
        / max(height, 1.0e-12),
        1.0e-9,
    )
    mass_scale = volume * increment_strain_scale

    trial_states: list[list[Any]] = []
    internal_u = np.zeros(n_u)
    mass = np.zeros(n_p)
    momentum_norm = mass_norm = residual_norm = float("inf")
    scalar_samples: list[tuple[float, float]] = []
    bracket_mode = (
        homogeneous_bracketing
        and transform_u.shape[1] == 1
        and transform_p.shape[1] == 1
    )

    for iteration in range(1, max_iterations + 1):
        K_uu = np.zeros((n_u, n_u), dtype=float)
        K_up = np.zeros((n_u, n_p), dtype=float)
        K_pu = np.zeros((n_p, n_u), dtype=float)
        K_pp = np.zeros((n_p, n_p), dtype=float)
        internal_u = np.zeros(n_u, dtype=float)
        mass = np.zeros(n_p, dtype=float)
        trial_states = []
        cache: dict[tuple[Any, ...], Any] = {}

        for element_index, element in enumerate(mesh.elements):
            u_dofs = displacement_dofs(element)
            p_nodes = np.asarray(element.node_ids, dtype=int)
            coordinates = mesh.element_coordinates(element)
            u_element = du[u_dofs]
            dp_element = dp[p_nodes]
            p_new_element = old_pressure[p_nodes] + dp_element
            h_element = characteristic_length(coordinates)

            f_u = np.zeros(8, dtype=float)
            f_p = np.zeros(4, dtype=float)
            k_uu = np.zeros((8, 8), dtype=float)
            k_up = np.zeros((8, 4), dtype=float)
            k_pu = np.zeros((4, 8), dtype=float)
            k_pp = np.zeros((4, 4), dtype=float)
            element_states: list[Any] = []

            for point_index, point in enumerate(QUAD4_GAUSS_POINTS):
                kin = axisymmetric_kinematics(
                    coordinates, point.coordinates, point.weight
                )
                tension_strain = kin.strain_displacement @ u_element
                soil_strain = -tension_strain
                old_state = committed_states[element_index][point_index]
                key = (
                    state_key(old_state),
                    np.round(soil_strain, 12).tobytes(),
                    float(dt),
                )
                response = cache.get(key)
                if response is None:
                    response = integrate_axisymmetric(
                        model,
                        old_state,
                        soil_strain,
                        dt,
                        compute_tangent=not bracket_mode,
                    )
                    cache[key] = response

                N = kin.shape
                B = kin.strain_displacement
                grad = kin.gradients
                weight = kin.integration_weight
                pore_increment = float(N @ dp_element)
                pore_pressure = float(N @ p_new_element)
                effective_tension_increment = -(response.stress - old_state.stress)
                total_tension_increment = effective_tension_increment - (
                    hydraulics.biot_coefficient
                    * pore_increment
                    * VOLUMETRIC_VECTOR
                )
                f_u += B.T @ total_tension_increment * weight
                k_uu += B.T @ response.tangent @ B * weight
                k_up += -hydraulics.biot_coefficient * np.outer(
                    B.T @ VOLUMETRIC_VECTOR, N
                ) * weight

                volumetric_tension_increment = float(
                    VOLUMETRIC_VECTOR @ tension_strain
                )
                fluid_content_increment = (
                    hydraulics.biot_coefficient * volumetric_tension_increment
                    + hydraulics.storage * pore_increment
                )
                f_p += N * fluid_content_increment * weight
                k_pu += hydraulics.biot_coefficient * np.outer(
                    N, VOLUMETRIC_VECTOR @ B
                ) * weight
                k_pp += hydraulics.storage * np.outer(N, N) * weight

                conductivity = hydraulics.mobility * (grad @ grad.T) * weight
                f_p += dt * conductivity @ p_new_element
                k_pp += dt * conductivity

                elastic_shear = max(
                    float(model.elastic_matrix(old_state)[4, 4]), 1.0e-9
                )
                tau = (
                    hydraulics.stabilization_factor
                    * h_element**2
                    / elastic_shear
                )
                stabilization = tau * (grad @ grad.T) * weight
                f_p += stabilization @ dp_element
                k_pp += stabilization
                element_states.append(response.state.copy())

            internal_u[u_dofs] += f_u
            mass[p_nodes] += f_p
            K_uu[np.ix_(u_dofs, u_dofs)] += k_uu
            K_up[np.ix_(u_dofs, p_nodes)] += k_up
            K_pu[np.ix_(p_nodes, u_dofs)] += k_pu
            K_pp[np.ix_(p_nodes, p_nodes)] += k_pp
            trial_states.append(element_states)

        residual_u = external_u - internal_u
        residual_p = source_p - mass
        reduced_u = transform_u.T @ residual_u
        reduced_p = transform_p.T @ residual_p
        momentum_norm = float(np.linalg.norm(reduced_u)) / force_scale
        mass_norm = float(np.linalg.norm(reduced_p)) / mass_scale
        residual_norm = max(momentum_norm, mass_norm)
        if residual_norm <= tolerance:
            return HMStepResult(
                du,
                dp,
                trial_states,
                internal_u - external_u,
                mass - source_p,
                iteration,
                residual_norm,
                momentum_norm,
                mass_norm,
            )

        Kuu_reduced = transform_u.T @ K_uu @ transform_u
        Kup_reduced = transform_u.T @ K_up @ transform_p
        Kpu_reduced = transform_p.T @ K_pu @ transform_u
        Kpp_reduced = transform_p.T @ K_pp @ transform_p

        if bracket_mode:
            # A homogeneous triaxial patch has one membrane-displacement and
            # one uniform-pressure unknown.  Constitutive yield corners can
            # make a local Newton tangent jump.  Bracket the physical radial
            # displacement while statically enforcing the (linear) mass
            # equation; this is much more robust and is still the exact u-p
            # residual of the general assembler.
            pressure_denominator = float(Kpp_reduced[0, 0])
            if abs(pressure_denominator) <= 1.0e-20:
                raise RuntimeError(
                    "uniform-pressure triaxial solve needs positive fluid storage"
                )
            # Do not record a momentum sample until its pressure satisfies
            # mass balance.  With nearly incompressible water, even a tiny
            # trial volume error otherwise produces an irrelevant pressure.
            if mass_norm > tolerance:
                dp += transform_p[:, 0] * (
                    float(reduced_p[0]) / pressure_denominator
                )
                continue

            basis_u = transform_u[:, 0]
            scalar = float(basis_u @ du / (basis_u @ basis_u))
            radial_residual = float(reduced_u[0])
            scalar_samples.append((scalar, radial_residual))
            axial_increment = max(
                max(
                    abs(value)
                    for value in prescribed_displacement_increment.values()
                )
                / max(height, 1.0e-12),
                1.0e-12,
            )
            scalar_limit = axial_increment * radius
            ordered = sorted(scalar_samples)
            brackets = [
                (left, right)
                for left, right in zip(ordered[:-1], ordered[1:])
                if left[1] * right[1] < 0.0
            ]
            if brackets:
                left, right = min(
                    brackets, key=lambda pair: pair[1][0] - pair[0][0]
                )
                denominator = right[1] - left[1]
                new_scalar = left[0] - left[1] * (
                    right[0] - left[0]
                ) / denominator
                width = right[0] - left[0]
                new_scalar = float(
                    np.clip(
                        new_scalar,
                        left[0] + 0.1 * width,
                        right[0] - 0.1 * width,
                    )
                )
            elif not any(abs(sample[0]) < 1.0e-14 for sample in ordered):
                new_scalar = 0.0
            elif not any(
                abs(sample[0] - scalar_limit) < 1.0e-14 for sample in ordered
            ):
                new_scalar = scalar_limit
            else:
                best = sorted(ordered, key=lambda sample: abs(sample[1]))[:2]
                denominator = best[1][1] - best[0][1]
                if abs(denominator) <= 1.0e-14 * max(abs(best[0][1]), 1.0):
                    new_scalar = 0.5 * scalar_limit
                else:
                    new_scalar = best[0][0] - best[0][1] * (
                        best[1][0] - best[0][0]
                    ) / denominator
            if any(abs(new_scalar - sample[0]) < 1.0e-13 for sample in ordered):
                if brackets:
                    new_scalar = 0.5 * (left[0] + right[0])
                else:
                    new_scalar += np.copysign(0.02 * scalar_limit, radial_residual)
            new_scalar = float(np.clip(new_scalar, 0.0, scalar_limit))
            delta_u = new_scalar - scalar
            delta_p = (
                float(reduced_p[0]) - float(Kpu_reduced[0, 0]) * delta_u
            ) / pressure_denominator
            du += basis_u * delta_u
            dp += transform_p[:, 0] * delta_p
            continue

        top = np.hstack((reduced_u, reduced_p))
        jacobian = np.block(
            [
                [
                    Kuu_reduced,
                    Kup_reduced,
                ],
                [
                    Kpu_reduced,
                    Kpp_reduced,
                ],
            ]
        )
        try:
            correction = np.linalg.solve(jacobian, top)
        except np.linalg.LinAlgError as error:
            raise RuntimeError("singular coupled u-p Newton matrix") from error
        split = transform_u.shape[1]
        du += transform_u @ correction[:split]
        dp += transform_p @ correction[split:]

    raise RuntimeError(
        "coupled u-p Newton solver failed after "
        f"{max_iterations} iterations; momentum={momentum_norm:.3e}, "
        f"mass={mass_norm:.3e}"
    )
