"""Nonlinear 2-D axisymmetric small-strain mechanical finite elements.

The solver uses four-node quadrilaterals with 2x2 Gauss integration. Each node
has radial and axial displacement degrees of freedom. The constitutive routine
remains fully three-dimensional: the axisymmetric strain vector is embedded as
``[rr, hoop, zz, r-hoop, rz, hoop-z]`` before being passed to the common
material interface. This is the correct 2-D representation of a triaxial
cylinder.

The analysis is effective-stress, mechanical, and drained. There is no pore
pressure degree of freedom or consolidation equation in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import pi
from typing import Any, Iterable

import numpy as np

from plasticity import Array, MechanicalMaterial


NATURAL_COORDS = np.array(
    [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]], dtype=float
)
GAUSS_POINTS = tuple(product((-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)), repeat=2))


@dataclass(frozen=True)
class Quad4:
    nodes: tuple[int, int, int, int]


@dataclass
class Mesh:
    """Axisymmetric mesh with nodal coordinates ``[radius, axial]``."""

    coordinates: Array
    elements: list[Quad4]

    @property
    def ndof(self) -> int:
        return 2 * len(self.coordinates)


@dataclass
class FEStepResult:
    displacement_increment: Array
    states: list[list[Any]]
    reactions: Array
    iterations: int
    residual_norm: float


def axisymmetric_specimen_mesh(
    radius: float = 1.0,
    height: float = 2.0,
    radial_elements: int = 1,
    axial_elements: int = 1,
) -> Mesh:
    """Create a structured r-z mesh of a cylindrical triaxial specimen."""

    if radius <= 0.0 or height <= 0.0 or radial_elements < 1 or axial_elements < 1:
        raise ValueError("positive dimensions and element counts are required")
    r = np.linspace(0.0, radius, radial_elements + 1)
    z = np.linspace(0.0, height, axial_elements + 1)
    coordinates = np.array([[rv, zv] for zv in z for rv in r], dtype=float)

    def node(ir: int, iz: int) -> int:
        return iz * (radial_elements + 1) + ir

    elements = []
    for iz in range(axial_elements):
        for ir in range(radial_elements):
            elements.append(
                Quad4(
                    (
                        node(ir, iz),
                        node(ir + 1, iz),
                        node(ir + 1, iz + 1),
                        node(ir, iz + 1),
                    )
                )
            )
    return Mesh(coordinates, elements)


def _shape_functions(natural: Array) -> tuple[Array, Array]:
    xi, eta = map(float, natural)
    N = 0.25 * np.array(
        [
            (1.0 - xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 + eta),
            (1.0 - xi) * (1.0 + eta),
        ]
    )
    gradients = 0.25 * np.array(
        [
            [-(1.0 - eta), -(1.0 - xi)],
            [1.0 - eta, -(1.0 + xi)],
            [1.0 + eta, 1.0 + xi],
            [-(1.0 + eta), 1.0 - xi],
        ]
    )
    return N, gradients


def _B_matrix(coords: Array, natural: Array) -> tuple[Array, float, float]:
    """Return the 6x8 axisymmetric B matrix, det(J), and Gauss radius."""

    N, grad_nat = _shape_functions(natural)
    jacobian = coords.T @ grad_nat
    detJ = float(np.linalg.det(jacobian))
    if detJ <= 0.0:
        raise ValueError("Quad4 element has nonpositive Jacobian")
    grad = grad_nat @ np.linalg.inv(jacobian)
    radius = float(N @ coords[:, 0])
    if radius <= 0.0:
        raise ValueError("axisymmetric integration point has nonpositive radius")

    # Material ordering: [rr, hoop, zz, r-hoop, rz, hoop-z].
    B = np.zeros((6, 8))
    for a, (dr, dz) in enumerate(grad):
        j = 2 * a
        B[0, j] = dr
        B[1, j] = N[a] / radius
        B[2, j + 1] = dz
        B[4, j] = dz
        B[4, j + 1] = dr
    return B, detJ, radius


def initialize_integration_states(mesh: Mesh, state0: Any) -> list[list[Any]]:
    return [[state0.copy() for _ in GAUSS_POINTS] for _ in mesh.elements]


def _element_dofs(element: Quad4) -> Array:
    return np.array([2 * node + dof for node in element.nodes for dof in range(2)], dtype=int)


def _state_key(state: Any) -> tuple[Any, ...]:
    """Hash a dataclass-like state for within-assembly material-call reuse."""

    values: list[Any] = []
    for name, value in vars(state).items():
        if isinstance(value, np.ndarray):
            values.append((name, value.shape, np.round(value, 12).tobytes()))
        else:
            values.append((name, float(value)))
    return tuple(values)


def _integrate_axisymmetric(
    model, old, deps_soil: Array, dt: float, compute_tangent: bool = True
):
    """Material update plus the four tangent columns used by axisymmetry."""

    with np.errstate(over="ignore", invalid="ignore"):
        base = model.integrate_raw(old, deps_soil, dt)
    tangent = np.zeros((6, 6))
    if not compute_tangent:
        base.tangent = tangent
        return base
    strain_scale = max(float(np.linalg.norm(deps_soil)), 1.0e-5)
    h = max(1.0e-9, 1.0e-5 * strain_scale)
    for column in (0, 1, 2, 4):
        perturbed = deps_soil.copy()
        perturbed[column] += h
        with np.errstate(over="ignore", invalid="ignore"):
            plus = model.integrate_raw(old, perturbed, dt)
        tangent[:, column] = (plus.stress - base.stress) / h
    base.tangent = tangent
    return base


def solve_increment(
    mesh: Mesh,
    model: MechanicalMaterial,
    committed_states: list[list[Any]],
    prescribed_increment: dict[int, float],
    force_increment: Array | None = None,
    dt: float = 0.0,
    tolerance: float = 1.0e-7,
    max_iterations: int = 20,
    tie_groups: tuple[tuple[int, ...], ...] = (),
) -> FEStepResult:
    """Solve one axisymmetric displacement/load increment by Newton iteration."""

    ndof = mesh.ndof
    df_ext = np.zeros(ndof) if force_increment is None else np.asarray(force_increment, dtype=float)
    if df_ext.shape != (ndof,):
        raise ValueError("force_increment has incorrect size")
    prescribed = np.array(sorted(prescribed_increment), dtype=int)
    free = np.setdiff1d(np.arange(ndof), prescribed)
    handled: set[int] = set()
    columns: list[list[int]] = []
    free_set = set(map(int, free))
    for group in tie_groups:
        group_list = list(map(int, group))
        if len(group_list) < 2 or not set(group_list) <= free_set:
            raise ValueError("each tie group must contain at least two free DOFs")
        if handled.intersection(group_list):
            raise ValueError("tie groups may not overlap")
        handled.update(group_list)
        columns.append(group_list)
    columns.extend([[int(dof)] for dof in free if int(dof) not in handled])
    transform = np.zeros((ndof, len(columns)))
    for column, dofs in enumerate(columns):
        transform[dofs, column] = 1.0
    scalar_mode = len(columns) == 1
    du = np.zeros(ndof)
    for dof, value in prescribed_increment.items():
        du[dof] = value
    if tie_groups:
        height = float(np.max(mesh.coordinates[:, 1]) - np.min(mesh.coordinates[:, 1]))
        radius = float(np.max(mesh.coordinates[:, 0]))
        axial_strain_increment = max(
            max(abs(value) for value in prescribed_increment.values()) / height,
            1.0e-8,
        )
        radial_predictor = float(getattr(model.p, "nu", 0.2)) * axial_strain_increment * radius
        for dof in tie_groups[0]:
            du[dof] = radial_predictor

    trial_states: list[list[Any]] = []
    residual = np.zeros(ndof)
    scalar_samples: list[tuple[float, float]] = []
    for iteration in range(1, max_iterations + 1):
        stiffness = np.zeros((ndof, ndof))
        df_internal = np.zeros(ndof)
        trial_states = []
        material_cache: dict[tuple[Any, ...], Any] = {}
        for eidx, element in enumerate(mesh.elements):
            edofs = _element_dofs(element)
            ue = du[edofs]
            coords = mesh.coordinates[np.array(element.nodes)]
            element_stiffness = np.zeros((8, 8))
            element_force = np.zeros(8)
            element_states: list[Any] = []
            for gidx, natural in enumerate(GAUSS_POINTS):
                B, detJ, radius = _B_matrix(coords, np.asarray(natural))
                # FE strain is tension-positive; soil strain is compression-positive.
                deps_soil = -(B @ ue)
                old = committed_states[eidx][gidx]
                key = (
                    _state_key(old),
                    np.round(deps_soil, 12).tobytes(),
                    float(dt),
                )
                response = material_cache.get(key)
                if response is None:
                    response = _integrate_axisymmetric(
                        model, old, deps_soil, dt, compute_tangent=not scalar_mode
                    )
                    material_cache[key] = response
                weight = 2.0 * pi * radius * detJ
                ds_tension = -(response.stress - old.stress)
                element_force += B.T @ ds_tension * weight
                element_stiffness += B.T @ response.tangent @ B * weight
                element_states.append(response.state.copy())
            df_internal[edofs] += element_force
            stiffness[np.ix_(edofs, edofs)] += element_stiffness
            trial_states.append(element_states)

        residual = df_ext - df_internal
        reduced_residual = transform.T @ residual
        reduced_external = transform.T @ df_ext
        reduced_internal = transform.T @ df_internal
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
        specimen_radius = float(np.max(mesh.coordinates[:, 0]))
        specimen_height = float(
            np.max(mesh.coordinates[:, 1]) - np.min(mesh.coordinates[:, 1])
        )
        reference_force = initial_stress_scale * 2.0 * pi * specimen_radius * specimen_height
        scale = max(
            float(np.linalg.norm(reduced_external)),
            float(np.linalg.norm(reduced_internal)),
            reference_force,
        )
        residual_norm = float(np.linalg.norm(reduced_residual)) / scale
        if residual_norm <= tolerance:
            return FEStepResult(
                du, trial_states, df_internal - df_ext, iteration, residual_norm
            )
        reduced_stiffness = transform.T @ stiffness @ transform
        if scalar_mode:
            basis = transform[:, 0]
            scalar = float(basis @ du / (basis @ basis))
            scalar_residual = float(reduced_residual[0])
            height = float(
                np.max(mesh.coordinates[:, 1]) - np.min(mesh.coordinates[:, 1])
            )
            radius = float(np.max(mesh.coordinates[:, 0]))
            axial_strain_increment = max(
                max(abs(value) for value in prescribed_increment.values()) / height,
                1.0e-8,
            )
            displacement_scale = axial_strain_increment * radius
            scalar_samples.append((scalar, scalar_residual))
            ordered = sorted(scalar_samples)
            brackets = [
                (left, right)
                for left, right in zip(ordered[:-1], ordered[1:])
                if left[1] * right[1] < 0.0
            ]
            if brackets:
                left, right = min(brackets, key=lambda pair: pair[1][0] - pair[0][0])
                denominator = right[1] - left[1]
                new_scalar = left[0] - left[1] * (right[0] - left[0]) / denominator
                width = right[0] - left[0]
                # Keep the false-position update away from a stuck endpoint.
                new_scalar = float(
                    np.clip(new_scalar, left[0] + 0.1 * width, right[0] - 0.1 * width)
                )
            elif not any(abs(value[0]) < 1.0e-14 for value in ordered):
                new_scalar = 0.0
            elif not any(abs(value[0] - displacement_scale) < 1.0e-14 for value in ordered):
                new_scalar = displacement_scale
            else:
                best = sorted(ordered, key=lambda value: abs(value[1]))[:2]
                denominator = best[1][1] - best[0][1]
                if abs(denominator) <= 1.0e-14 * max(abs(best[0][1]), 1.0):
                    raise RuntimeError("could not bracket the drained radial-stress condition")
                new_scalar = best[0][0] - best[0][1] * (
                    best[1][0] - best[0][0]
                ) / denominator

            if any(abs(new_scalar - value[0]) < 1.0e-13 for value in ordered):
                if brackets:
                    new_scalar = 0.5 * (left[0] + right[0])
                else:
                    new_scalar = float(
                        np.clip(
                            new_scalar + np.copysign(0.05 * displacement_scale, scalar_residual),
                            0.0,
                            displacement_scale,
                        )
                    )

            new_scalar = float(
                np.clip(new_scalar, 0.0, displacement_scale)
            )
            du += basis * (new_scalar - scalar)
        else:
            correction = np.linalg.solve(reduced_stiffness, reduced_residual)
            du += transform @ correction

    raise RuntimeError(
        f"axisymmetric Newton solver failed after {max_iterations} iterations; "
        f"relative residual={residual_norm:.3e}; "
        f"scalar samples={scalar_samples[-6:] if scalar_mode else 'n/a'}"
    )


def triaxial_displacement_bcs(
    mesh: Mesh, axial_compression_increment: float
) -> dict[int, float]:
    """Axis, base, and platen constraints for displacement-controlled TXD."""

    if axial_compression_increment <= 0.0:
        raise ValueError("positive axial compression increment required")
    bc: dict[int, float] = {}
    tolerance = 1.0e-12 * max(float(np.max(mesh.coordinates)), 1.0)
    top = float(np.max(mesh.coordinates[:, 1]))
    for node, (radius, axial) in enumerate(mesh.coordinates):
        if abs(radius) <= tolerance:
            bc[2 * node] = 0.0
        if abs(axial) <= tolerance:
            bc[2 * node + 1] = 0.0
        if abs(axial - top) <= tolerance:
            bc[2 * node + 1] = -axial_compression_increment * top
    return bc


def triaxial_tie_groups(mesh: Mesh) -> tuple[tuple[int, ...], ...]:
    """Tie outer-radius radial DOFs to a rigid cylindrical membrane."""

    outer = float(np.max(mesh.coordinates[:, 0]))
    tolerance = 1.0e-12 * max(outer, 1.0)
    radial_dofs = tuple(
        2 * node
        for node, (radius, _) in enumerate(mesh.coordinates)
        if abs(radius - outer) <= tolerance
    )
    return (radial_dofs,) if len(radial_dofs) > 1 else ()


def average_stress(states: Iterable[Any]) -> Array:
    values = [state.stress for state in states]
    return np.mean(np.asarray(values), axis=0)


def flatten_states(states: list[list[Any]]) -> list[Any]:
    return [state for element_states in states for state in element_states]
