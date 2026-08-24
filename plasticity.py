"""Common small-strain constitutive interface for mechanical finite elements.

All soil models use compression-positive effective stress and normal strain.
Voigt vectors are ``[xx, yy, zz, xy, xz, yz]``; shear strain components are
engineering shear strains.  The finite-element adapter handles the sign change
to the conventional tension-positive displacement formulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import asin, sqrt
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray


Array = NDArray[np.float64]


@dataclass
class MaterialResponse:
    """Result of one constitutive strain increment."""

    state: Any
    stress: Array
    tangent: Array
    yielded: bool
    substeps: int = 1


class MechanicalMaterial(Protocol):
    """Interface required by the material-point and finite-element solvers."""

    def integrate_raw(self, state: Any, deps: Array, dt: float) -> MaterialResponse:
        ...


def stress_tensor(stress: Array) -> Array:
    """Convert six physical stress components to a symmetric tensor."""

    s = np.asarray(stress, dtype=float).reshape(6)
    return np.array(
        [[s[0], s[3], s[4]], [s[3], s[1], s[5]], [s[4], s[5], s[2]]],
        dtype=float,
    )


def invariants(stress: Array) -> tuple[float, float, float, Array, Array]:
    """Return ``p, q, Lode angle, dp/dstress, dq/dstress``.

    The Lode angle is in ``[-pi/6, pi/6]`` and equals ``pi/6`` in triaxial
    compression under the compression-positive convention.
    """

    stress = np.asarray(stress, dtype=float).reshape(6)
    p = float(np.mean(stress[:3]))
    dev = stress.copy()
    dev[:3] -= p
    j2 = 0.5 * (
        float(np.dot(dev[:3], dev[:3]))
        + 2.0 * float(np.dot(dev[3:], dev[3:]))
    )
    q = sqrt(max(3.0 * j2, 0.0))
    dp = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]) / 3.0
    if q > 1.0e-14 * max(abs(p), 1.0):
        dq = 3.0 / (2.0 * q) * np.array(
            [dev[0], dev[1], dev[2], 2.0 * dev[3], 2.0 * dev[4], 2.0 * dev[5]]
        )
        j3 = float(np.linalg.det(stress_tensor(dev)))
        sin3theta = 3.0 * sqrt(3.0) * j3 / (2.0 * max(j2, 1.0e-30) ** 1.5)
        theta = asin(float(np.clip(sin3theta, -1.0, 1.0))) / 3.0
    else:
        dq = np.zeros(6)
        theta = 0.0
    return p, q, theta, dp, dq


def strain_invariants(strain: Array) -> tuple[float, float]:
    """Return compression-positive volumetric and equivalent shear strain."""

    strain = np.asarray(strain, dtype=float).reshape(6)
    ev = float(np.sum(strain[:3]))
    dev = strain.copy()
    dev[:3] -= ev / 3.0
    # Engineering shear components contribute gamma^2/2 to e:e.
    norm2 = float(np.dot(dev[:3], dev[:3])) + 0.5 * float(np.dot(dev[3:], dev[3:]))
    eq = sqrt(max(2.0 * norm2 / 3.0, 0.0))
    return ev, eq


def isotropic_stiffness(K: float, G: float) -> Array:
    """Six-by-six isotropic elastic matrix for engineering shear strain."""

    lame = K - 2.0 * G / 3.0
    return np.array(
        [
            [lame + 2 * G, lame, lame, 0, 0, 0],
            [lame, lame + 2 * G, lame, 0, 0, 0],
            [lame, lame, lame + 2 * G, 0, 0, 0],
            [0, 0, 0, G, 0, 0],
            [0, 0, 0, 0, G, 0],
            [0, 0, 0, 0, 0, G],
        ],
        dtype=float,
    )


def integrate_material(
    model: MechanicalMaterial,
    state: Any,
    deps: Array,
    dt: float = 0.0,
    compute_tangent: bool = True,
) -> MaterialResponse:
    """General constitutive entry point used at every FE integration point.

    Models supply only their stress-integration algorithm.  This function
    optionally constructs a model-independent numerical algorithmic tangent,
    so the mechanical Newton solver does not contain model-specific branches.
    """

    deps = np.asarray(deps, dtype=float).reshape(6)
    base = model.integrate_raw(state, deps, dt)
    if not compute_tangent:
        return base

    tangent = np.zeros((6, 6))
    strain_scale = max(float(np.linalg.norm(deps)), 1.0e-5)
    h = max(1.0e-9, 1.0e-5 * strain_scale)
    for j in range(6):
        perturbed = deps.copy()
        perturbed[j] += h
        plus = model.integrate_raw(state, perturbed, dt)
        tangent[:, j] = (plus.stress - base.stress) / h
    base.tangent = tangent
    return base

