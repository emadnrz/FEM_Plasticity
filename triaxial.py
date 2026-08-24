"""Model-independent material-point triaxial drivers.

The drained driver solves the lateral-strain increment required to keep radial
effective stress constant.  The undrained driver imposes zero total volumetric
strain; excess pore pressure is then recovered from constant total cell stress.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from plasticity import Array, MechanicalMaterial, integrate_material, invariants


Mode = Literal["drained", "undrained"]


@dataclass
class TriaxialHistory:
    data: dict[str, Array]

    def __getitem__(self, name: str) -> Array:
        return self.data[name]


def _volume(state: Any) -> float:
    if hasattr(state, "V"):
        return float(state.V)
    if hasattr(state, "e"):
        return 1.0 + float(state.e)
    return float("nan")


def _hardening_variable(state: Any) -> float:
    for name in ("pc", "p0", "pi"):
        if hasattr(state, name):
            return float(getattr(state, name))
    return float("nan")


def solve_triaxial(
    model: MechanicalMaterial,
    state0: Any,
    axial_strain_end: float,
    mode: Mode = "drained",
    increments: int = 200,
    strain_rate: float = 1.0e-3,
) -> TriaxialHistory:
    """Solve monotonic triaxial compression using the common material API."""

    if axial_strain_end <= 0 or increments < 1 or strain_rate <= 0:
        raise ValueError("positive strain, increments, and strain_rate are required")
    state = state0.copy()
    p_initial = invariants(state.stress)[0]
    radial_target = float(state.stress[1])
    V_initial = _volume(state)
    names = (
        "eps_axial", "eps_radial", "eps_v", "p", "q", "eta", "u",
        "V", "hardening", "psi", "yielded", "substeps",
    )
    data = {name: np.zeros(increments + 1) for name in names}
    p0, q0, *_ = invariants(state.stress)
    data["p"][0], data["q"][0], data["eta"][0] = p0, q0, q0 / p0
    data["V"][0] = V_initial
    data["hardening"][0] = _hardening_variable(state)
    if hasattr(model, "point"):
        data["psi"][0] = model.point(state.stress, state.e, state.pi).psi

    dea = axial_strain_end / increments
    dt = dea / strain_rate
    eps_r_total = 0.0

    for i in range(1, increments + 1):
        if mode == "undrained":
            der = -0.5 * dea
            deps = np.array([dea, der, der, 0.0, 0.0, 0.0])
            response = integrate_material(model, state, deps, dt, compute_tangent=False)
        elif mode == "drained":
            # Constant cell pressure: solve sigma_r(new)-sigma_r(initial)=0.
            der = -0.25 * dea
            response = None
            for _ in range(20):
                deps = np.array([dea, der, der, 0.0, 0.0, 0.0])
                response = integrate_material(model, state, deps, dt, compute_tangent=False)
                residual = float(response.stress[1] - radial_target)
                if abs(residual) <= 2.0e-8 * max(abs(radial_target), 1.0):
                    break
                h = max(1.0e-9, 1.0e-4 * abs(dea))
                deps_h = np.array([dea, der + h, der + h, 0.0, 0.0, 0.0])
                plus = integrate_material(model, state, deps_h, dt, compute_tangent=False)
                derivative = float(plus.stress[1] - response.stress[1]) / h
                if abs(derivative) < 1.0e-12:
                    raise RuntimeError("singular lateral-strain control in triaxial driver")
                correction = float(np.clip(residual / derivative, -2.0 * abs(dea), 2.0 * abs(dea)))
                der -= correction
            else:
                raise RuntimeError("drained triaxial lateral-stress iteration did not converge")
            assert response is not None
        else:
            raise ValueError("mode must be 'drained' or 'undrained'")

        state = response.state
        eps_r_total += der
        p, q, *_ = invariants(state.stress)
        data["eps_axial"][i] = i * dea
        data["eps_radial"][i] = eps_r_total
        data["eps_v"][i] = i * dea + 2.0 * eps_r_total
        data["p"][i], data["q"][i], data["eta"][i] = p, q, q / p
        data["V"][i] = _volume(state)
        data["hardening"][i] = _hardening_variable(state)
        data["yielded"][i] = float(response.yielded)
        data["substeps"][i] = response.substeps
        if mode == "undrained":
            data["u"][i] = p_initial - (p - q / 3.0)
        if hasattr(model, "point"):
            data["psi"][i] = model.point(state.stress, state.e, state.pi).psi

    return TriaxialHistory(data)

