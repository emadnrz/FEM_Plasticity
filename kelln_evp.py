"""Kelln et al. elastic-viscoplastic critical-state model.

This is the previously verified EVP integration recast behind the common
``integrate_material`` interface used by the mechanical FE solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log, sqrt

import numpy as np

from plasticity import Array, MaterialResponse, invariants, isotropic_stiffness


@dataclass(frozen=True)
class EVPParameters:
    M: float = 1.2
    lambda_c: float = 0.06
    kappa: float = 0.01
    psi_visc: float = 1.0e-4
    N: float = 2.0
    t0: float = 1.0
    nu: float = 0.2
    Z: float = 1.0
    p_ref: float = 100.0
    stress_tolerance: float = 1.0e-4
    flow_floor: float = 1.0e-4
    max_substeps: int = 200_000
    p_min: float = 1.0e-6

    def __post_init__(self) -> None:
        if not (
            self.M > 0
            and self.lambda_c > self.kappa > 0
            and self.psi_visc > 0
            and self.t0 > 0
            and self.p_ref > 0
        ):
            raise ValueError("invalid EVP material parameters")


@dataclass
class EVPState:
    stress: Array
    V: float
    p0: float
    eps_vp: Array

    def copy(self) -> "EVPState":
        return EVPState(self.stress.copy(), float(self.V), float(self.p0), self.eps_vp.copy())


@dataclass(frozen=True)
class _Flow:
    rate: Array
    p_rate: float
    q_rate: float
    axial_rate: float
    yield_f: float


class KellnEVP:
    def __init__(self, parameters: EVPParameters):
        self.p = parameters

    def initialize_isotropic(self, p0: float, pc0: float) -> EVPState:
        if not pc0 >= p0 > 0.0:
            raise ValueError("EVP initialization requires pc0 >= p0 > 0")
        m = self.p
        V = m.N - m.lambda_c * log(pc0 / m.p_ref) + m.kappa * log(pc0 / p0)
        gate = m.p_ref * exp(
            (m.Z - V - m.kappa * log(p0 / m.p_ref)) / (m.lambda_c - m.kappa)
        )
        return EVPState(np.array([p0, p0, p0, 0.0, 0.0, 0.0]), V, gate, np.zeros(6))

    def elastic_matrix(self, state: EVPState) -> Array:
        p = max(invariants(state.stress)[0], self.p.p_min)
        K = state.V * p / self.p.kappa
        G = 3.0 * (1.0 - 2.0 * self.p.nu) * K / (2.0 * (1.0 + self.p.nu))
        return isotropic_stiffness(K, G)

    def _flow(self, stress: Array, V: float, gate: float) -> _Flow:
        m = self.p
        p, q, _, dp, dq = invariants(stress)
        if p <= m.p_min:
            raise ValueError(f"EVP mean effective stress became nonpositive: {p:g}")
        pm = p + q * q / (m.M * m.M * p)
        Vm = V - m.kappa * log(pm / p)
        dgdp = 1.0 - q * q / (m.M * m.M * p * p)
        dgdq = 2.0 * q / (m.M * m.M * p)
        yf = q * q / (m.M * m.M * p) - gate + p
        log_rate = (
            log(m.psi_visc / (Vm * m.t0))
            + (Vm - m.N + m.lambda_c * log(pm / m.p_ref)) / m.psi_visc
        )
        base = exp(float(np.clip(log_rate, -700.0, 700.0)))
        scalar = base / max(abs(dgdp), m.flow_floor) if yf > 0.0 else 0.0
        rate = scalar * (dgdp * dp + dgdq * dq)
        p_rate = float(np.sum(rate[:3]))
        q_rate = scalar * dgdq
        return _Flow(rate, p_rate, q_rate, p_rate / 3.0 + q_rate, yf)

    def integrate_raw(self, state0: EVPState, deps: Array, dt: float) -> MaterialResponse:
        deps = np.asarray(deps, dtype=float).reshape(6)
        if dt < 0.0:
            raise ValueError("dt must be nonnegative")
        state = state0.copy()
        D0 = self.elastic_matrix(state)
        trial_stress = state.stress + D0 @ deps
        trial_V = state.V * (1.0 - float(np.sum(deps[:3])))
        f0 = self._flow(state.stress, state.V, state.p0).yield_f
        ft = self._flow(trial_stress, trial_V, state.p0).yield_f
        if ft <= 0.0 or dt == 0.0:
            state.stress = trial_stress
            state.V = trial_V
            return MaterialResponse(state, state.stress.copy(), self.elastic_matrix(state), False, 1)

        alpha = 0.0 if f0 >= 0.0 else float(np.clip(f0 / (f0 - ft), 0.0, 1.0))
        state.stress += D0 @ (alpha * deps)
        state.V *= 1.0 - float(np.sum((alpha * deps)[:3]))
        deps_path = (1.0 - alpha) * deps
        dt_path = (1.0 - alpha) * dt
        progress = 0.0
        fraction = 1.0
        accepted = rejected = 0

        while progress < 1.0 - 10.0 * np.finfo(float).eps:
            if accepted + rejected >= self.p.max_substeps:
                raise RuntimeError("EVP maximum stress-integration substeps exceeded")
            fraction = min(fraction, 1.0 - progress)
            de = fraction * deps_path
            dtime = fraction * dt_path
            old = state.copy()

            D1 = self.elastic_matrix(old)
            r1 = self._flow(old.stress, old.V, old.p0)
            ds1 = D1 @ (de - r1.rate * dtime)
            dep1 = r1.rate * dtime
            dp01 = old.V * old.p0 / (self.p.lambda_c - self.p.kappa) * r1.p_rate * dtime

            predictor = old.copy()
            predictor.stress = old.stress + ds1
            predictor.V = old.V * (1.0 - float(np.sum(de[:3])))
            predictor.p0 = old.p0 + dp01
            D2 = self.elastic_matrix(predictor)
            r2 = self._flow(predictor.stress, predictor.V, predictor.p0)
            ds2 = D2 @ (de - r2.rate * dtime)
            dep2 = r2.rate * dtime
            dp02 = (
                predictor.V
                * predictor.p0
                / (self.p.lambda_c - self.p.kappa)
                * r2.p_rate
                * dtime
            )

            ds = 0.5 * (ds1 + ds2)
            error = float(np.linalg.norm(0.5 * (ds2 - ds1))) / max(
                float(np.linalg.norm(old.stress + ds)), self.p.p_min
            )
            if not np.isfinite(error) or error > self.p.stress_tolerance:
                beta = 0.1 if not np.isfinite(error) else 0.8 * sqrt(self.p.stress_tolerance / error)
                fraction *= min(max(beta, 0.1), 2.0)
                rejected += 1
                continue

            state.stress = old.stress + ds
            state.V = predictor.V
            state.p0 = old.p0 + 0.5 * (dp01 + dp02)
            state.eps_vp = old.eps_vp + 0.5 * (dep1 + dep2)
            progress += fraction
            accepted += 1
            beta = 2.0 if error <= np.finfo(float).eps else 0.8 * sqrt(self.p.stress_tolerance / error)
            fraction *= min(max(beta, 0.1), 2.0)

        return MaterialResponse(state, state.stress.copy(), self.elastic_matrix(state), True, accepted)

    @staticmethod
    def _triaxial_stress(p: float, q: float) -> Array:
        return np.array([p + 2.0 * q / 3.0, p - q / 3.0, p - q / 3.0, 0.0, 0.0, 0.0])

    def _triaxial_rates(self, y: Array, axial_rate: float, mode: str) -> Array:
        p, q, V, gate = map(float, y)
        state = EVPState(self._triaxial_stress(p, q), V, gate, np.zeros(6))
        vp = self._flow(state.stress, V, gate)
        D = self.elastic_matrix(state)
        G = float(D[3, 3])
        if mode == "drained":
            compliance = self.p.kappa / (9.0 * V * p) + 1.0 / (3.0 * G)
            dsigma1 = (axial_rate - vp.axial_rate) / compliance
            dp = dsigma1 / 3.0
            dq = dsigma1
            eps_v_rate = self.p.kappa * dp / (V * p) + vp.p_rate
            dV = -V * eps_v_rate
        elif mode == "undrained":
            dp = -(V * p / self.p.kappa) * vp.p_rate
            dq = 3.0 * G * (axial_rate - vp.q_rate)
            dV = 0.0
        else:
            raise ValueError("mode must be drained or undrained")
        dgate = V * gate / (self.p.lambda_c - self.p.kappa) * vp.p_rate
        return np.array([dp, dq, dV, dgate])

    def solve_triaxial(
        self,
        state0: EVPState,
        axial_strain_end: float,
        mode: str = "drained",
        increments: int = 200,
        strain_rate: float = 1.0e-3,
    ):
        """Adaptive ideal-triaxial integration of the Kelln EVP equations."""

        from triaxial import TriaxialHistory

        if mode not in ("drained", "undrained"):
            raise ValueError("mode must be drained or undrained")
        p_initial = invariants(state0.stress)[0]
        y = np.array([p_initial, 0.0, state0.V, state0.p0])
        V_initial = state0.V
        names = (
            "eps_axial", "eps_radial", "eps_v", "p", "q", "eta", "u",
            "V", "hardening", "psi", "yielded", "substeps",
        )
        data = {name: np.zeros(increments + 1) for name in names}
        data["p"][0], data["V"][0], data["hardening"][0] = y[0], y[2], y[3]
        output_strain = axial_strain_end / increments
        output_dt = output_strain / strain_rate

        for i in range(1, increments + 1):
            progress, fraction = 0.0, 1.0
            accepted = 0
            while progress < 1.0 - 10.0 * np.finfo(float).eps:
                if accepted >= self.p.max_substeps:
                    raise RuntimeError("EVP triaxial substep limit exceeded")
                fraction = min(fraction, 1.0 - progress)
                h = fraction * output_dt
                k1 = self._triaxial_rates(y, strain_rate, mode)
                predictor = y + h * k1
                if predictor[0] <= self.p.p_min or predictor[2] <= 1.0:
                    fraction *= 0.25
                    continue
                k2 = self._triaxial_rates(predictor, strain_rate, mode)
                dy = 0.5 * h * (k1 + k2)
                error = float(np.linalg.norm(0.5 * h * (k2[:2] - k1[:2]))) / max(
                    float(np.linalg.norm(y[:2] + dy[:2])), self.p.p_min
                )
                candidate = y + dy
                if (
                    not np.all(np.isfinite(candidate))
                    or candidate[0] <= self.p.p_min
                    or candidate[2] <= 1.0
                    or error > self.p.stress_tolerance
                ):
                    beta = 0.1 if not np.isfinite(error) else 0.8 * sqrt(self.p.stress_tolerance / error)
                    fraction *= min(max(beta, 0.1), 2.0)
                    continue
                y = candidate
                progress += fraction
                accepted += 1
                beta = 2.0 if error <= np.finfo(float).eps else 0.8 * sqrt(self.p.stress_tolerance / error)
                fraction *= min(max(beta, 0.1), 2.0)

            data["eps_axial"][i] = i * output_strain
            data["p"][i], data["q"][i], data["V"][i], data["hardening"][i] = y
            data["eta"][i] = y[1] / y[0]
            data["eps_v"][i] = -log(y[2] / V_initial)
            data["eps_radial"][i] = 0.5 * (data["eps_v"][i] - data["eps_axial"][i])
            data["substeps"][i] = accepted
            data["yielded"][i] = 1.0
            if mode == "undrained":
                data["u"][i] = p_initial - (y[0] - y[1] / 3.0)
        return TriaxialHistory(data)
