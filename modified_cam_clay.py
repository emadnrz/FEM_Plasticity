"""Rate-independent Modified Cam-Clay under a common FE material interface."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log

import numpy as np

from plasticity import Array, MaterialResponse, invariants, isotropic_stiffness


@dataclass(frozen=True)
class MCCParameters:
    M: float = 1.2
    lambda_c: float = 0.06
    kappa: float = 0.01
    N: float = 2.0
    nu: float = 0.2
    p_ref: float = 100.0
    max_strain_step: float = 2.0e-5
    p_min: float = 1.0e-6

    def __post_init__(self) -> None:
        if not (self.M > 0 and self.lambda_c > self.kappa > 0 and self.p_ref > 0):
            raise ValueError("require M>0, lambda_c>kappa>0, and p_ref>0")
        if not -1.0 < self.nu < 0.5:
            raise ValueError("Poisson ratio must satisfy -1 < nu < 0.5")


@dataclass
class MCCState:
    stress: Array
    V: float
    pc: float
    eps_p: Array

    def copy(self) -> "MCCState":
        return MCCState(self.stress.copy(), float(self.V), float(self.pc), self.eps_p.copy())


class ModifiedCamClay:
    """Six-component MCC model with associated flow and isotropic hardening."""

    def __init__(self, parameters: MCCParameters):
        self.p = parameters

    def initialize_isotropic(self, p0: float, pc0: float) -> MCCState:
        if not pc0 >= p0 > 0.0:
            raise ValueError("MCC initialization requires pc0 >= p0 > 0")
        m = self.p
        V = m.N - m.lambda_c * log(pc0 / m.p_ref) + m.kappa * log(pc0 / p0)
        return MCCState(np.array([p0, p0, p0, 0.0, 0.0, 0.0]), V, pc0, np.zeros(6))

    def elastic_matrix(self, state: MCCState) -> Array:
        p = max(invariants(state.stress)[0], self.p.p_min)
        K = state.V * p / self.p.kappa
        G = 3.0 * (1.0 - 2.0 * self.p.nu) * K / (2.0 * (1.0 + self.p.nu))
        return isotropic_stiffness(K, G)

    def yield_value(self, stress: Array, pc: float) -> float:
        p, q, *_ = invariants(stress)
        return q * q / self.p.M**2 - p * (pc - p)

    def _normal(self, stress: Array, pc: float) -> tuple[Array, float]:
        p, q, _, dp, dq = invariants(stress)
        dfdp = 2.0 * p - pc
        dfdq = 2.0 * q / self.p.M**2
        return dfdp * dp + dfdq * dq, dfdp

    def _correct(self, state: MCCState) -> None:
        for _ in range(12):
            p = invariants(state.stress)[0]
            f = self.yield_value(state.stress, state.pc)
            if abs(f) <= 1.0e-10 * max(p * state.pc, 1.0):
                return
            D = self.elastic_matrix(state)
            n, dfdp = self._normal(state.stress, state.pc)
            hard = state.V * state.pc / (self.p.lambda_c - self.p.kappa)
            denom = float(n @ D @ n) + p * hard * dfdp
            if denom <= 0.0:
                return
            dl = f / denom
            state.stress -= dl * (D @ n)
            state.pc += hard * dfdp * dl
            state.eps_p += dl * n

    def integrate_raw(self, state0: MCCState, deps: Array, dt: float = 0.0) -> MaterialResponse:
        del dt
        deps = np.asarray(deps, dtype=float).reshape(6)
        count = max(1, int(ceil(float(np.linalg.norm(deps)) / self.p.max_strain_step)))
        de = deps / count
        state = state0.copy()
        yielded = False

        for _ in range(count):
            D = self.elastic_matrix(state)
            trial_stress = state.stress + D @ de
            trial_V = state.V * (1.0 - float(np.sum(de[:3])))
            p0 = invariants(state.stress)[0]
            pt = invariants(trial_stress)[0]
            f0 = self.yield_value(state.stress, state.pc)
            ft = self.yield_value(trial_stress, state.pc)
            tol = 1.0e-10 * max(pt * state.pc, 1.0)
            if ft <= tol:
                state.stress = trial_stress
                state.V = trial_V
                continue

            yielded = True
            remaining = de.copy()
            if f0 < -tol:
                lo, hi = 0.0, 1.0
                for _ in range(50):
                    a = 0.5 * (lo + hi)
                    s = state.stress + a * (D @ de)
                    if self.yield_value(s, state.pc) <= 0.0:
                        lo = a
                    else:
                        hi = a
                elastic = lo * de
                state.stress += D @ elastic
                state.V *= 1.0 - float(np.sum(elastic[:3]))
                remaining = (1.0 - lo) * de

            D = self.elastic_matrix(state)
            p = invariants(state.stress)[0]
            n, dfdp = self._normal(state.stress, state.pc)
            hard = state.V * state.pc / (self.p.lambda_c - self.p.kappa)
            denom = float(n @ D @ n) + p * hard * dfdp
            dl = max(0.0, float(n @ D @ remaining) / denom) if denom > 0.0 else 0.0
            state.stress += D @ (remaining - dl * n)
            state.pc += hard * dfdp * dl
            state.V *= 1.0 - float(np.sum(remaining[:3]))
            state.eps_p += dl * n
            self._correct(state)

        return MaterialResponse(state, state.stress.copy(), self.elastic_matrix(state), yielded, count)

    def _triaxial_elastic_step(self, y: Array, de_axial: float, mode: str) -> Array:
        p, q, V, pc = map(float, y)
        state = MCCState(
            np.array([p + 2.0 * q / 3.0, p - q / 3.0, p - q / 3.0, 0, 0, 0]),
            V, pc, np.zeros(6),
        )
        D = self.elastic_matrix(state)
        K = float((D[0, 0] + 2.0 * D[0, 1]) / 3.0)
        G = float(D[3, 3])
        if mode == "undrained":
            return np.array([p, q + 3.0 * G * de_axial, V, pc])
        compliance = self.p.kappa / (9.0 * V * p) + 1.0 / (3.0 * G)
        dsigma1 = de_axial / compliance
        dp = dsigma1 / 3.0
        dq = dsigma1
        return np.array([p + dp, q + dq, V * (1.0 - dp / K), pc])

    def _triaxial_plastic_rates(self, y: Array, mode: str) -> Array:
        p, q, V, pc = map(float, y)
        state = MCCState(
            np.array([p + 2.0 * q / 3.0, p - q / 3.0, p - q / 3.0, 0, 0, 0]),
            V, pc, np.zeros(6),
        )
        D = self.elastic_matrix(state)
        K = float((D[0, 0] + 2.0 * D[0, 1]) / 3.0)
        G = float(D[3, 3])
        dfdp = 2.0 * p - pc
        dfdq = 2.0 * q / self.p.M**2
        hard = V * pc / (self.p.lambda_c - self.p.kappa)
        if mode == "undrained":
            denom = K * dfdp**2 + 3.0 * G * dfdq**2 + p * hard * dfdp
            dl = 3.0 * G * dfdq / denom if denom > 0.0 else 0.0
            dp = -K * dfdp * dl
            dq = 3.0 * G * (1.0 - dfdq * dl)
            dV = 0.0
        else:
            A = np.array(
                [
                    [1.0, -1.0 / 3.0, 0.0],
                    [1.0 / (3.0 * K), 1.0 / (3.0 * G), dfdp / 3.0 + dfdq],
                    [dfdp, dfdq, -p * hard * dfdp],
                ]
            )
            dp, dq, dl = np.linalg.solve(A, np.array([0.0, 1.0, 0.0]))
            dV = -V * (dp / K + dl * dfdp)
        return np.array([dp, dq, dV, hard * dfdp * dl])

    def solve_triaxial(
        self,
        state0: MCCState,
        axial_strain_end: float,
        mode: str = "drained",
        increments: int = 200,
        max_internal_strain_step: float = 2.0e-5,
    ):
        """Fast, consistent ideal-triaxial MCC solution."""

        from triaxial import TriaxialHistory

        if mode not in ("drained", "undrained"):
            raise ValueError("mode must be drained or undrained")
        p_initial = invariants(state0.stress)[0]
        y = np.array([p_initial, 0.0, state0.V, state0.pc])
        V_initial = state0.V
        names = (
            "eps_axial", "eps_radial", "eps_v", "p", "q", "eta", "u",
            "V", "hardening", "psi", "yielded", "substeps",
        )
        data = {name: np.zeros(increments + 1) for name in names}
        data["p"][0], data["V"][0], data["hardening"][0] = y[0], y[2], y[3]
        output_step = axial_strain_end / increments

        for i in range(1, increments + 1):
            count = max(1, int(ceil(abs(output_step) / max_internal_strain_step)))
            step = output_step / count
            yielded = False
            for _ in range(count):
                current_f = y[1] ** 2 / self.p.M**2 - y[0] * (y[3] - y[0])
                trial = self._triaxial_elastic_step(y, step, mode)
                trial_f = trial[1] ** 2 / self.p.M**2 - trial[0] * (y[3] - trial[0])
                tol = 1.0e-10 * max(y[0] * y[3], 1.0)
                if current_f < -tol and trial_f <= tol:
                    y = trial
                    continue
                remaining = step
                if current_f < -tol:
                    lo, hi = 0.0, 1.0
                    for _ in range(45):
                        a = 0.5 * (lo + hi)
                        mid = self._triaxial_elastic_step(y, a * step, mode)
                        fmid = mid[1] ** 2 / self.p.M**2 - mid[0] * (y[3] - mid[0])
                        if fmid <= 0.0:
                            lo = a
                        else:
                            hi = a
                    y = self._triaxial_elastic_step(y, lo * step, mode)
                    remaining = (1.0 - lo) * step
                if remaining > 0.0:
                    yielded = True
                    k1 = self._triaxial_plastic_rates(y, mode)
                    predictor = y + remaining * k1
                    predictor[3] = predictor[0] + predictor[1] ** 2 / (
                        self.p.M**2 * predictor[0]
                    )
                    k2 = self._triaxial_plastic_rates(predictor, mode)
                    y += 0.5 * remaining * (k1 + k2)
                    y[3] = y[0] + y[1] ** 2 / (self.p.M**2 * y[0])

            data["eps_axial"][i] = i * output_step
            data["p"][i], data["q"][i], data["V"][i], data["hardening"][i] = y
            data["eta"][i] = y[1] / y[0]
            data["eps_v"][i] = -log(y[2] / V_initial)
            data["eps_radial"][i] = 0.5 * (data["eps_v"][i] - data["eps_axial"][i])
            data["yielded"][i] = float(yielded)
            data["substeps"][i] = count
            if mode == "undrained":
                data["u"][i] = p_initial - (y[0] - y[1] / 3.0)
        return TriaxialHistory(data)
