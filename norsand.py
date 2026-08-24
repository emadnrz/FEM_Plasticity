"""Monotonic NorSand outer-surface model for general 3-D stress states.

Implemented features: pressure-dependent isotropic elasticity, semi-log CSL,
state parameter at the image stress, the Shuttle--Jefferies Lode-angle strength
interpolation specified in the supplied algorithm note, associated flow, and
image-pressure hardening with selectable algorithm-note/paper softening laws.

The optional inner cap and principal-stress-rotation term are outside this
monotonic implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, cos, exp, log, pi

import numpy as np

from plasticity import Array, MaterialResponse, invariants, isotropic_stiffness, strain_invariants


@dataclass(frozen=True)
class NorSandParameters:
    Gamma: float = 1.20
    lambda_c: float = 0.06
    M_tc: float = 1.47
    chi_tc: float = 3.2
    N_coupling: float = 0.5
    H0: float = 75.0
    Hy: float = 400.0
    G_ref: float = 20_970.0
    m: float = 0.47
    nu: float = 0.15
    p_ref: float = 100.0
    Z: bool = False
    softening_formulation: str = "algorithm"
    max_strain_step: float = 2.0e-5
    p_min: float = 1.0e-5

    def __post_init__(self) -> None:
        if not (
            self.lambda_c > 0
            and self.M_tc > 0
            and self.chi_tc > 0
            and self.N_coupling >= 0
            and self.H0 > 0
            and self.G_ref > 0
            and 0 <= self.m <= 1
            and self.p_ref > 0
        ):
            raise ValueError("invalid NorSand parameters")
        if self.M_tc <= self.lambda_c * self.chi_tc:
            raise ValueError("M_tc must exceed lambda_c*chi_tc")
        if not -1.0 < self.nu < 0.5:
            raise ValueError("Poisson ratio must satisfy -1 < nu < 0.5")
        if self.softening_formulation not in ("algorithm", "paper_2022"):
            raise ValueError("softening_formulation must be 'algorithm' or 'paper_2022'")


@dataclass
class NorSandState:
    stress: Array
    e: float
    pi: float
    eps_p: Array

    def copy(self) -> "NorSandState":
        return NorSandState(self.stress.copy(), float(self.e), float(self.pi), self.eps_p.copy())


@dataclass(frozen=True)
class NorSandPoint:
    p: float
    q: float
    theta: float
    eta: float
    psi: float
    psi_i: float
    chi_i: float
    Mi_tc: float
    Mi: float
    g: float
    pi_max: float
    eta_L: float
    H: float
    dilatancy: float
    yield_f: float


class NorSand:
    """NorSand material satisfying the common six-component FE interface."""

    def __init__(self, parameters: NorSandParameters):
        self.p = parameters

    def critical_void_ratio(self, p: float) -> float:
        return self.p.Gamma - self.p.lambda_c * log(max(p, self.p.p_min) / self.p.p_ref)

    def initialize_isotropic(self, p0: float, psi0: float, OCR: float = 1.0) -> NorSandState:
        if not p0 > 0.0 or not OCR >= 1.0:
            raise ValueError("NorSand initialization requires p0>0 and OCR>=1")
        e0 = self.critical_void_ratio(p0) + psi0
        # At q=0 the outer surface intersects p/pi=exp(1).  OCR scales pi.
        pi0 = OCR * p0 / exp(1.0)
        return NorSandState(np.array([p0, p0, p0, 0.0, 0.0, 0.0]), e0, pi0, np.zeros(6))

    def initialize(self, stress: Array, e0: float, OCR: float = 1.0) -> NorSandState:
        """Initialize an arbitrary stress state as specified in section 8.

        A Newton solve locates the OCR=1 image pressure on the yield surface;
        the requested OCR then scales that pressure.  The supplied note omits
        ``F/(dF/dpi)`` in its printed Newton update; the dimensionally correct
        Newton quotient is used here.
        """

        stress = np.asarray(stress, dtype=float).reshape(6)
        p, *_ = invariants(stress)
        if p <= self.p.p_min or OCR < 1.0:
            raise ValueError("NorSand initialization requires p>0 and OCR>=1")
        image_pressure = p / exp(1.0)
        for _ in range(50):
            f = self.yield_value(stress, e0, image_pressure)
            if abs(f) <= 1.0e-12:
                break
            h = 2.0e-7 * max(abs(image_pressure), 1.0)
            plus = self.yield_value(stress, e0, image_pressure + h)
            minus_pressure = max(image_pressure - h, self.p.p_min)
            minus = self.yield_value(stress, e0, minus_pressure)
            derivative = (plus - minus) / (image_pressure + h - minus_pressure)
            if abs(derivative) < 1.0e-14:
                raise RuntimeError("singular Newton update in NorSand initialization")
            candidate = image_pressure - f / derivative
            if candidate <= self.p.p_min:
                candidate = 0.5 * image_pressure
            image_pressure = candidate
        else:
            raise RuntimeError("NorSand image-pressure initialization did not converge")
        return NorSandState(stress.copy(), float(e0), OCR * image_pressure, np.zeros(6))

    def elastic_matrix(self, state: NorSandState) -> Array:
        p = max(invariants(state.stress)[0], self.p.p_min)
        G = self.p.G_ref * (p / self.p.p_ref) ** self.p.m
        K = 2.0 * (1.0 + self.p.nu) * G / (3.0 * (1.0 - 2.0 * self.p.nu))
        return isotropic_stiffness(K, G)

    def _M_lode(self, theta: float) -> float:
        """Critical stress ratio from algorithm-note Eq. (15).

        The note uses compression-negative stress, for which triaxial
        compression is theta=-pi/6.  This code is compression-positive, so
        theta changes sign and the cosine argument changes accordingly.
        """

        Mtc = self.p.M_tc
        return Mtc - Mtc**2 / (3.0 + Mtc) * cos(1.5 * theta + pi / 4.0)

    def point(self, stress: Array, e: float, image_pressure: float) -> NorSandPoint:
        p, q, theta, *_ = invariants(stress)
        if p <= self.p.p_min or image_pressure <= self.p.p_min:
            raise ValueError("NorSand requires positive p and image pressure")
        psi = e - self.critical_void_ratio(p)
        psi_i = e - self.critical_void_ratio(image_pressure)
        chi_i = self.p.M_tc * self.p.chi_tc / (
            self.p.M_tc - self.p.lambda_c * self.p.chi_tc
        )
        state_factor = 1.0 - self.p.N_coupling * chi_i * abs(psi_i) / self.p.M_tc
        Mi_tc = self.p.M_tc * state_factor
        Mi = self._M_lode(theta) * state_factor
        if Mi_tc <= 0.0 or Mi <= 0.0:
            raise RuntimeError("NorSand image friction ratio became nonpositive")
        g = Mi / Mi_tc
        eta = q / p
        yf = eta / Mi - 1.0 + log(p / image_pressure)
        if self.p.softening_formulation == "paper_2022":
            limit_exponent = -self.p.chi_tc * psi / Mi_tc
        else:
            limit_exponent = -chi_i * psi_i / Mi_tc
        pi_max = p * exp(float(np.clip(limit_exponent, -50.0, 50.0)))
        eta_L = Mi * (1.0 - chi_i * psi_i / Mi_tc)
        H = self.p.H0 - self.p.Hy * psi
        return NorSandPoint(
            p, q, theta, eta, psi, psi_i, chi_i, Mi_tc, Mi, g,
            pi_max, eta_L, H, Mi - eta, yf,
        )

    def yield_value(self, stress: Array, e: float, image_pressure: float) -> float:
        return self.point(stress, e, image_pressure).yield_f

    def _yield_gradient(self, stress: Array, e: float, image_pressure: float) -> Array:
        stress = np.asarray(stress, dtype=float)
        p, q, *_ = invariants(stress)
        h = 2.0e-7 * max(abs(p), abs(q), 1.0)
        gradient = np.zeros(6)
        for j in range(6):
            plus = stress.copy()
            minus = stress.copy()
            plus[j] += h
            minus[j] -= h
            gradient[j] = (
                self.yield_value(plus, e, image_pressure)
                - self.yield_value(minus, e, image_pressure)
            ) / (2.0 * h)
        return gradient

    def _state_derivatives(self, stress: Array, e: float, image_pressure: float) -> tuple[float, float]:
        he = 1.0e-7 * max(abs(e), 1.0)
        hp = 1.0e-7 * max(abs(image_pressure), 1.0)
        fe = (
            self.yield_value(stress, e + he, image_pressure)
            - self.yield_value(stress, e - he, image_pressure)
        ) / (2.0 * he)
        fpi = (
            self.yield_value(stress, e, image_pressure + hp)
            - self.yield_value(stress, e, image_pressure - hp)
        ) / (2.0 * hp)
        return fe, fpi

    def _image_pressure_rate(self, state: NorSandState, pt: NorSandPoint | None = None) -> float:
        """Return dpi/d(equivalent plastic shear strain)."""

        if pt is None:
            pt = self.point(state.stress, state.e, state.pi)
        D = self.elastic_matrix(state)
        K = float((D[0, 0] + 2.0 * D[0, 1]) / 3.0)
        rate = (
            state.pi
            * pt.H
            * (pt.Mi / pt.Mi_tc)
            * (pt.p / state.pi) ** 2
            * (pt.pi_max / pt.p - state.pi / pt.p)
        )
        if not self.p.Z:
            return rate

        # The augmented term is a *softening* contribution and is active only
        # on the contractive branch.  Without this Macaulay bracket it changes
        # sign during dilation and spuriously becomes an extra hardening law.
        contractive_dilatancy = max(pt.Mi - pt.eta, 0.0)

        if self.p.softening_formulation == "algorithm":
            if abs(pt.eta_L) <= 1.0e-12:
                raise RuntimeError("NorSand limiting stress ratio eta_L is zero")
            # Compression-positive equivalent of algorithm-note Eq. (45).
            softening = (
                state.pi
                * (pt.eta / pt.eta_L)
                * (K / pt.p)
                * contractive_dilatancy
                / (1.0 + pt.chi_i * self.p.lambda_c / pt.Mi_tc)
            )
        else:
            # Marinelli et al. (2022), Eq. (5), used for its Figure 1.
            softening = (
                (1.0 - self.p.lambda_c * self.p.chi_tc / self.p.M_tc)
                * state.pi
                * (pt.eta / pt.Mi)
                * (K / pt.p)
                * contractive_dilatancy
            )
        return rate - softening

    def _hardening_per_lambda(self, state: NorSandState, normal: Array) -> float:
        pt = self.point(state.stress, state.e, state.pi)
        _, deq_per_lambda = strain_invariants(normal)
        return self._image_pressure_rate(state, pt) * deq_per_lambda

    def _correct(self, state: NorSandState) -> None:
        for _ in range(15):
            pt = self.point(state.stress, state.e, state.pi)
            if abs(pt.yield_f) <= 2.0e-9:
                return
            D = self.elastic_matrix(state)
            n = self._yield_gradient(state.stress, state.e, state.pi)
            _, fpi = self._state_derivatives(state.stress, state.e, state.pi)
            hpi = self._hardening_per_lambda(state, n)
            denom = float(n @ D @ n) - fpi * hpi
            if denom <= 0.0:
                return
            dl = pt.yield_f / denom
            state.stress -= dl * (D @ n)
            state.pi = max(state.pi + hpi * dl, self.p.p_min)
            state.eps_p += dl * n

    def integrate_raw(self, state0: NorSandState, deps: Array, dt: float = 0.0) -> MaterialResponse:
        del dt
        deps = np.asarray(deps, dtype=float).reshape(6)
        count = max(1, int(ceil(float(np.linalg.norm(deps)) / self.p.max_strain_step)))
        de = deps / count
        state = state0.copy()
        yielded = False

        for _ in range(count):
            D = self.elastic_matrix(state)
            trial_stress = state.stress + D @ de
            de_void = -(1.0 + state.e) * float(np.sum(de[:3]))
            trial_e = state.e + de_void
            f0 = self.yield_value(state.stress, state.e, state.pi)
            ft = self.yield_value(trial_stress, trial_e, state.pi)
            if ft <= 2.0e-9:
                state.stress = trial_stress
                state.e = trial_e
                continue

            yielded = True
            remaining = de.copy()
            if f0 < -2.0e-9:
                lo, hi = 0.0, 1.0
                for _ in range(50):
                    a = 0.5 * (lo + hi)
                    s = state.stress + a * (D @ de)
                    e_mid = state.e - (1.0 + state.e) * float(np.sum((a * de)[:3]))
                    if self.yield_value(s, e_mid, state.pi) <= 0.0:
                        lo = a
                    else:
                        hi = a
                elastic = lo * de
                state.stress += D @ elastic
                state.e -= (1.0 + state.e) * float(np.sum(elastic[:3]))
                remaining = (1.0 - lo) * de

            D = self.elastic_matrix(state)
            n = self._yield_gradient(state.stress, state.e, state.pi)
            fe, fpi = self._state_derivatives(state.stress, state.e, state.pi)
            hpi = self._hardening_per_lambda(state, n)
            de_void = -(1.0 + state.e) * float(np.sum(remaining[:3]))
            denom = float(n @ D @ n) - fpi * hpi
            numerator = float(n @ D @ remaining) + fe * de_void
            dl = max(0.0, numerator / denom) if denom > 0.0 else 0.0
            state.stress += D @ (remaining - dl * n)
            state.e += de_void
            state.pi = max(state.pi + hpi * dl, self.p.p_min)
            state.eps_p += dl * n
            self._correct(state)

            if invariants(state.stress)[0] <= self.p.p_min:
                raise RuntimeError("NorSand path reached nonpositive mean effective stress")

        return MaterialResponse(state, state.stress.copy(), self.elastic_matrix(state), yielded, count)

    @staticmethod
    def _triaxial_stress(p: float, q: float) -> Array:
        return np.array([p + 2.0 * q / 3.0, p - q / 3.0, p - q / 3.0, 0.0, 0.0, 0.0])

    def _triaxial_f(self, y: Array) -> float:
        p, q, e, image_pressure = map(float, y)
        return self.yield_value(self._triaxial_stress(p, q), e, image_pressure)

    def _triaxial_derivatives(self, y: Array) -> Array:
        derivatives = np.zeros(4)
        for j in range(4):
            h = 2.0e-7 * max(abs(float(y[j])), 1.0)
            plus, minus = y.copy(), y.copy()
            plus[j] += h
            minus[j] -= h
            derivatives[j] = (self._triaxial_f(plus) - self._triaxial_f(minus)) / (2.0 * h)
        return derivatives

    def _triaxial_elastic_rates(self, y: Array, mode: str) -> Array:
        p, q, e, image_pressure = map(float, y)
        state = NorSandState(self._triaxial_stress(p, q), e, image_pressure, np.zeros(6))
        D = self.elastic_matrix(state)
        K = float((D[0, 0] + 2.0 * D[0, 1]) / 3.0)
        G = float(D[3, 3])
        if mode == "undrained":
            ev = 0.0
        else:
            ev = 3.0 * G / (3.0 * K + G)
        eq = 1.0 - ev / 3.0
        return np.array([K * ev, 3.0 * G * eq, -(1.0 + e) * ev, 0.0])

    def _triaxial_plastic_rates(self, y: Array, mode: str) -> Array:
        p, q, e, image_pressure = map(float, y)
        stress = self._triaxial_stress(p, q)
        state = NorSandState(stress, e, image_pressure, np.zeros(6))
        pt = self.point(stress, e, image_pressure)
        D = self.elastic_matrix(state)
        K = float((D[0, 0] + 2.0 * D[0, 1]) / 3.0)
        G = float(D[3, 3])
        fp, fq, fe, fpi = self._triaxial_derivatives(y)
        dpi_deq = self._image_pressure_rate(state, pt)
        Dp = pt.dilatancy

        if mode == "undrained":
            denominator = fp * K * Dp + fq * 3.0 * G - fpi * dpi_deq
            plastic_q = fq * 3.0 * G / denominator
            ev = 0.0
        else:
            A = np.array(
                [
                    [3.0 * K + G, 3.0 * (G - K * Dp)],
                    [
                        (fp + 3.0 * fq) * K - fe * (1.0 + e),
                        -(fp + 3.0 * fq) * K * Dp + fpi * dpi_deq,
                    ],
                ]
            )
            ev, plastic_q = np.linalg.solve(A, np.array([3.0 * G, 0.0]))

        eq = 1.0 - ev / 3.0
        dp = K * (ev - Dp * plastic_q)
        dq = 3.0 * G * (eq - plastic_q)
        return np.array([dp, dq, -(1.0 + e) * ev, dpi_deq * plastic_q])

    def _project_triaxial_image_pressure(self, y: Array) -> None:
        for _ in range(8):
            f = self._triaxial_f(y)
            if abs(f) <= 2.0e-10:
                return
            h = 2.0e-7 * max(abs(float(y[3])), 1.0)
            yp, ym = y.copy(), y.copy()
            yp[3] += h
            ym[3] -= h
            derivative = (self._triaxial_f(yp) - self._triaxial_f(ym)) / (2.0 * h)
            if abs(derivative) < 1.0e-14:
                return
            y[3] = max(y[3] - f / derivative, self.p.p_min)

    def solve_triaxial(
        self,
        state0: NorSandState,
        axial_strain_end: float,
        mode: str = "drained",
        increments: int = 200,
        max_internal_strain_step: float = 2.0e-5,
    ):
        """Fast, consistent axisymmetric driver for verification/calibration."""

        from triaxial import TriaxialHistory

        if mode not in ("drained", "undrained"):
            raise ValueError("mode must be drained or undrained")
        p_initial = invariants(state0.stress)[0]
        y = np.array([p_initial, 0.0, state0.e, state0.pi])
        V_initial = 1.0 + state0.e
        names = (
            "eps_axial", "eps_radial", "eps_v", "p", "q", "eta", "u",
            "V", "hardening", "psi", "yielded", "substeps",
        )
        data = {name: np.zeros(increments + 1) for name in names}
        data["p"][0] = p_initial
        data["V"][0] = V_initial
        data["hardening"][0] = state0.pi
        data["psi"][0] = self.point(state0.stress, state0.e, state0.pi).psi
        output_step = axial_strain_end / increments
        eps_v_total = 0.0

        for i in range(1, increments + 1):
            internal_count = max(1, int(ceil(abs(output_step) / max_internal_strain_step)))
            step = output_step / internal_count
            yielded = False
            for _ in range(internal_count):
                elastic_rate = self._triaxial_elastic_rates(y, mode)
                elastic_ev_rate = -elastic_rate[2] / (1.0 + y[2])
                trial = y + step * elastic_rate
                f0, ft = self._triaxial_f(y), self._triaxial_f(trial)
                remaining = step
                if f0 < -2.0e-10 and ft <= 2.0e-10:
                    y = trial
                    eps_v_total += step * elastic_ev_rate
                    continue
                if f0 < -2.0e-10:
                    lo, hi = 0.0, 1.0
                    for _ in range(45):
                        a = 0.5 * (lo + hi)
                        mid = y + a * step * elastic_rate
                        if self._triaxial_f(mid) <= 0.0:
                            lo = a
                        else:
                            hi = a
                    elastic_part = lo * step
                    y += elastic_part * elastic_rate
                    eps_v_total += elastic_part * elastic_ev_rate
                    remaining = (1.0 - lo) * step
                if remaining > 0.0:
                    yielded = True
                    e_before = y[2]
                    k1 = self._triaxial_plastic_rates(y, mode)
                    predictor = y + remaining * k1
                    self._project_triaxial_image_pressure(predictor)
                    k2 = self._triaxial_plastic_rates(predictor, mode)
                    y += 0.5 * remaining * (k1 + k2)
                    self._project_triaxial_image_pressure(y)
                    eps_v_total += -0.5 * remaining * (
                        k1[2] / (1.0 + max(e_before, -0.99))
                        + k2[2] / (1.0 + max(predictor[2], -0.99))
                    )

            data["eps_axial"][i] = i * output_step
            data["eps_v"][i] = eps_v_total
            data["eps_radial"][i] = 0.5 * (eps_v_total - data["eps_axial"][i])
            data["p"][i], data["q"][i] = y[0], y[1]
            data["eta"][i] = y[1] / y[0]
            data["V"][i] = 1.0 + y[2]
            data["hardening"][i] = y[3]
            stress = self._triaxial_stress(y[0], y[1])
            data["psi"][i] = self.point(stress, y[2], y[3]).psi
            data["yielded"][i] = float(yielded)
            data["substeps"][i] = internal_count
            if mode == "undrained":
                data["u"][i] = p_initial - (y[0] - y[1] / 3.0)
        return TriaxialHistory(data)
