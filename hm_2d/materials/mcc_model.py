"""Modified Cam-Clay setup for the coupled hydro-mechanical solver."""

from __future__ import annotations

from math import exp, log

from modified_cam_clay import MCCParameters, ModifiedCamClay
from plasticity import invariants

from .common import CommonCalibration, MaterialCase


def normal_consolidation_intercept(calibration: CommonCalibration) -> float:
    """Map an e-ln(p') critical-state line to the MCC NCL intercept."""

    c = calibration
    return 1.0 + c.critical_void_ratio_at_reference + (
        c.compression_slope - c.swelling_slope
    ) * log(2.0)


def preconsolidation_pressure(
    calibration: CommonCalibration, state_parameter: float
) -> float:
    """MCC pressure that gives the requested initial state parameter."""

    c = calibration
    return 2.0 * c.initial_effective_pressure * exp(
        -state_parameter / (c.compression_slope - c.swelling_slope)
    )


def create_mcc_case(
    calibration: CommonCalibration,
    state_parameter: float,
    max_strain_step: float = 2.0e-4,
) -> MaterialCase:
    c = calibration
    pc = preconsolidation_pressure(c, state_parameter)
    model = ModifiedCamClay(
        MCCParameters(
            M=c.critical_stress_ratio,
            lambda_c=c.compression_slope,
            kappa=c.swelling_slope,
            N=normal_consolidation_intercept(c),
            nu=c.poisson_ratio,
            p_ref=c.reference_pressure,
            max_strain_step=max_strain_step,
        )
    )
    state = model.initialize_isotropic(c.initial_effective_pressure, pc)
    return MaterialCase(
        "MCC",
        model,
        state,
        {
            "state_parameter": state_parameter,
            "preconsolidation_pressure": pc,
            "initial_specific_volume": state.V,
        },
    )


def drained_first_yield(case: MaterialCase) -> tuple[float, float, float]:
    """Exact constant-cell-pressure intersection with the MCC ellipse."""

    model = case.model
    state = case.initial_state
    p0 = invariants(state.stress)[0]
    lo, hi = p0, state.pc
    for _ in range(70):
        p = 0.5 * (lo + hi)
        q = 3.0 * (p - p0)
        if q * q / model.p.M**2 - p * (state.pc - p) <= 0.0:
            lo = p
        else:
            hi = p
    p = 0.5 * (lo + hi)
    q = 3.0 * (p - p0)
    specific_volume = state.V - model.p.kappa * log(p / p0)
    return p, q, specific_volume - 1.0
