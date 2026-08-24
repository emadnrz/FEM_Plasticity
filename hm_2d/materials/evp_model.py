"""Kelln EVP setup for the coupled hydro-mechanical solver."""

from __future__ import annotations

from kelln_evp import EVPParameters, KellnEVP

from .common import CommonCalibration, MaterialCase
from .mcc_model import normal_consolidation_intercept, preconsolidation_pressure


def create_evp_case(
    calibration: CommonCalibration,
    state_parameter: float,
    viscosity_index: float = 1.0e-4,
) -> MaterialCase:
    c = calibration
    pc = preconsolidation_pressure(c, state_parameter)
    model = KellnEVP(
        EVPParameters(
            M=c.critical_stress_ratio,
            lambda_c=c.compression_slope,
            kappa=c.swelling_slope,
            psi_visc=viscosity_index,
            N=normal_consolidation_intercept(c),
            t0=1.0,
            nu=c.poisson_ratio,
            Z=1.0,
            p_ref=c.reference_pressure,
        )
    )
    state = model.initialize_isotropic(c.initial_effective_pressure, pc)
    return MaterialCase(
        "EVP",
        model,
        state,
        {
            "state_parameter": state_parameter,
            "preconsolidation_pressure": pc,
            "viscosity_index": viscosity_index,
            "initial_specific_volume": state.V,
        },
    )
