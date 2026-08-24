"""NorSand setup for the coupled hydro-mechanical solver."""

from __future__ import annotations

from math import exp

from norsand import NorSand, NorSandParameters

from .common import CommonCalibration, MaterialCase


def _ocr_for_yield_point(
    model: NorSand,
    initial_pressure: float,
    target_p: float,
    target_q: float,
    target_void_ratio: float,
) -> float:
    stress = model._triaxial_stress(target_p, target_q)
    lo, hi = 1.0, 30.0
    for _ in range(70):
        ocr = 0.5 * (lo + hi)
        image_pressure = ocr * initial_pressure / exp(1.0)
        if model.yield_value(stress, target_void_ratio, image_pressure) > 0.0:
            lo = ocr
        else:
            hi = ocr
    return 0.5 * (lo + hi)


def create_norsand_case(
    calibration: CommonCalibration,
    state_parameter: float,
    matched_yield_point: tuple[float, float, float],
    max_strain_step: float = 2.0e-4,
) -> MaterialCase:
    """Create NorSand with CSL/yield onset matched to MCC.

    ``H0=75`` is the undrained-path calibration.  It makes both the dense and
    loose stress paths close to MCC without changing the common CSL or the
    independently specified initial state parameter.
    """

    c = calibration
    model = NorSand(
        NorSandParameters(
            Gamma=c.critical_void_ratio_at_reference,
            lambda_c=c.compression_slope,
            M_tc=c.critical_stress_ratio,
            chi_tc=4.5,
            N_coupling=0.30,
            H0=75.0,
            Hy=260.0,
            G_ref=13_400.0,
            m=1.0,
            nu=c.poisson_ratio,
            p_ref=c.reference_pressure,
            max_strain_step=max_strain_step,
        )
    )
    target_p, target_q, target_e = matched_yield_point
    ocr = _ocr_for_yield_point(
        model,
        c.initial_effective_pressure,
        target_p,
        target_q,
        target_e,
    )
    state = model.initialize_isotropic(
        c.initial_effective_pressure, state_parameter, ocr
    )
    return MaterialCase(
        "NorSand",
        model,
        state,
        {
            "state_parameter": state_parameter,
            "OCR": ocr,
            "initial_specific_volume": 1.0 + state.e,
            "H0_undrained_calibration": model.p.H0,
        },
    )
