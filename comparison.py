"""Verification and dense/loose matching study for MCC, EVP, and NorSand."""

from __future__ import annotations

from dataclasses import asdict
import csv
import json
from math import exp, log
from pathlib import Path

import numpy as np

from kelln_evp import EVPParameters, KellnEVP
from modified_cam_clay import MCCParameters, ModifiedCamClay
from norsand import NorSand, NorSandParameters
from plotting import save_matched_plot, save_norsand_verification_plot


P_INITIAL = 100.0
M_COMMON = 1.2
LAMBDA_COMMON = 0.06
KAPPA_COMMON = 0.01
GAMMA_E_COMMON = 0.80
NU_COMMON = 0.20
P_REF_COMMON = 100.0
STATE_PARAMETERS = {"dense": -0.05, "loose": 0.02}


def mcc_N_from_csl(Gamma_e: float, lambda_c: float, kappa: float) -> float:
    """Map the e-ln(p) CSL intercept to the MCC NCL intercept."""

    return 1.0 + Gamma_e + (lambda_c - kappa) * log(2.0)


def mcc_pc_from_state_parameter(p: float, psi_state: float, lambda_c: float, kappa: float) -> float:
    """MCC preconsolidation pressure giving the requested initial state."""

    return 2.0 * p * exp(-psi_state / (lambda_c - kappa))


def mcc_drained_first_yield(model: ModifiedCamClay, state) -> tuple[float, float, float]:
    """Exact elastic constant-cell-pressure intersection with the MCC ellipse."""

    p0 = float(np.mean(state.stress[:3]))
    lo, hi = p0, state.pc
    for _ in range(70):
        p = 0.5 * (lo + hi)
        q = 3.0 * (p - p0)
        f = q * q / model.p.M**2 - p * (state.pc - p)
        if f <= 0.0:
            lo = p
        else:
            hi = p
    p = 0.5 * (lo + hi)
    q = 3.0 * (p - p0)
    V = state.V - model.p.kappa * log(p / p0)
    return p, q, V - 1.0


def norsand_ocr_for_yield_match(
    model: NorSand,
    p_initial: float,
    target_p: float,
    target_q: float,
    target_e: float,
) -> float:
    """Choose NorSand OCR so its initial surface crosses the MCC yield point."""

    stress = model._triaxial_stress(target_p, target_q)
    lo, hi = 1.0, 30.0
    for _ in range(70):
        OCR = 0.5 * (lo + hi)
        image_pressure = OCR * p_initial / exp(1.0)
        if model.yield_value(stress, target_e, image_pressure) > 0.0:
            lo = OCR
        else:
            hi = OCR
    return 0.5 * (lo + hi)


def _write_csv(columns: dict[str, np.ndarray], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(zip(*columns.values(), strict=True))


def _write_rows(rows: list[dict[str, object]], path: Path) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_matched_models():
    N_mcc = mcc_N_from_csl(GAMMA_E_COMMON, LAMBDA_COMMON, KAPPA_COMMON)
    mcc_parameters = MCCParameters(
        M=M_COMMON, lambda_c=LAMBDA_COMMON, kappa=KAPPA_COMMON, N=N_mcc,
        nu=NU_COMMON, p_ref=P_REF_COMMON, max_strain_step=1.0e-4,
    )
    evp_parameters = EVPParameters(
        M=M_COMMON, lambda_c=LAMBDA_COMMON, kappa=KAPPA_COMMON,
        psi_visc=1.0e-4, N=N_mcc, t0=1.0, nu=NU_COMMON, Z=1.0,
        p_ref=P_REF_COMMON,
    )
    # G_ref matches the average initial MCC shear modulus of the two states.
    norsand_parameters = NorSandParameters(
        Gamma=GAMMA_E_COMMON, lambda_c=LAMBDA_COMMON, M_tc=M_COMMON,
        chi_tc=4.5, N_coupling=0.30, H0=37.0, Hy=260.0,
        G_ref=13_400.0, m=1.0, nu=NU_COMMON, p_ref=P_REF_COMMON,
        max_strain_step=1.0e-4,
    )
    return (
        ModifiedCamClay(mcc_parameters),
        KellnEVP(evp_parameters),
        NorSand(norsand_parameters),
    )


def run_norsand_reference(results_dir: Path):
    """Run the public Itasca dense/loose monotonic verification parameters."""

    parameters = NorSandParameters(
        Gamma=1.2, lambda_c=0.06, M_tc=1.47, chi_tc=3.2,
        N_coupling=0.5, H0=75.0, Hy=400.0, G_ref=20_970.0,
        m=0.47, nu=0.15, p_ref=100.0,
    )
    model = NorSand(parameters)
    histories = {
        "dense": model.solve_triaxial(
            model.initialize_isotropic(395.0, -0.025, 1.1),
            0.10, "undrained", 100, 5.0e-5,
        ),
        "loose": model.solve_triaxial(
            model.initialize_isotropic(395.0, 0.10, 1.1),
            0.10, "undrained", 100, 5.0e-5,
        ),
    }
    columns = {"eps_axial": histories["dense"]["eps_axial"]}
    rows = []
    for label, history in histories.items():
        for name in ("p", "q", "u", "eta", "psi"):
            columns[f"{label}_{name}"] = history[name]
        rows.append(
            {
                "state": label,
                "q_peak_kpa": float(np.max(history["q"])),
                "q_end_kpa": float(history["q"][-1]),
                "p_end_kpa": float(history["p"][-1]),
                "psi_end": float(history["psi"][-1]),
            }
        )
    _write_csv(columns, results_dir / "norsand_itasca_reference_histories.csv")
    _write_rows(rows, results_dir / "norsand_itasca_reference_summary.csv")
    save_norsand_verification_plot(histories, results_dir / "norsand_itasca_reference.png")
    return histories, rows, parameters


def run_matched_comparison(results_dir: Path):
    mcc, evp, norsand = build_matched_models()
    q_cs = M_COMMON * P_INITIAL / (1.0 - M_COMMON / 3.0)
    p_cs = P_INITIAL + q_cs / 3.0
    V_cs = 1.0 + GAMMA_E_COMMON - LAMBDA_COMMON * log(p_cs / P_REF_COMMON)
    all_histories = {}
    rows: list[dict[str, object]] = []
    mappings = {}

    for label, psi_state in STATE_PARAMETERS.items():
        pc = mcc_pc_from_state_parameter(
            P_INITIAL, psi_state, LAMBDA_COMMON, KAPPA_COMMON
        )
        mcc_state = mcc.initialize_isotropic(P_INITIAL, pc)
        evp_state = evp.initialize_isotropic(P_INITIAL, pc)
        py, qy, ey = mcc_drained_first_yield(mcc, mcc_state)
        OCR = norsand_ocr_for_yield_match(norsand, P_INITIAL, py, qy, ey)
        norsand_state = norsand.initialize_isotropic(P_INITIAL, psi_state, OCR)

        histories = {
            "MCC": mcc.solve_triaxial(mcc_state, 0.30, "drained", 100, 5.0e-5),
            "EVP": evp.solve_triaxial(evp_state, 0.30, "drained", 100, 1.0e-3),
            "NorSand": norsand.solve_triaxial(
                norsand_state, 0.30, "drained", 100, 1.0e-4
            ),
        }
        all_histories[label] = histories
        mappings[label] = {
            "state_parameter": psi_state,
            "mcc_pc_kpa": pc,
            "mcc_pc_over_p": pc / P_INITIAL,
            "matched_yield_p_kpa": py,
            "matched_yield_q_kpa": qy,
            "norsand_OCR": OCR,
        }

        columns = {"eps_axial": histories["MCC"]["eps_axial"]}
        for model_name, history in histories.items():
            tag = model_name.lower()
            for name in ("p", "q", "eps_v", "V", "eta"):
                columns[f"{tag}_{name}"] = history[name]
            mcc_history = histories["MCC"]
            q_rmse = 100.0 * float(np.sqrt(np.mean((history["q"] - mcc_history["q"]) ** 2))) / float(
                np.max(mcc_history["q"])
            )
            V_rmse = 100.0 * float(np.sqrt(np.mean((history["V"] - mcc_history["V"]) ** 2))) / float(
                np.mean(mcc_history["V"])
            )
            rows.append(
                {
                    "state": label,
                    "model": model_name,
                    "state_parameter_initial": psi_state,
                    "mcc_pc_kpa": pc,
                    "norsand_OCR": OCR,
                    "q_peak_kpa": float(np.max(history["q"])),
                    "q_end_kpa": float(history["q"][-1]),
                    "q_end_error_from_common_cs_pct": 100.0 * (float(history["q"][-1]) - q_cs) / q_cs,
                    "V_end": float(history["V"][-1]),
                    "V_end_error_from_common_cs_pct": 100.0 * (float(history["V"][-1]) - V_cs) / V_cs,
                    "q_curve_rmse_vs_mcc_pct": q_rmse,
                    "V_curve_rmse_vs_mcc_pct": V_rmse,
                }
            )
        _write_csv(columns, results_dir / f"matched_{label}_histories.csv")
        save_matched_plot(histories, label.capitalize(), results_dir / f"matched_{label}.png")

    _write_rows(rows, results_dir / "matched_summary.csv")
    metadata = {
        "common_critical_state": {"p_kpa": p_cs, "q_kpa": q_cs, "V": V_cs},
        "state_mappings": mappings,
        "mcc_parameters": asdict(mcc.p),
        "evp_parameters": asdict(evp.p),
        "norsand_parameters": asdict(norsand.p),
    }
    (results_dir / "matched_parameters.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return all_histories, rows, metadata


def run_all_comparisons(results_dir: Path | None = None):
    results_dir = results_dir or Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    _, reference_rows, _ = run_norsand_reference(results_dir)
    _, matched_rows, metadata = run_matched_comparison(results_dir)
    print("NorSand public-reference qualitative verification")
    for row in reference_rows:
        print(
            f"  {row['state']}: q_peak={row['q_peak_kpa']:.3f} kPa, "
            f"q_end={row['q_end_kpa']:.3f} kPa, p_end={row['p_end_kpa']:.3f} kPa"
        )
    print("Matched critical-state comparison")
    for row in matched_rows:
        print(
            f"  {row['state']:5s} {row['model']:7s}: q_peak={row['q_peak_kpa']:.3f}, "
            f"q_end={row['q_end_kpa']:.3f}, q_RMSE={row['q_curve_rmse_vs_mcc_pct']:.3f}%"
        )
    return matched_rows, metadata


if __name__ == "__main__":
    run_all_comparisons()

