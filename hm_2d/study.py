"""Dense/loose undrained comparison and verification workflow."""

from __future__ import annotations

from dataclasses import asdict
import csv
import json
from pathlib import Path

import numpy as np

from triaxial import solve_triaxial as solve_material_point

from .materials import (
    CommonCalibration,
    MaterialCase,
    create_evp_case,
    create_mcc_case,
    create_norsand_case,
)
from .materials.mcc_model import drained_first_yield
from .mesh import structured_cylinder_mesh
from .plotting import save_undrained_comparison
from .triaxial import HMTriaxialHistory, run_undrained_triaxial
from .verification import run_basic_verifications


STATE_PARAMETERS = {"dense": -0.05, "loose": 0.02}


def _write_rows(rows: list[dict[str, object]], path: Path) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_columns(columns: dict[str, np.ndarray], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(zip(*columns.values(), strict=True))


def _normalized_rmse(computed, reference) -> float:
    computed = np.asarray(computed, dtype=float)
    reference = np.asarray(reference, dtype=float)
    scale = max(float(np.max(np.abs(reference))), 1.0)
    return 100.0 * float(np.sqrt(np.mean((computed - reference) ** 2))) / scale


def build_cases(
    calibration: CommonCalibration, state_parameter: float
) -> tuple[MaterialCase, MaterialCase, MaterialCase]:
    mcc = create_mcc_case(calibration, state_parameter)
    evp = create_evp_case(calibration, state_parameter)
    norsand = create_norsand_case(
        calibration, state_parameter, drained_first_yield(mcc)
    )
    return mcc, evp, norsand


def run_undrained_study(
    results_directory: Path | None = None,
    axial_strain_end: float = 0.15,
    increments: int = 100,
) -> tuple[dict[str, dict[str, HMTriaxialHistory]], list[dict[str, object]]]:
    """Run all six HM tests and write plots, histories, and verification data."""

    results = results_directory or Path(__file__).resolve().parent / "results"
    results.mkdir(parents=True, exist_ok=True)
    calibration = CommonCalibration()
    mesh = structured_cylinder_mesh(
        radius=1.0, height=2.0, radial_elements=1, axial_elements=1
    )
    all_histories: dict[str, dict[str, HMTriaxialHistory]] = {}
    summary: list[dict[str, object]] = []
    parameter_data: dict[str, object] = {"common": asdict(calibration), "states": {}}

    basic = run_basic_verifications()
    (results / "basic_verification.json").write_text(
        json.dumps(basic, indent=2), encoding="utf-8"
    )

    for state_label, state_parameter in STATE_PARAMETERS.items():
        cases = build_cases(calibration, state_parameter)
        histories: dict[str, HMTriaxialHistory] = {}
        references = {}
        parameter_data["states"][state_label] = {
            case.name: case.metadata for case in cases
        }
        for case in cases:
            histories[case.name] = run_undrained_triaxial(
                mesh,
                case,
                axial_strain_end=axial_strain_end,
                increments=increments,
                strain_rate=calibration.strain_rate,
            )
            references[case.name] = solve_material_point(
                case.model,
                case.initial_state,
                axial_strain_end,
                "undrained",
                increments,
                calibration.strain_rate,
            )
        all_histories[state_label] = histories
        mcc_history = histories["MCC"]
        columns: dict[str, np.ndarray] = {"eps_axial": mcc_history["eps_axial"]}

        for case in cases:
            name = case.name
            history = histories[name]
            reference = references[name]
            tag = name.lower()
            for field in (
                "q",
                "p_effective",
                "pore_pressure",
                "eps_v",
                "radial_total_stress",
            ):
                columns[f"{tag}_fe_{field}"] = history[field]
            columns[f"{tag}_material_q"] = reference["q"]
            columns[f"{tag}_material_p_effective"] = reference["p"]
            columns[f"{tag}_material_pore_pressure"] = reference["u"]

            summary.append(
                {
                    "state": state_label,
                    "model": name,
                    "initial_state_parameter": state_parameter,
                    "q_peak_kpa": float(np.max(history["q"])),
                    "q_end_kpa": float(history["q"][-1]),
                    "p_effective_end_kpa": float(history["p_effective"][-1]),
                    "pore_pressure_min_kpa": float(np.min(history["pore_pressure"])),
                    "pore_pressure_max_kpa": float(np.max(history["pore_pressure"])),
                    "pore_pressure_end_kpa": float(history["pore_pressure"][-1]),
                    "q_fe_vs_material_rmse_pct": _normalized_rmse(
                        history["q"], reference["q"]
                    ),
                    "u_fe_vs_material_rmse_pct": _normalized_rmse(
                        history["pore_pressure"], reference["u"]
                    ),
                    "q_curve_rmse_vs_mcc_pct": _normalized_rmse(
                        history["q"], mcc_history["q"]
                    ),
                    "u_curve_rmse_vs_mcc_pct": _normalized_rmse(
                        history["pore_pressure"], mcc_history["pore_pressure"]
                    ),
                    "max_abs_volumetric_strain": float(
                        np.max(np.abs(history["eps_v"]))
                    ),
                    "max_cell_pressure_error_kpa": float(
                        np.max(
                            np.abs(
                                history["radial_total_stress"]
                                - calibration.initial_effective_pressure
                            )
                        )
                    ),
                    "max_mass_residual_norm": float(
                        np.max(history["mass_residual"])
                    ),
                    "max_momentum_residual_norm": float(
                        np.max(history["equilibrium_residual"])
                    ),
                    "max_newton_iterations": int(
                        np.max(history["newton_iterations"])
                    ),
                }
            )

        _write_columns(columns, results / f"undrained_{state_label}_histories.csv")
        save_undrained_comparison(
            histories,
            state_label.capitalize(),
            results / f"undrained_{state_label}.png",
        )

    _write_rows(summary, results / "undrained_summary.csv")
    (results / "matched_parameters.json").write_text(
        json.dumps(parameter_data, indent=2), encoding="utf-8"
    )

    interpolation = basic["interpolation_and_quadrature"]
    elastic = basic["elastic_undrained_patch"]
    if max(interpolation.values()) > 1.0e-12:
        raise AssertionError("shape-function/quadrature verification failed")
    if max(elastic.values()) > 1.0e-8:
        raise AssertionError("closed-form elastic u-p patch verification failed")
    if max(float(row["q_fe_vs_material_rmse_pct"]) for row in summary) > 2.0:
        raise AssertionError("FE/material-point q verification exceeded 2%")
    if max(float(row["u_fe_vs_material_rmse_pct"]) for row in summary) > 3.0:
        raise AssertionError("FE/material-point pore-pressure verification exceeded 3%")
    if max(float(row["max_cell_pressure_error_kpa"]) for row in summary) > 1.0:
        raise AssertionError("constant total cell pressure error exceeded 1 kPa")
    if max(float(row["q_curve_rmse_vs_mcc_pct"]) for row in summary) > 4.0:
        raise AssertionError("cross-model q matching exceeded 4%")
    if max(float(row["u_curve_rmse_vs_mcc_pct"]) for row in summary) > 8.0:
        raise AssertionError("cross-model pore-pressure matching exceeded 8%")
    return all_histories, summary


def main() -> None:
    _, rows = run_undrained_study()
    print("2-D axisymmetric hydro-mechanical undrained comparison")
    for row in rows:
        print(
            f"  {row['state']:5s} {row['model']:7s}: "
            f"q_peak={row['q_peak_kpa']:.2f}, q_end={row['q_end_kpa']:.2f}, "
            f"u_end={row['pore_pressure_end_kpa']:.2f}, "
            f"FE q-RMSE={row['q_fe_vs_material_rmse_pct']:.2f}%, "
            f"FE u-RMSE={row['u_fe_vs_material_rmse_pct']:.2f}%"
        )
    print(f"Results written to {Path(__file__).resolve().parent / 'results'}")


if __name__ == "__main__":
    main()
