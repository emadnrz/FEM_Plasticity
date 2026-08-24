"""Matched loose/dense drained triaxial tests with the 2-D FE solver."""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import numpy as np

from comparison import (
    KAPPA_COMMON,
    LAMBDA_COMMON,
    P_INITIAL,
    STATE_PARAMETERS,
    build_matched_models,
    mcc_drained_first_yield,
    mcc_pc_from_state_parameter,
    norsand_ocr_for_yield_match,
)
from fem import (
    average_stress,
    axisymmetric_specimen_mesh,
    flatten_states,
    initialize_integration_states,
    solve_increment,
    triaxial_displacement_bcs,
    triaxial_tie_groups,
)
from kelln_evp import KellnEVP
from modified_cam_clay import ModifiedCamClay
from norsand import NorSand
from plasticity import invariants
from plotting import save_matched_plot
from triaxial import TriaxialHistory


AXIAL_STRAIN_END = 0.30
INCREMENTS = 120
STRAIN_RATE = 1.0e-3


def _volume(state) -> float:
    if hasattr(state, "V"):
        return float(state.V)
    return 1.0 + float(state.e)


def _fe_models():
    """Use a safe constitutive substep size for the FE Newton iterations."""

    mcc, evp, norsand = build_matched_models()
    return (
        ModifiedCamClay(replace(mcc.p, max_strain_step=5.0e-4)),
        KellnEVP(evp.p),
        NorSand(replace(norsand.p, max_strain_step=5.0e-4)),
    )


def _initial_states(models, psi_state: float):
    mcc, evp, norsand = models
    pc = mcc_pc_from_state_parameter(
        P_INITIAL, psi_state, LAMBDA_COMMON, KAPPA_COMMON
    )
    mcc_state = mcc.initialize_isotropic(P_INITIAL, pc)
    evp_state = evp.initialize_isotropic(P_INITIAL, pc)
    yield_p, yield_q, yield_e = mcc_drained_first_yield(mcc, mcc_state)
    norsand_ocr = norsand_ocr_for_yield_match(
        norsand, P_INITIAL, yield_p, yield_q, yield_e
    )
    norsand_state = norsand.initialize_isotropic(
        P_INITIAL, psi_state, norsand_ocr
    )
    return {
        "MCC": mcc_state,
        "EVP": evp_state,
        "NorSand": norsand_state,
    }, {"mcc_pc": pc, "norsand_OCR": norsand_ocr}


def solve_fe_triaxial(model, state0) -> TriaxialHistory:
    """Run a homogeneous cylindrical TXD test with one axisymmetric Quad4."""

    mesh = axisymmetric_specimen_mesh(radius=1.0, height=2.0)
    states = initialize_integration_states(mesh, state0)
    ties = triaxial_tie_groups(mesh)
    de_axial = AXIAL_STRAIN_END / INCREMENTS
    dt = de_axial / STRAIN_RATE
    total_displacement = np.zeros(mesh.ndof)
    initial_volume = _volume(state0)
    names = (
        "eps_axial", "eps_radial", "eps_v", "p", "q", "eta", "V",
        "radial_stress_error", "newton_iterations", "residual_norm",
    )
    data = {name: np.zeros(INCREMENTS + 1) for name in names}
    data["p"][0] = P_INITIAL
    data["V"][0] = initial_volume
    outer_nodes = np.where(np.isclose(mesh.coordinates[:, 0], 1.0))[0]

    for step in range(1, INCREMENTS + 1):
        result = solve_increment(
            mesh,
            model,
            states,
            triaxial_displacement_bcs(mesh, de_axial),
            dt=dt,
            tolerance=1.0e-5 if isinstance(model, ModifiedCamClay) else 1.0e-3,
            max_iterations=45,
            tie_groups=ties,
        )
        states = result.states
        total_displacement += result.displacement_increment
        integration_states = flatten_states(states)
        stress = average_stress(integration_states)
        p, q, *_ = invariants(stress)
        radial_tension_strain = float(
            np.mean(total_displacement[2 * outer_nodes])
        )
        eps_axial = step * de_axial
        eps_radial = -radial_tension_strain
        data["eps_axial"][step] = eps_axial
        data["eps_radial"][step] = eps_radial
        data["eps_v"][step] = eps_axial + 2.0 * eps_radial
        data["p"][step] = p
        data["q"][step] = q
        data["eta"][step] = q / p
        data["V"][step] = float(np.mean([_volume(state) for state in integration_states]))
        data["radial_stress_error"][step] = max(
            abs(float(stress[0]) - P_INITIAL),
            abs(float(stress[1]) - P_INITIAL),
        )
        data["newton_iterations"][step] = result.iterations
        data["residual_norm"][step] = result.residual_norm
    return TriaxialHistory(data)


def _material_point_history(model, state0):
    if isinstance(model, KellnEVP):
        return model.solve_triaxial(
            state0, AXIAL_STRAIN_END, "drained", INCREMENTS, STRAIN_RATE
        )
    return model.solve_triaxial(
        state0, AXIAL_STRAIN_END, "drained", INCREMENTS, 5.0e-5
    )


def _write_histories(histories, path: Path) -> None:
    columns = {"eps_axial": next(iter(histories.values()))["eps_axial"]}
    for model_name, history in histories.items():
        tag = model_name.lower()
        for quantity in (
            "p", "q", "eps_v", "eta", "V", "radial_stress_error",
            "newton_iterations", "residual_norm",
        ):
            columns[f"{tag}_{quantity}"] = history[quantity]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(zip(*columns.values(), strict=True))


def run_fe_demo(results_dir: Path | None = None):
    """Run both states with all three material files and compare responses."""

    results_dir = results_dir or Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    models = _fe_models()
    named_models = dict(zip(("MCC", "EVP", "NorSand"), models, strict=True))
    all_histories = {}
    rows = []

    for state_name, psi_state in STATE_PARAMETERS.items():
        initial_states, mapping = _initial_states(models, psi_state)
        histories = {}
        material_histories = {}
        for model_name, model in named_models.items():
            histories[model_name] = solve_fe_triaxial(
                model, initial_states[model_name]
            )
            material_histories[model_name] = _material_point_history(
                model, initial_states[model_name]
            )
        all_histories[state_name] = histories
        mcc_history = histories["MCC"]
        q_scale = float(np.max(mcc_history["q"]))
        volume_scale = max(float(np.max(np.abs(mcc_history["eps_v"]))), 0.01)

        for model_name, history in histories.items():
            material = material_histories[model_name]
            rows.append(
                {
                    "state": state_name,
                    "model": model_name,
                    "initial_state_parameter": psi_state,
                    **mapping,
                    "q_peak_kpa": float(np.max(history["q"])),
                    "q_end_kpa": float(history["q"][-1]),
                    "p_end_kpa": float(history["p"][-1]),
                    "eps_v_end": float(history["eps_v"][-1]),
                    "q_curve_rmse_vs_mcc_pct": 100.0
                    * float(np.sqrt(np.mean((history["q"] - mcc_history["q"]) ** 2)))
                    / q_scale,
                    "eps_v_curve_rmse_vs_mcc_pct": 100.0
                    * float(
                        np.sqrt(np.mean((history["eps_v"] - mcc_history["eps_v"]) ** 2))
                    )
                    / volume_scale,
                    "q_fe_vs_material_rmse_pct": 100.0
                    * float(np.sqrt(np.mean((history["q"] - material["q"]) ** 2)))
                    / max(float(np.max(material["q"])), 1.0),
                    "eps_v_fe_vs_material_rmse_pct": 100.0
                    * float(
                        np.sqrt(np.mean((history["eps_v"] - material["eps_v"]) ** 2))
                    )
                    / max(float(np.max(np.abs(material["eps_v"]))), 0.01),
                    "max_radial_stress_error_kpa": float(
                        np.max(history["radial_stress_error"])
                    ),
                    "max_newton_iterations": int(np.max(history["newton_iterations"])),
                    "max_residual_norm": float(np.max(history["residual_norm"])),
                }
            )

        _write_histories(
            histories, results_dir / f"fe2d_drained_{state_name}_histories.csv"
        )
        save_matched_plot(
            histories,
            f"2-D axisymmetric FE, drained {state_name}",
            results_dir / f"fe2d_drained_{state_name}.png",
        )

    summary_path = results_dir / "fe2d_drained_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    if max(row["q_fe_vs_material_rmse_pct"] for row in rows) > 2.0:
        raise AssertionError("2-D FE/material-point q mismatch exceeded 2%")
    if max(row["max_radial_stress_error_kpa"] for row in rows) > 2.5:
        raise AssertionError("constant confining-pressure error exceeded 2.5 kPa")

    print("2-D axisymmetric drained triaxial comparison")
    for row in rows:
        print(
            f"  {row['state']:5s} {row['model']:7s}: "
            f"q_peak={row['q_peak_kpa']:.2f}, q_end={row['q_end_kpa']:.2f}, "
            f"q-RMSE vs MCC={row['q_curve_rmse_vs_mcc_pct']:.2f}%"
        )
    return all_histories, rows


if __name__ == "__main__":
    run_fe_demo()
