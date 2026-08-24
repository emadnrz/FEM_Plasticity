"""Reproduce Figure 1 of Marinelli et al. (Geo-Congress 2022).

The reference ordinates below were digitized from the red VBA curves in the
user-supplied PDF.  They are plot-resolution data, not author-supplied tables,
so normalized curve errors are more meaningful than excessive decimal-place
comparisons.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from norsand import NorSand, NorSandParameters
from plotting import save_norsand_figure1_plot


def _array(values) -> np.ndarray:
    return np.asarray(values, dtype=float)


FIGURE1_REFERENCE = {
    "txu_q_dense": (
        _array([0.0, 0.020, 0.030, 0.040, 0.050, 0.075, 0.100, 0.125, 0.135]),
        _array([0.0, 231.2595, 244.0061, 285.8877, 333.2322, 393.3232, 494.3854, 553.1108, 591.8058]),
    ),
    "txu_q_loose": (
        _array([0.0, 0.005, 0.010, 0.020, 0.030, 0.040, 0.050, 0.075, 0.100, 0.125, 0.150, 0.175, 0.200]),
        _array([0.0, 95.5994, 96.5099, 69.1958, 65.5539, 60.0910, 60.0910, 61.0015, 62.8225, 63.7329, 64.1882, 64.6434, 65.5539]),
    ),
    "txd_q_dense": (
        _array([0.0, 0.010, 0.020, 0.030, 0.040, 0.050, 0.075, 0.100, 0.125, 0.150, 0.175, 0.195]),
        _array([0.0, 238.6606, 427.3973, 519.9391, 594.2161, 614.9163, 722.0700, 756.1644, 771.9939, 776.8645, 774.4292, 769.5586]),
    ),
    "txd_q_loose": (
        _array([0.0, 0.010, 0.020, 0.030, 0.040, 0.050, 0.075, 0.100, 0.125, 0.150, 0.175, 0.195]),
        _array([0.0, 20.7002, 73.0594, 92.5419, 148.5540, 207.0015, 280.0609, 373.8204, 432.2679, 477.3212, 513.8508, 541.8569]),
    ),
    "txd_ev_dense": (
        _array([0.0, 0.010, 0.020, 0.030, 0.040, 0.050, 0.075, 0.100, 0.125, 0.150, 0.175, 0.195]),
        _array([0.0, 0.0050, 0.0123, 0.0123, 0.0147, 0.0149, 0.0128, 0.0089, 0.0048, 0.0008, -0.0034, -0.0069]),
    ),
    "txd_ev_loose": (
        _array([0.0, 0.010, 0.020, 0.030, 0.040, 0.050, 0.075, 0.100, 0.125, 0.150, 0.175, 0.195]),
        _array([0.0, 0.0058, 0.0145, 0.0268, 0.0297, 0.0375, 0.0543, 0.0633, 0.0700, 0.0759, 0.0823, 0.0852]),
    ),
    # Approximate points digitized from panel (a); the other panels are used
    # for the quantitative pointwise metrics.
    "txu_pq_dense": (
        _array([250.0, 238.0, 218.0, 188.0, 174.0, 158.0, 175.0, 212.0, 290.0, 368.0, 425.0]),
        _array([0.0, 25.0, 100.0, 150.0, 175.0, 200.0, 250.0, 300.0, 400.0, 500.0, 592.0]),
    ),
    "txu_pq_loose": (
        _array([250.0, 238.0, 202.0, 184.0, 136.0, 122.0, 80.0, 65.0, 58.0]),
        _array([0.0, 25.0, 50.0, 75.0, 90.0, 99.0, 90.0, 75.0, 65.0]),
    ),
}


def figure1_parameters(*, softening: bool) -> NorSandParameters:
    """Parameter set A from the paper, in kPa and dimensionless units."""

    return NorSandParameters(
        G_ref=20_000.0,
        p_ref=100.0,
        m=0.6,
        nu=0.10,
        Gamma=1.115,
        lambda_c=0.076,
        M_tc=1.40,
        N_coupling=0.30,
        chi_tc=2.5,
        H0=30.0,
        Hy=200.0,
        Z=softening,
        softening_formulation="paper_2022",
    )


def _simulate(max_internal_step: float):
    histories = {}
    # PLAXIS constrains the S term to undrained loading; it is identically zero
    # in TXD even though Table 1 lists the material flag S=1.
    for mode, softening in (("undrained", True), ("drained", False)):
        model = NorSand(figure1_parameters(softening=softening))
        for state_name, psi0 in (("dense", -0.12), ("loose", 0.12)):
            histories[(mode, state_name)] = model.solve_triaxial(
                model.initialize_isotropic(250.0, psi0, OCR=1.0),
                axial_strain_end=0.20,
                mode=mode,
                increments=200,
                max_internal_strain_step=max_internal_step,
            )
    return histories


def _write_histories(histories, path: Path) -> None:
    eps = histories[("undrained", "dense")]["eps_axial"]
    columns = {"eps_axial": eps}
    for (mode, state_name), history in histories.items():
        prefix = f"{mode}_{state_name}"
        for quantity in ("p", "q", "eps_v", "eta", "hardening", "psi"):
            columns[f"{prefix}_{quantity}"] = history[quantity]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(zip(*columns.values(), strict=True))


def _write_reference(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("series", "x", "y", "source"))
        for name, (x, y) in FIGURE1_REFERENCE.items():
            for xv, yv in zip(x, y, strict=True):
                writer.writerow((name, xv, yv, "digitized from attached paper Figure 1"))


def _curve_metric(history, x_ref, y_ref, quantity: str, scale: float) -> dict[str, float]:
    calculated = np.interp(x_ref, history["eps_axial"], history[quantity])
    error = calculated - y_ref
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "nrmse_pct": 100.0 * float(np.sqrt(np.mean(error**2))) / scale,
        "max_normalized_error_pct": 100.0 * float(np.max(np.abs(error))) / scale,
    }


def _summary_rows(histories, coarse_histories):
    definitions = (
        ("undrained", "dense", "q", "txu_q_dense", 600.0),
        ("undrained", "loose", "q", "txu_q_loose", 600.0),
        ("drained", "dense", "q", "txd_q_dense", 800.0),
        ("drained", "loose", "q", "txd_q_loose", 800.0),
        ("drained", "dense", "eps_v", "txd_ev_dense", 0.12),
        ("drained", "loose", "eps_v", "txd_ev_loose", 0.12),
    )
    rows = []
    for mode, state_name, quantity, reference_name, scale in definitions:
        history = histories[(mode, state_name)]
        x_ref, y_ref = FIGURE1_REFERENCE[reference_name]
        metrics = _curve_metric(history, x_ref, y_ref, quantity, scale)
        coarse = coarse_histories[(mode, state_name)]
        coarse_values = np.interp(history["eps_axial"], coarse["eps_axial"], coarse[quantity])
        convergence = 100.0 * float(
            np.sqrt(np.mean((history[quantity] - coarse_values) ** 2))
        ) / scale
        rows.append(
            {
                "mode": mode,
                "state": state_name,
                "quantity": quantity,
                "reference_series": reference_name,
                **metrics,
                "step_convergence_nrmse_pct": convergence,
            }
        )
    return rows


def run_figure1_verification(results_dir: Path | None = None):
    results_dir = results_dir or Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    histories = _simulate(max_internal_step=2.0e-5)
    coarse_histories = _simulate(max_internal_step=5.0e-5)
    rows = _summary_rows(histories, coarse_histories)
    if max(row["nrmse_pct"] for row in rows) > 5.0:
        raise AssertionError("Figure 1 curve NRMSE exceeded the 5% digitized-plot tolerance")
    if max(row["step_convergence_nrmse_pct"] for row in rows) > 0.1:
        raise AssertionError("Figure 1 integration step sensitivity exceeded 0.1%")

    _write_histories(histories, results_dir / "norsand_figure1_histories.csv")
    _write_reference(results_dir / "norsand_figure1_digitized.csv")
    with (results_dir / "norsand_figure1_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_norsand_figure1_plot(
        histories, FIGURE1_REFERENCE, results_dir / "norsand_figure1_verification.png"
    )

    print("Marinelli et al. Figure 1 digitized-curve verification")
    for row in rows:
        print(
            f"  {row['mode']:10s} {row['state']:5s} {row['quantity']:5s}: "
            f"NRMSE={row['nrmse_pct']:.3f}%, "
            f"step check={row['step_convergence_nrmse_pct']:.4f}%"
        )
    return histories, rows


if __name__ == "__main__":
    run_figure1_verification()
