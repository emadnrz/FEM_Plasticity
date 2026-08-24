"""Generate unified-calibration and zero-dilatancy evidence for the report.

This is a material-point diagnostic.  The production constitutive files are
not modified: ``ZeroDilatancyNorSand`` changes only the ideal-triaxial rate
equations so the consequence of imposing D=0 can be shown explicitly.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from comparison import (
    GAMMA_E_COMMON,
    KAPPA_COMMON,
    LAMBDA_COMMON,
    M_COMMON,
    NU_COMMON,
    P_INITIAL,
    P_REF_COMMON,
    STATE_PARAMETERS,
    build_matched_models,
    mcc_drained_first_yield,
    mcc_pc_from_state_parameter,
    norsand_ocr_for_yield_match,
)
from norsand import NorSand, NorSandParameters, NorSandState
from plotting import _panel


RESULTS = Path(__file__).resolve().parent / "results"
UNIFIED_PARAMETERS = {
    "chi_tc": 6.0,
    "N_coupling": 0.10,
    "H0": 44.0,
    "Hy": 720.0,
}


class ZeroDilatancyNorSand(NorSand):
    """Diagnostic NorSand triaxial driver with plastic D forced to zero."""

    def _triaxial_plastic_rates(self, y: np.ndarray, mode: str) -> np.ndarray:
        p, q, e, image_pressure = map(float, y)
        stress = self._triaxial_stress(p, q)
        state = NorSandState(stress, e, image_pressure, np.zeros(6))
        point = self.point(stress, e, image_pressure)
        D = self.elastic_matrix(state)
        K = float((D[0, 0] + 2.0 * D[0, 1]) / 3.0)
        G = float(D[3, 3])
        fp, fq, fe, fpi = self._triaxial_derivatives(y)
        dpi_deq = self._image_pressure_rate(state, point)
        plastic_dilatancy = 0.0

        if mode == "undrained":
            denominator = (
                fp * K * plastic_dilatancy + fq * 3.0 * G - fpi * dpi_deq
            )
            plastic_q = fq * 3.0 * G / denominator
            volumetric_rate = 0.0
        elif mode == "drained":
            matrix = np.array(
                [
                    [
                        3.0 * K + G,
                        3.0 * (G - K * plastic_dilatancy),
                    ],
                    [
                        (fp + 3.0 * fq) * K - fe * (1.0 + e),
                        -(fp + 3.0 * fq) * K * plastic_dilatancy
                        + fpi * dpi_deq,
                    ],
                ]
            )
            volumetric_rate, plastic_q = np.linalg.solve(
                matrix, np.array([3.0 * G, 0.0])
            )
        else:
            raise ValueError("mode must be drained or undrained")

        equivalent_rate = 1.0 - volumetric_rate / 3.0
        dp = K * (volumetric_rate - plastic_dilatancy * plastic_q)
        dq = 3.0 * G * (equivalent_rate - plastic_q)
        return np.array(
            [
                dp,
                dq,
                -(1.0 + e) * volumetric_rate,
                dpi_deq * plastic_q,
            ]
        )


def _make_norsand(zero_dilatancy: bool = False) -> NorSand:
    cls = ZeroDilatancyNorSand if zero_dilatancy else NorSand
    return cls(
        NorSandParameters(
            Gamma=GAMMA_E_COMMON,
            lambda_c=LAMBDA_COMMON,
            M_tc=M_COMMON,
            chi_tc=UNIFIED_PARAMETERS["chi_tc"],
            N_coupling=UNIFIED_PARAMETERS["N_coupling"],
            H0=UNIFIED_PARAMETERS["H0"],
            Hy=UNIFIED_PARAMETERS["Hy"],
            G_ref=13_400.0,
            m=1.0,
            nu=NU_COMMON,
            p_ref=P_REF_COMMON,
            max_strain_step=1.0e-4,
        )
    )


def _rmse_percent(computed, reference, floor: float = 1.0) -> float:
    computed = np.asarray(computed, dtype=float)
    reference = np.asarray(reference, dtype=float)
    scale = max(float(np.max(np.abs(reference))), floor)
    return 100.0 * float(np.sqrt(np.mean((computed - reference) ** 2))) / scale


def _write_history(histories, path: Path) -> None:
    columns: dict[str, np.ndarray] = {
        "eps_axial": histories["MCC"]["eps_axial"]
    }
    for name, history in histories.items():
        tag = name.lower().replace(" ", "_").replace("=", "")
        for field in ("p", "q", "eps_v", "u", "eta"):
            columns[f"{tag}_{field}"] = history[field]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(zip(*columns.values(), strict=True))


def _save_plot(state_label: str, drained, undrained, path: Path) -> None:
    colors = {
        "MCC": (25, 30, 36),
        "EVP": (42, 107, 184),
        "NS unified": (205, 91, 43),
        "NS D=0": (117, 77, 156),
    }
    widths = {"MCC": 5, "EVP": 3, "NS unified": 4, "NS D=0": 3}
    image = Image.new("RGB", (1400, 1000), "white")
    draw = ImageDraw.Draw(image)
    panels = (
        (drained, "eps_axial", "q", "axial strain", "q (kPa)", "drained q"),
        (
            drained,
            "eps_axial",
            "eps_v",
            "axial strain",
            "volumetric strain",
            "drained volume",
        ),
        (
            undrained,
            "eps_axial",
            "q",
            "axial strain",
            "q (kPa)",
            "undrained q",
        ),
        (
            undrained,
            "eps_axial",
            "u",
            "axial strain",
            "excess u (kPa)",
            "undrained water pressure",
        ),
    )
    for index, (histories, xfield, yfield, xlabel, ylabel, title) in enumerate(panels):
        row, column = divmod(index, 2)
        curves = [
            (
                history[xfield],
                history[yfield],
                colors[name],
                name,
                widths[name],
            )
            for name, history in histories.items()
        ]
        _panel(
            draw,
            (700 * column, 500 * row, 700 * (column + 1), 500 * (row + 1)),
            curves,
            xlabel,
            ylabel,
            f"{state_label}: {title}",
        )
    image.save(path, "PNG", optimize=True)


def run_analysis(increments: int = 100):
    RESULTS.mkdir(parents=True, exist_ok=True)
    mcc, evp, _ = build_matched_models()
    rows: list[dict[str, object]] = []
    all_histories = {}

    for state_label, state_parameter in STATE_PARAMETERS.items():
        pc = mcc_pc_from_state_parameter(
            P_INITIAL, state_parameter, LAMBDA_COMMON, KAPPA_COMMON
        )
        mcc_state = mcc.initialize_isotropic(P_INITIAL, pc)
        evp_state = evp.initialize_isotropic(P_INITIAL, pc)
        yield_p, yield_q, yield_e = mcc_drained_first_yield(mcc, mcc_state)
        unified = _make_norsand(False)
        zero = _make_norsand(True)
        ocr = norsand_ocr_for_yield_match(
            unified, P_INITIAL, yield_p, yield_q, yield_e
        )
        unified_state = unified.initialize_isotropic(
            P_INITIAL, state_parameter, ocr
        )
        zero_state = zero.initialize_isotropic(P_INITIAL, state_parameter, ocr)

        histories_by_mode = {}
        for mode, end in (("drained", 0.30), ("undrained", 0.15)):
            histories = {
                "MCC": mcc.solve_triaxial(
                    mcc_state, end, mode, increments, 1.0e-4
                ),
                "EVP": evp.solve_triaxial(
                    evp_state,
                    end,
                    mode,
                    increments,
                    1.0e-3,
                ),
                "NS unified": unified.solve_triaxial(
                    unified_state, end, mode, increments, 1.0e-4
                ),
                "NS D=0": zero.solve_triaxial(
                    zero_state, end, mode, increments, 1.0e-4
                ),
            }
            histories_by_mode[mode] = histories
            reference = histories["MCC"]
            secondary = "eps_v" if mode == "drained" else "u"
            for name, history in histories.items():
                rows.append(
                    {
                        "state": state_label,
                        "mode": mode,
                        "model": name,
                        "state_parameter": state_parameter,
                        "norsand_OCR_unified": ocr if name.startswith("NS") else "",
                        "q_peak_kpa": float(np.max(history["q"])),
                        "q_end_kpa": float(history["q"][-1]),
                        "p_end_kpa": float(history["p"][-1]),
                        "eps_v_end": float(history["eps_v"][-1]),
                        "u_end_kpa": float(history["u"][-1]),
                        "q_rmse_vs_mcc_pct": _rmse_percent(
                            history["q"], reference["q"]
                        ),
                        f"{secondary}_rmse_vs_mcc_pct": _rmse_percent(
                            history[secondary],
                            reference[secondary],
                            1.0e-3 if secondary == "eps_v" else 1.0,
                        ),
                    }
                )
            _write_history(
                histories,
                RESULTS / f"unified_{state_label}_{mode}_histories.csv",
            )
        all_histories[state_label] = histories_by_mode
        _save_plot(
            state_label.capitalize(),
            histories_by_mode["drained"],
            histories_by_mode["undrained"],
            RESULTS / f"model_equivalence_{state_label}.png",
        )

    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (RESULTS / "unified_matching_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return all_histories, rows


if __name__ == "__main__":
    _, summary = run_analysis()
    for row in summary:
        if row["model"] in ("NS unified", "NS D=0"):
            secondary = "eps_v" if row["mode"] == "drained" else "u"
            print(
                f"{row['state']:5s} {row['mode']:9s} {row['model']:10s}: "
                f"q RMSE={row['q_rmse_vs_mcc_pct']:.2f}%, "
                f"{secondary} RMSE={row[f'{secondary}_rmse_vs_mcc_pct']:.2f}%"
            )
