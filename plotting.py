"""Dependency-light PNG plots for constitutive comparisons."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _font(size: int, bold: bool = False):
    path = Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def _limits(values: list[np.ndarray]) -> tuple[float, float]:
    all_values = np.concatenate([np.asarray(v, dtype=float) for v in values])
    lo, hi = float(np.min(all_values)), float(np.max(all_values))
    pad = max(0.06 * (hi - lo), 1.0e-7)
    return lo - pad, hi + pad


def _panel(draw, box, curves, xlabel, ylabel, title):
    left, top, right, bottom = box
    x0, y0, x1, y1 = left + 84, top + 58, right - 28, bottom - 70
    xmin, xmax = _limits([c[0] for c in curves])
    ymin, ymax = _limits([c[1] for c in curves])
    axis, grid = (35, 42, 50), (220, 225, 232)
    for i in range(6):
        xx = x0 + i * (x1 - x0) / 5
        yy = y0 + i * (y1 - y0) / 5
        draw.line((xx, y0, xx, y1), fill=grid)
        draw.line((x0, yy, x1, yy), fill=grid)
        draw.text((xx - 20, y1 + 10), f"{xmin+i*(xmax-xmin)/5:.3g}", font=_font(14), fill=axis)
        draw.text((x0 - 72, yy - 8), f"{ymax-i*(ymax-ymin)/5:.4g}", font=_font(14), fill=axis)
    draw.line((x0, y0, x0, y1), fill=axis, width=2)
    draw.line((x0, y1, x1, y1), fill=axis, width=2)

    def points(x, y):
        return [
            (
                x0 + (float(a) - xmin) / (xmax - xmin) * (x1 - x0),
                y1 - (float(b) - ymin) / (ymax - ymin) * (y1 - y0),
            )
            for a, b in zip(x, y, strict=True)
        ]

    legend_y = y0 + 5
    for x, y, color, label, width in curves:
        draw.line(points(x, y), fill=color, width=width, joint="curve")
        draw.line((x1 - 160, legend_y + 8, x1 - 128, legend_y + 8), fill=color, width=width)
        draw.text((x1 - 120, legend_y), label, font=_font(14), fill=axis)
        legend_y += 24
    draw.text((x0, top + 14), title, font=_font(20, True), fill=axis)
    draw.text(((x0 + x1) / 2 - 45, y1 + 42), xlabel, font=_font(17), fill=axis)
    draw.text((left + 5, top + 30), ylabel, font=_font(17), fill=axis)


def save_matched_plot(histories, state_label: str, path: Path) -> None:
    colors = {"MCC": (20, 25, 30), "EVP": (45, 110, 185), "NorSand": (205, 95, 45)}
    widths = {"MCC": 5, "EVP": 3, "NorSand": 3}
    image = Image.new("RGB", (1650, 570), "white")
    draw = ImageDraw.Draw(image)
    panels = (
        ("eps_axial", "q", "axial strain", "q (kPa)", f"{state_label}: stress-strain"),
        ("eps_axial", "eps_v", "axial strain", "volumetric strain", "volume response"),
        ("eps_axial", "eta", "axial strain", "q/p'", "critical stress-ratio approach"),
    )
    for i, (xname, yname, xlabel, ylabel, title) in enumerate(panels):
        curves = [
            (h[xname], h[yname], colors[name], name, widths[name])
            for name, h in histories.items()
        ]
        _panel(draw, (550 * i, 0, 550 * (i + 1), 570), curves, xlabel, ylabel, title)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, "PNG", optimize=True)


def save_norsand_verification_plot(histories, path: Path) -> None:
    image = Image.new("RGB", (1200, 570), "white")
    draw = ImageDraw.Draw(image)
    colors = {"dense": (35, 105, 175), "loose": (205, 90, 50)}
    for i, (xname, xlabel, title) in enumerate(
        (("eps_axial", "axial strain", "NorSand undrained response"),
         ("p", "p' (kPa)", "NorSand effective stress paths"))
    ):
        curves = [
            (h[xname], h["q"], colors[name], name, 4)
            for name, h in histories.items()
        ]
        _panel(draw, (600 * i, 0, 600 * (i + 1), 570), curves, xlabel, "q (kPa)", title)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, "PNG", optimize=True)


def save_norsand_figure1_plot(histories, references, path: Path) -> None:
    """Plot the four panels used for the Marinelli et al. Figure 1 check."""

    image = Image.new("RGB", (1400, 1120), "white")
    draw = ImageDraw.Draw(image)
    dense = (36, 102, 170)
    loose = (205, 82, 48)
    reference = (90, 96, 104)

    txu_dense = histories[("undrained", "dense")]
    txu_loose = histories[("undrained", "loose")]
    txd_dense = histories[("drained", "dense")]
    txd_loose = histories[("drained", "loose")]
    # Figure 1 clips the dense TXU curve where it reaches q=600 kPa.
    txu_dense_visible = txu_dense["eps_axial"] <= 0.135 + 1.0e-12

    panels = [
        (
            (0, 0, 700, 560),
            [
                (references["txu_pq_dense"][0], references["txu_pq_dense"][1], reference, "paper dense", 7),
                (references["txu_pq_loose"][0], references["txu_pq_loose"][1], reference, "paper loose", 7),
                (txu_dense["p"][txu_dense_visible], txu_dense["q"][txu_dense_visible], dense, "computed dense", 3),
                (txu_loose["p"], txu_loose["q"], loose, "computed loose", 3),
            ],
            "p' (kPa)", "q (kPa)", "(a) TXU effective-stress paths",
        ),
        (
            (700, 0, 1400, 560),
            [
                (references["txu_q_dense"][0], references["txu_q_dense"][1], reference, "paper dense", 7),
                (references["txu_q_loose"][0], references["txu_q_loose"][1], reference, "paper loose", 7),
                (txu_dense["eps_axial"][txu_dense_visible], txu_dense["q"][txu_dense_visible], dense, "computed dense", 3),
                (txu_loose["eps_axial"], txu_loose["q"], loose, "computed loose", 3),
            ],
            "axial strain", "q (kPa)", "(b) TXU stress-strain",
        ),
        (
            (0, 560, 700, 1120),
            [
                (references["txd_q_dense"][0], references["txd_q_dense"][1], reference, "paper dense", 7),
                (references["txd_q_loose"][0], references["txd_q_loose"][1], reference, "paper loose", 7),
                (txd_dense["eps_axial"], txd_dense["q"], dense, "computed dense", 3),
                (txd_loose["eps_axial"], txd_loose["q"], loose, "computed loose", 3),
            ],
            "axial strain", "q (kPa)", "(c) TXD stress-strain",
        ),
        (
            (700, 560, 1400, 1120),
            [
                (references["txd_ev_dense"][0], references["txd_ev_dense"][1], reference, "paper dense", 7),
                (references["txd_ev_loose"][0], references["txd_ev_loose"][1], reference, "paper loose", 7),
                (txd_dense["eps_axial"], txd_dense["eps_v"], dense, "computed dense", 3),
                (txd_loose["eps_axial"], txd_loose["eps_v"], loose, "computed loose", 3),
            ],
            "axial strain", "volumetric strain", "(d) TXD volume response",
        ),
    ]
    for box, curves, xlabel, ylabel, title in panels:
        _panel(draw, box, curves, xlabel, ylabel, title)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, "PNG", optimize=True)

