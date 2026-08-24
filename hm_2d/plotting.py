"""Dependency-light plots for the undrained HM comparison."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _font(size: int, bold: bool = False):
    path = Path(
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"
    )
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def _limits(values) -> tuple[float, float]:
    joined = np.concatenate([np.asarray(value, dtype=float) for value in values])
    low, high = float(np.min(joined)), float(np.max(joined))
    padding = max(0.07 * (high - low), 1.0e-9)
    return low - padding, high + padding


def _panel(draw, box, curves, xlabel: str, ylabel: str, title: str) -> None:
    left, top, right, bottom = box
    x0, y0, x1, y1 = left + 92, top + 58, right - 30, bottom - 68
    xmin, xmax = _limits([curve[0] for curve in curves])
    ymin, ymax = _limits([curve[1] for curve in curves])
    ink, grid = (31, 39, 48), (222, 227, 233)
    for index in range(6):
        x = x0 + index * (x1 - x0) / 5.0
        y = y0 + index * (y1 - y0) / 5.0
        draw.line((x, y0, x, y1), fill=grid)
        draw.line((x0, y, x1, y), fill=grid)
        draw.text(
            (x - 22, y1 + 10),
            f"{xmin + index * (xmax - xmin) / 5.0:.3g}",
            font=_font(14),
            fill=ink,
        )
        draw.text(
            (x0 - 82, y - 8),
            f"{ymax - index * (ymax - ymin) / 5.0:.4g}",
            font=_font(14),
            fill=ink,
        )
    draw.line((x0, y0, x0, y1), fill=ink, width=2)
    draw.line((x0, y1, x1, y1), fill=ink, width=2)

    def map_points(x_values, y_values):
        return [
            (
                x0 + (float(x) - xmin) / (xmax - xmin) * (x1 - x0),
                y1 - (float(y) - ymin) / (ymax - ymin) * (y1 - y0),
            )
            for x, y in zip(x_values, y_values, strict=True)
        ]

    legend_y = y0 + 5
    for x_values, y_values, color, label, width in curves:
        draw.line(map_points(x_values, y_values), fill=color, width=width, joint="curve")
        draw.line(
            (x1 - 165, legend_y + 8, x1 - 130, legend_y + 8),
            fill=color,
            width=width,
        )
        draw.text((x1 - 122, legend_y), label, font=_font(14), fill=ink)
        legend_y += 24
    draw.text((x0, top + 14), title, font=_font(20, True), fill=ink)
    draw.text(((x0 + x1) / 2 - 55, y1 + 40), xlabel, font=_font(17), fill=ink)
    draw.text((left + 5, top + 31), ylabel, font=_font(17), fill=ink)


def save_undrained_comparison(histories, state_label: str, path: Path) -> None:
    """Plot q, excess pressure, effective path, and volume constraint."""

    colors = {
        "MCC": (25, 30, 36),
        "EVP": (42, 107, 184),
        "NorSand": (205, 91, 43),
    }
    widths = {"MCC": 5, "EVP": 4, "NorSand": 4}
    image = Image.new("RGB", (1400, 1000), "white")
    draw = ImageDraw.Draw(image)
    specifications = (
        ("eps_axial", "q", "axial strain", "q (kPa)", "deviator stress"),
        (
            "eps_axial",
            "pore_pressure",
            "axial strain",
            "excess u (kPa)",
            "water-pressure response",
        ),
        ("p_effective", "q", "p' (kPa)", "q (kPa)", "effective-stress path"),
        (
            "eps_axial",
            "eps_v",
            "axial strain",
            "volumetric strain",
            "closed-boundary volume check",
        ),
    )
    for panel, (x_name, y_name, x_label, y_label, title) in enumerate(specifications):
        row, column = divmod(panel, 2)
        curves = [
            (
                history[x_name],
                history[y_name],
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
            x_label,
            y_label,
            f"{state_label}: {title}",
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, "PNG", optimize=True)
