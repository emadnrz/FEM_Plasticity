"""Shared calibration data without constitutive-model implementation code."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CommonCalibration:
    """Critical-state quantities deliberately shared by all three models."""

    initial_effective_pressure: float = 100.0
    critical_stress_ratio: float = 1.2
    compression_slope: float = 0.06
    swelling_slope: float = 0.01
    critical_void_ratio_at_reference: float = 0.80
    poisson_ratio: float = 0.20
    reference_pressure: float = 100.0
    strain_rate: float = 1.0e-3


@dataclass
class MaterialCase:
    """A named constitutive model, initial state, and calibration metadata."""

    name: str
    model: Any
    initial_state: Any
    metadata: dict[str, float] = field(default_factory=dict)
