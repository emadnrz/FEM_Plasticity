"""Hydraulic and Biot coupling properties for saturated soil."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HydraulicProperties:
    """Saturated single-phase hydraulic properties.

    Parameters use kPa, seconds, and the mesh length unit. ``storage`` is the
    inverse Biot modulus in 1/kPa. ``mobility`` multiplies the pore-pressure
    gradient in the Darcy term and therefore has units length^2/(kPa s).
    Impermeable undrained tests use zero mobility and natural no-flow
    boundaries.  Equal-order pressure stabilization vanishes for a uniform
    pressure field.
    """

    biot_coefficient: float = 1.0
    storage: float = 0.0
    mobility: float = 0.0
    stabilization_factor: float = 0.05

    def __post_init__(self) -> None:
        if not 0.0 < self.biot_coefficient <= 1.0:
            raise ValueError("Biot coefficient must be in (0, 1]")
        if self.storage < 0.0 or self.mobility < 0.0:
            raise ValueError("storage and mobility must be nonnegative")
        if self.stabilization_factor < 0.0:
            raise ValueError("stabilization factor must be nonnegative")

    @classmethod
    def saturated_soil(
        cls,
        porosity: float,
        water_bulk_modulus: float = 2.2e6,
        grain_bulk_modulus: float = 36.0e6,
        biot_coefficient: float = 1.0,
        mobility: float = 0.0,
        stabilization_factor: float = 0.05,
    ) -> "HydraulicProperties":
        """Construct storage from water/grain compressibility."""

        if not 0.0 < porosity < 1.0:
            raise ValueError("porosity must be in (0, 1)")
        if water_bulk_modulus <= 0.0 or grain_bulk_modulus <= 0.0:
            raise ValueError("constituent bulk moduli must be positive")
        storage = porosity / water_bulk_modulus + (
            biot_coefficient - porosity
        ) / grain_bulk_modulus
        return cls(
            biot_coefficient,
            storage,
            mobility,
            stabilization_factor,
        )
