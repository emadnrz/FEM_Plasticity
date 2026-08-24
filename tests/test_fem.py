"""Mechanical finite-element patch tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fem import (  # noqa: E402
    average_stress,
    axisymmetric_specimen_mesh,
    flatten_states,
    initialize_integration_states,
    solve_increment,
    triaxial_displacement_bcs,
)
from modified_cam_clay import MCCParameters, ModifiedCamClay  # noqa: E402
from plasticity import invariants  # noqa: E402


class MechanicalFETests(unittest.TestCase):
    def test_quad4_axisymmetric_constant_confining_pressure_patch(self) -> None:
        model = ModifiedCamClay(MCCParameters())
        state = model.initialize_isotropic(100.0, 400.0)
        mesh = axisymmetric_specimen_mesh()
        states = initialize_integration_states(mesh, state)
        result = solve_increment(
            mesh,
            model,
            states,
            triaxial_displacement_bcs(mesh, 1.0e-5),
        )
        stress = average_stress(flatten_states(result.states))
        p, q, *_ = invariants(stress)
        self.assertLess(abs(stress[0] - 100.0), 1.0e-6)
        self.assertLess(abs(stress[1] - 100.0), 1.0e-6)
        self.assertGreater(q, 0.0)
        self.assertLess(result.residual_norm, 1.0e-7)
        self.assertAlmostEqual(p, 100.0 + q / 3.0, places=7)


if __name__ == "__main__":
    unittest.main(verbosity=2)
