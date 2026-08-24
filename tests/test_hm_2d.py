"""Verification tests for the modular axisymmetric u-p implementation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hm_2d.materials import CommonCalibration, create_mcc_case  # noqa: E402
from hm_2d.mesh import structured_cylinder_mesh  # noqa: E402
from hm_2d.triaxial import run_undrained_triaxial  # noqa: E402
from hm_2d.verification import (  # noqa: E402
    verify_elastic_undrained_patch,
    verify_shape_and_axisymmetric_volume,
)
from triaxial import solve_triaxial as solve_material_point  # noqa: E402


class HydroMechanical2DTests(unittest.TestCase):
    def test_mesh_entities_are_created_outside_solver(self) -> None:
        mesh = structured_cylinder_mesh(
            radius=1.0, height=2.0, radial_elements=2, axial_elements=3
        )
        self.assertEqual(mesh.n_nodes, 12)
        self.assertEqual(len(mesh.elements), 6)
        self.assertEqual(mesh.elements[0].node_ids, (0, 1, 4, 3))

    def test_shape_quadrature_and_axisymmetric_volume(self) -> None:
        checks = verify_shape_and_axisymmetric_volume()
        for error in checks.values():
            self.assertLess(error, 1.0e-12)

    def test_closed_form_elastic_undrained_patch(self) -> None:
        checks = verify_elastic_undrained_patch()
        for error in checks.values():
            self.assertLess(error, 1.0e-8)

    def test_mcc_hm_matches_zero_volume_material_point(self) -> None:
        calibration = CommonCalibration()
        case = create_mcc_case(calibration, state_parameter=0.02)
        fe = run_undrained_triaxial(
            structured_cylinder_mesh(), case, axial_strain_end=0.02, increments=10
        )
        reference = solve_material_point(
            case.model,
            case.initial_state,
            0.02,
            "undrained",
            10,
            calibration.strain_rate,
        )
        q_error = np.sqrt(np.mean((fe["q"] - reference["q"]) ** 2)) / np.max(
            reference["q"]
        )
        u_error = np.sqrt(
            np.mean((fe["pore_pressure"] - reference["u"]) ** 2)
        ) / np.max(np.abs(reference["u"]))
        self.assertLess(q_error, 0.005)
        self.assertLess(u_error, 0.005)
        self.assertLess(np.max(np.abs(fe["eps_v"])), 5.0e-5)
        self.assertLess(np.max(np.abs(fe["radial_total_stress"] - 100.0)), 0.01)


if __name__ == "__main__":
    unittest.main(verbosity=2)
