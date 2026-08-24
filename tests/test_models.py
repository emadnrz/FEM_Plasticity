"""Constitutive verification tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kelln_evp import EVPParameters, KellnEVP  # noqa: E402
from modified_cam_clay import MCCParameters, ModifiedCamClay  # noqa: E402
from norsand import NorSand, NorSandParameters  # noqa: E402
from plasticity import integrate_material, invariants  # noqa: E402


class CriticalStateModelsTests(unittest.TestCase):
    def test_triaxial_invariants(self) -> None:
        p, q, theta, *_ = invariants(np.array([200.0, 100.0, 100.0, 0, 0, 0]))
        self.assertAlmostEqual(p, 400.0 / 3.0)
        self.assertAlmostEqual(q, 100.0)
        self.assertAlmostEqual(theta, np.pi / 6.0, places=7)

    def test_mcc_published_undrained_endpoint(self) -> None:
        model = ModifiedCamClay(
            MCCParameters(
                M=1.2, lambda_c=0.066, kappa=0.0077, N=1.788,
                nu=0.3, p_ref=1.0,
            )
        )
        history = model.solve_triaxial(
            model.initialize_isotropic(200.0, 200.0),
            0.05, "undrained", 100,
        )
        self.assertLess(abs(history["q"][-1] - 130.1) / 130.1, 0.002)
        self.assertLess(abs(history["u"][-1] - 134.95) / 134.95, 0.002)

    def test_evp_published_undrained_endpoint(self) -> None:
        model = KellnEVP(
            EVPParameters(
                M=1.2, lambda_c=0.066, kappa=0.0077,
                psi_visc=5.0e-5, N=1.788, t0=1.0,
                nu=0.3, Z=1.0, p_ref=1.0,
            )
        )
        history = model.solve_triaxial(
            model.initialize_isotropic(200.0, 200.0),
            0.05, "undrained", 100, 0.0015,
        )
        self.assertLess(abs(history["q"][-1] - 129.7) / 129.7, 0.005)
        self.assertLess(abs(history["u"][-1] - 134.95) / 134.95, 0.005)

    def test_norsand_public_reference_topology(self) -> None:
        model = NorSand(
            NorSandParameters(
                Gamma=1.2, lambda_c=0.06, M_tc=1.47, chi_tc=3.2,
                N_coupling=0.5, H0=75.0, Hy=400.0,
                G_ref=20_970.0, m=0.47, nu=0.15, p_ref=100.0,
            )
        )
        dense = model.solve_triaxial(
            model.initialize_isotropic(395.0, -0.025, 1.1),
            0.10, "undrained", 50, 1.0e-4,
        )
        loose = model.solve_triaxial(
            model.initialize_isotropic(395.0, 0.10, 1.1),
            0.10, "undrained", 50, 1.0e-4,
        )
        # Published example: dense hardening and rising p'; loose peak,
        # softening, and falling p' with the same parameters except psi0.
        self.assertGreater(dense["p"][-1], 395.0)
        self.assertAlmostEqual(dense["q"][-1], float(np.max(dense["q"])))
        self.assertLess(loose["p"][-1], 0.5 * 395.0)
        self.assertGreater(float(np.max(loose["q"])), 1.3 * loose["q"][-1])

    def test_norsand_algorithm_note_lode_interpolation(self) -> None:
        parameters = NorSandParameters(M_tc=1.4)
        model = NorSand(parameters)
        self.assertAlmostEqual(model._M_lode(np.pi / 6.0), parameters.M_tc)
        self.assertAlmostEqual(
            model._M_lode(-np.pi / 6.0),
            3.0 * parameters.M_tc / (3.0 + parameters.M_tc),
        )
        expected_plane_strain = parameters.M_tc - parameters.M_tc**2 / (
            3.0 + parameters.M_tc
        ) * np.cos(np.pi / 4.0)
        self.assertAlmostEqual(model._M_lode(0.0), expected_plane_strain)

    def test_norsand_general_initialization_is_on_surface(self) -> None:
        model = NorSand(NorSandParameters())
        stress = np.array([240.0, 120.0, 120.0, 0.0, 0.0, 0.0])
        p = invariants(stress)[0]
        e = model.critical_void_ratio(p) - 0.03
        state = model.initialize(stress, e, OCR=1.0)
        self.assertLess(abs(model.yield_value(state.stress, state.e, state.pi)), 1.0e-10)

    def test_norsand_softening_is_contractive_only(self) -> None:
        model = NorSand(NorSandParameters(Z=True, softening_formulation="paper_2022"))
        loose = model.initialize_isotropic(250.0, 0.12)
        # Put the stress on a contractive triaxial-compression point.
        stress = model._triaxial_stress(150.0, 75.0)
        state = type(loose)(stress, loose.e, loose.pi, loose.eps_p.copy())
        point = model.point(state.stress, state.e, state.pi)
        self.assertGreater(point.dilatancy, 0.0)
        softened_rate = model._image_pressure_rate(state, point)
        pure = NorSand(
            NorSandParameters(Z=False, softening_formulation="paper_2022")
        )
        self.assertLess(softened_rate, pure._image_pressure_rate(state))

    def test_common_interface_returns_finite_tangent(self) -> None:
        models_states = []
        mcc = ModifiedCamClay(MCCParameters())
        models_states.append((mcc, mcc.initialize_isotropic(100.0, 400.0)))
        evp = KellnEVP(EVPParameters())
        models_states.append((evp, evp.initialize_isotropic(100.0, 400.0)))
        ns = NorSand(NorSandParameters())
        models_states.append((ns, ns.initialize_isotropic(100.0, -0.02, 2.0)))
        deps = np.array([1.0e-6, -2.0e-7, -2.0e-7, 0, 0, 0])
        for model, state in models_states:
            response = integrate_material(model, state, deps, dt=0.0, compute_tangent=True)
            self.assertEqual(response.tangent.shape, (6, 6))
            self.assertTrue(np.all(np.isfinite(response.tangent)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
