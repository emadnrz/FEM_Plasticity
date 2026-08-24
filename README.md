# Mechanical critical-state finite elements: MCC, EVP, and NorSand

This Python project provides a common small-strain mechanical constitutive
interface, three separate critical-state soil model files, a nonlinear 2-D
axisymmetric Quad4 finite-element solver, model verification tests, and
matched dense/loose drained triaxial comparisons.  The separate `hm_2d/`
package adds a modular saturated displacement--pore-pressure formulation and
dense/loose undrained comparisons.

## Implemented components

- `plasticity.py`: the model-independent Gauss-point function
  `integrate_material(model, state, deps, dt, compute_tangent)`.  It returns
  updated stress/state and a numerical algorithmic tangent used by Newton FE
  equilibrium iterations.
- `modified_cam_clay.py`: rate-independent MCC yield surface, associated flow,
  isotropic hardening, 3-D stress integration, and a consistent ideal-triaxial
  driver.
- `kelln_evp.py`: the verified Kelln elastic-viscoplastic model, adaptive
  modified-Euler integration, and time/rate-dependent triaxial driver.
- `norsand.py`: monotonic NorSand outer surface with pressure-dependent
  elasticity, the supplied algorithm note's Lode-angle interpolation,
  associated flow, arbitrary-stress initialization, and selectable
  algorithm-note/Marinelli-2022 image-pressure hardening and softening.
- `fem.py`: model-independent 2-D axisymmetric four-node quadrilateral
  mechanics with 2x2 Gauss integration, radial/axial displacement DOFs, hoop
  strain, displacement/force increments, integration-point state rollback,
  rigid-membrane constraints, and Newton equilibrium iterations.
- `triaxial.py`: a general material-point mixed-control driver.  Each model
  also provides a faster consistent axisymmetric driver for verification and
  parameter studies.
- `hm_2d/`: a separate 2-D axisymmetric monolithic u-p implementation. Shape
  functions, quadrature, nodes/elements, element kinematics, hydraulics,
  assembly, and postprocessing are split into expandable modules. Its
  `materials/` directory contains separate MCC, EVP, and NorSand setup files.

Stress, time, and strain units are kPa, days, and dimensionless strain.
Constitutive vectors use compression-positive
`[xx, yy, zz, xy, xz, yz]`; shear strains are engineering strains.  The FE
adapter converts automatically to the conventional tension-positive
displacement formulation.

## Run the mechanical workflow

The bundled Codex Python runtime can run the project with:

```powershell
python run_all.py
```

The command runs the original mechanical tests, the digitized Marinelli et al. Figure 1 benchmark,
the public NorSand dense/loose reference case, the matched three-model
comparison, and loose/dense drained tests with all three models in the 2-D FE
specimen. Results are written to `results/`.

Run the new undrained HM study separately with:

```powershell
python -m hm_2d.study
```

It writes coupled stress and excess-water-pressure results to `hm_2d/results/`.

## Verification status

- MCC reproduces the quoted normally consolidated undrained endpoint:
  `q=130.104 kPa`, `u=134.947 kPa`, versus `130.1/134.95 kPa`.
- EVP reproduces the Kelln endpoint: `q=129.645 kPa`, `u=135.148 kPa`, versus
  `129.7/134.95 kPa`.
- NorSand reproduces all four triaxial paths in Figure 1 of Marinelli et al.
  (2022), using its parameter set A.  Against points digitized from the paper,
  curve NRMSE is `0.244-3.893%` of the plotted range; the integration-step
  sensitivity is at most `0.0666%`.  It also retains the separate public Itasca
  dense/loose topology check.
- The 2-D axisymmetric specimen reproduces the material-point q curves within
  `0.02-0.81%` RMSE. Across the three models, q-curve RMSE versus MCC is
  `0.72-0.89%` for dense soil and `0.88-2.85%` for loose soil. All approach
  the common drained critical-state target `q=200 kPa`, `p'=166.67 kPa`.
- The coupled 2-D u-p specimen reproduces independent undrained material-point
  q curves within `0.02-0.50%` RMSE and pore-pressure curves within
  `0.05-0.90%`. Cross-model q RMSE versus MCC is at most `1.32%`; the largest
  pore-pressure-curve difference is `4.20%` for dense NorSand.

See `NORSAND_AUDIT.md` for the supplied-document equation audit and Figure 1
results.  See `MODEL_MATCHING.md` for the cross-model parameter mapping.
See `MODEL_EQUIVALENCE_REPORT.md` for the combined drained/undrained property
tables, unified NorSand calibration, parameter-sensitivity discussion, and the
zero-dilatancy diagnostic.

## Scope and limitations

This is a transparent research/reference solver, not a replacement for a
validated production geotechnical FE package. Current scope is small-strain
and quasi-static. The legacy solver is mechanical-only; `hm_2d/` adds saturated
pore-pressure DOFs, storage, Darcy mobility, and fluid mass balance. The first
HM verification is a homogeneous closed-boundary undrained test, not yet a
spatial consolidation benchmark. Finite strain, partial saturation, contact,
dynamics, localization regularization, and an arc-length solver are absent.

The NorSand implementation is the monotonic outer-surface subset.  The
optional loose-sand softening term is included; the inner cap and
principal-stress-rotation term required for cyclic mobility remain outside
this version.

## Main outputs

- `results/matched_dense.png` and `matched_loose.png`
- `results/matched_summary.csv`
- `results/matched_parameters.json`
- `results/norsand_itasca_reference.png`
- `results/norsand_figure1_verification.png`
- `results/norsand_figure1_summary.csv`
- `results/fe2d_drained_dense.png`
- `results/fe2d_drained_loose.png`
- `results/fe2d_drained_summary.csv`
- `hm_2d/results/undrained_dense.png` and `undrained_loose.png`
- `hm_2d/results/undrained_summary.csv`
- `hm_2d/results/basic_verification.json`

## Technical sources

- [Jefferies (1993), original NorSand paper](https://doi.org/10.1680/geot.1993.43.1.91)
- [Itasca NorSand equations](https://docs.itascacg.com/itasca930/common/models/norsand/doc/modelnorsand.html)
- [Itasca dense/loose undrained verification example](https://docs.itascacg.com/itasca930/flac3d/zone/test3d/ConstitutiveModels/UndrainedTriaxialNorSand/undrainedtriaxialnorsand.html)
- [Cheng and Jefferies (2020), 3-D implementation](https://doi.org/10.1061/9780784482810.002)
- [Kelln et al. (2008), EVP FE integration](https://doi.org/10.1016/j.compgeo.2007.10.003)
