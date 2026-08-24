# Modular 2-D hydro-mechanical finite elements

This folder is a separate saturated hydro-mechanical extension of the original
drained mechanical code. It solves quasi-static, small-strain, axisymmetric
problems with nodal radial/axial displacement and pore-pressure unknowns.

The triaxial specimen is represented by the r-z meridian of a cylinder. This
is a true 2-D discretization, but the element constructs the complete
axisymmetric strain vector

```text
[epsilon_rr, epsilon_hoop, epsilon_zz, gamma_r-hoop, gamma_rz, gamma_hoop-z]
```

before calling the 3-D MCC, EVP, or NorSand effective-stress update.

## Code organization

The numerical ingredients requested for later expansion are independent of
the FE solver:

```text
hm_2d/
|-- mesh.py                 Node, Quad4Element, mesh generation
|-- shape_functions.py      Quad4 interpolation and natural gradients
|-- quadrature.py           Gauss coordinates and weights
|-- elements.py             Jacobian, spatial gradients, B matrix, DOF maps
|-- hydraulics.py           Biot coefficient, storage, mobility, stabilization
|-- constitutive.py         model-independent effective-stress adapter
|-- boundary_conditions.py  triaxial and field-constraint helpers
|-- solver.py               coupled u-p assembly and nonlinear solution
|-- postprocessing.py       volume-averaged stress, strain, and pressure
|-- triaxial.py             closed-boundary undrained FE driver
|-- verification.py         analytical interpolation/poroelastic checks
|-- study.py                dense/loose three-model study and output writer
`-- materials/
    |-- mcc_model.py        separate MCC setup
    |-- evp_model.py        separate Kelln EVP setup
    `-- norsand_model.py    separate NorSand setup
```

The constitutive equations themselves remain in the original separate files
`modified_cam_clay.py`, `kelln_evp.py`, and `norsand.py`; the three files in
`hm_2d/materials/` contain only HM calibration and initialization.

## Coupled equations

The solver uses a monolithic backward-Euler displacement-pressure formulation.
With compression-positive pore pressure and effective soil stress, it enforces

```text
div(sigma' + alpha u I) = 0
alpha Delta epsilon_v(tension) + S Delta u
    - Delta t div(mobility grad u) = Delta fluid source
```

where `alpha` is the Biot coefficient and `S=1/M_Biot` is storage. Constitutive
stress is effective stress only; pore pressure is added at element assembly.
The default undrained tests use natural no-flow boundaries and zero mobility.
Water and grain compressibility give a very small, physical volume change
rather than imposing exactly zero volume algebraically.

Quad4 displacement and pressure interpolation are equal order. A
gradient-based pressure stabilization term is included. It is zero for a
uniform pore-pressure field and therefore does not alter the homogeneous
triaxial solution. The demonstration ties pressure nodes because a homogeneous,
impermeable triaxial specimen must have uniform pressure; remove that tie for
future spatial consolidation problems.

## Run

From the project root:

```powershell
python -m hm_2d.study
python -m unittest discover -s tests -v
```

The study runs dense `psi=-0.05` and loose `psi=+0.02` specimens to 15% axial
strain for MCC, EVP, and NorSand. It also runs an independent material-point
solution with exactly zero volumetric strain for every case.

## Verification results

The generated `results/basic_verification.json` reports:

- Quad4 partition-of-unity error: `1.11e-16`.
- natural-gradient sum error: `2.78e-17`.
- axisymmetric cylinder-volume integration error: `2.45e-16`.
- closed-form poroelastic pore-pressure error: `1.29e-12` relative.
- closed-form poroelastic q error: `5.49e-14` relative.
- analytical patch momentum and mass residuals: approximately `2e-16`.

Across the six nonlinear tests:

- FE/material-point q RMSE: `0.02-0.50%`.
- FE/material-point excess-pore-pressure RMSE: `0.05-0.90%`.
- maximum mass residual: below `2e-14`.
- maximum absolute volumetric strain: below `1.66e-5`.
- total radial cell-pressure error: below `0.005 kPa` for MCC/EVP and below
  `0.99 kPa` for the explicit NorSand yield-corner safeguard.

The matched model curves relative to MCC are:

| State | Model | q curve RMSE | pore-pressure curve RMSE |
|---|---:|---:|---:|
| Dense | EVP | 0.68% | 1.14% |
| Dense | NorSand | 0.54% | 4.20% |
| Loose | EVP | 0.75% | 0.55% |
| Loose | NorSand | 1.32% | 1.51% |

Matching is obtained by sharing the CSL, `M`, compression/swelling slopes,
elasticity, initial effective pressure, and state parameter; matching the
initial MCC/NorSand yield onset; making EVP rate effects small at the selected
rate; and using `H0=75` for the NorSand undrained-path calibration. This shows
that all three critical-state models can reproduce the same target response
under a deliberate calibration. It does not imply that their equations are
generally identical.

## Outputs

- `results/undrained_dense.png`
- `results/undrained_loose.png`
- `results/undrained_dense_histories.csv`
- `results/undrained_loose_histories.csv`
- `results/undrained_summary.csv`
- `results/basic_verification.json`
- `results/matched_parameters.json`

## Current boundary

This is a saturated, single-phase, quasi-static, small-strain research solver.
The core already contains storage, Darcy mobility, pressure boundary values,
and fluid source vectors, but this first study verifies only homogeneous
closed-boundary undrained loading. A spatial drainage/consolidation benchmark
should be added before using nonzero permeability for engineering predictions.
Finite strain, partial saturation, cavitation, cyclic NorSand features,
localization regularization, contact, and dynamics remain outside the scope.
