# NorSand algorithm audit and Figure 1 verification

## Outcome

`norsand.py` implements the supplied algorithm note in the project's
compression-positive, six-component 3-D material convention. The 2-D FE layer
embeds axisymmetric radial, hoop, axial, and radial-axial shear strains in this
vector; no FE-specific NorSand law is required.

The Figure 1 verification uses parameter set A from Marinelli et al. (2022),
with `p0=250 kPa`, `psi0=+/-0.12`, `OCR=1`, and both TXU and TXD paths.  The
reference points were digitized from the red VBA curves in the attached PDF.

| Path | Quantity | NRMSE (% of plotted range) | Step-size check NRMSE |
|---|---|---:|---:|
| TXU dense | q | 1.545% | 0.0531% |
| TXU loose | q | 0.244% | 0.0666% |
| TXD dense | q | 1.992% | 0.0139% |
| TXD loose | q | 3.893% | 0.0074% |
| TXD dense | volumetric strain | 0.900% | 0.0096% |
| TXD loose | volumetric strain | 2.945% | 0.0053% |

The largest curve NRMSE is 3.893%.  This is a digitized-plot comparison, not a
claim of accuracy against an author-supplied numerical history.

## Equation mapping

| Supplied note | Implementation |
|---|---|
| Stress convention is compression-negative | All signs are transformed to the documented project convention: compression-positive stress/strain |
| 4-component plane-strain vector with tensor shear | General 6-component material vector; the 2-D axisymmetric FE layer supplies radial, hoop, axial, and engineering radial-axial shear components |
| Eqs. 5-14: p, q, J2, J3, and Lode angle | `plasticity.invariants` |
| Eq. 15: Lode-dependent M | `NorSand._M_lode`; the cosine argument is sign-transformed because triaxial compression is `theta=+pi/6` here |
| Eqs. 16-22: CSL, image state, chi_i, Mi | `critical_void_ratio` and `point` |
| Eqs. 23-24: pressure-dependent G and K | `elastic_matrix` |
| Eq. 25: outer yield surface | An equivalent dimensionless surface is used; it has the same zero set and associated-flow direction up to a positive scalar |
| Eqs. 26-32: derivatives | Centered numerical derivatives include all Lode/state coupling and avoid the printed singular `1/psi` expression |
| Eqs. 33-34: image-pressure limit and eta_L | `point.pi_max` and `point.eta_L` |
| Eqs. 35-36: initialization | Isotropic initialization is exact; arbitrary stress initialization uses Newton iteration |
| Eqs. 43-45: image-pressure hardening/softening | `_image_pressure_rate`, with selectable `algorithm` and `paper_2022` formulations |
| Eqs. 46-53: plastic correction/update | Yield crossing, explicit substeps, Heun update in the triaxial verifier, and a final consistency projection |

## Necessary clarifications and corrections

1. The non-spherical initialization update printed in the note omits the
   Newton quotient.  The code uses `pi_new = pi_old - F/(dF/dpi)`, which is the
   dimensionally and mathematically correct Newton update.
2. The multiplier numerator printed in Eq. 47 is dimensionally ambiguous.
   The implementation evaluates the standard consistency equation, including
   stress, void-ratio, and image-pressure derivatives, then projects residual
   yield drift to the surface.
3. The conference paper and the supplied note give two related but different
   hardening-limit/softening expressions.  Both are available:
   `softening_formulation="algorithm"` follows the supplied note, while
   `"paper_2022"` follows Eqs. 4-5 of the benchmark paper.
4. The PLAXIS form sets augmented softening to zero for drained loading and
   for `Dp<=0`.  The Figure 1 runner therefore uses `Z=True` only for TXU and
   the code applies a Macaulay bracket to the contractive dilatancy.  Omitting
   these restrictions incorrectly turns the term into extra dense-sand
   hardening.
5. The algorithm note's void-ratio sign is transformed consistently:
   `de=-(1+e)*deps_v` in this project's compression-positive convention.

## Reproduce

```powershell
python paper_figure1_verification.py
```

Outputs:

- `results/norsand_figure1_verification.png`
- `results/norsand_figure1_summary.csv`
- `results/norsand_figure1_histories.csv`
- `results/norsand_figure1_digitized.csv`

`python run_all.py` also runs the equation tests, Figure 1 verification, the
three-model comparisons, and the loose/dense 2-D axisymmetric FE checks.
