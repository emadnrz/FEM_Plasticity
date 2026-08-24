# When MCC, Kelln EVP, and NorSand can give the same result

## Short answer

The three models can share the same critical-state endpoint if they use the
same critical stress ratio, the same critical-state line, the same initial
state, and the same drainage/loading path.  Their complete transient curves
are not mathematically identical because their yield surfaces and hardening
laws differ, and finite-viscosity EVP is rate dependent.  They can nevertheless
be closely matched for selected monotonic paths by aligning elasticity, first
yield, and hardening/dilatancy.

In this project, one common calibration gives the following drained triaxial
results at 30% axial strain:

| State | Model | Peak q (kPa) | Final q (kPa) | Final-q error from common critical state | q-curve RMSE versus MCC |
|---|---|---:|---:|---:|---:|
| Dense, psi=-0.05 | MCC | 309.210 | 200.057 | +0.029% | reference |
|  | EVP | 310.122 | 199.939 | -0.031% | 0.144% |
|  | NorSand | 310.287 | 198.000 | -1.000% | 1.294% |
| Loose, psi=+0.02 | MCC | 199.921 | 199.921 | -0.039% | reference |
|  | EVP | 199.697 | 199.697 | -0.152% | 0.149% |
|  | NorSand | 198.812 | 198.812 | -0.594% | 2.260% |

The volume-curve RMSE values versus MCC are below `0.21%` for every matched
case.  Exact data are in `results/matched_summary.csv`.

## 1. Common critical-state line

NorSand uses

```text
ec(p') = Gamma_e - lambda ln(p'/p_ref)
psi_state = e - ec(p').
```

At critical state, MCC gives

```text
Vcs = N - (lambda-kappa) ln(2) - lambda ln(p'/p_ref).
```

Because `V=1+e`, the exact intercept mapping is

```text
N_MCC = 1 + Gamma_e + (lambda-kappa) ln(2).
```

The comparison therefore uses the same `M=1.2`, `lambda=0.06`,
`Gamma_e=0.80`, and `p_ref=100 kPa`, with `kappa=0.01` and
`N_MCC=1.834657`.

## 2. Common density/state mapping

For an isotropic MCC state at `(p',pc')`, substitution into the common CSL
gives

```text
psi_state = (lambda-kappa) ln(2p'/pc')
pc'/p' = 2 exp[-psi_state/(lambda-kappa)].
```

At `p'=100 kPa`, the selected states are:

| State | psi_state | MCC pc'/p' | MCC pc' (kPa) |
|---|---:|---:|---:|
| Dense | -0.05 | 5.4366 | 543.656 |
| Loose | +0.02 | 1.3406 | 134.064 |

This mapping exposes an important limitation.  An admissible isotropic MCC
state requires `pc' >= p'`, hence

```text
psi_state <= (lambda-kappa) ln(2).
```

For this calibration the upper limit is `+0.03466`.  The public NorSand loose
verification state `psi=+0.10` therefore cannot be represented by MCC at the
same pressure and CSL without starting outside the MCC yield surface.  The
matched loose state is deliberately `+0.02`.

## 3. Elastic matching

MCC and Kelln EVP use

```text
K = Vp'/kappa,
G = 3(1-2nu)K/[2(1+nu)].
```

NorSand uses

```text
G = Gref (p'/p_ref)^m,
K = 2(1+nu)G/[3(1-2nu)].
```

The comparison uses common `nu=0.2`, `m=1`, and `Gref=13,400 kPa`, which is
the average initial MCC modulus for the two states.  Exact elastic identity at
every pressure and void ratio is not possible with these two native elastic
laws, but their initial stiffnesses are within about 2%.

## 4. Matching first yield

MCC first yield under constant cell pressure is obtained by intersecting

```text
q^2/M^2 - p'(pc'-p') = 0
q = 3(p'-p0').
```

NorSand uses the outer surface

```text
q/(p'Mi) = 1 - ln(p'/pi').
```

The code solves for the initial NorSand image pressure, expressed by OCR, so
this surface passes through the MCC first-yield point.  The mappings are:

| State | Matched yield p' (kPa) | Matched yield q (kPa) | NorSand OCR |
|---|---:|---:|---:|
| Dense | 205.438 | 316.315 | 7.4585 |
| Loose | 117.601 | 52.802 | 1.7183 |

This is a constitutive surface mapping; NorSand OCR and conventional MCC OCR
are not numerically interchangeable definitions.

## 5. Matching transient plastic response

NorSand's transient response is controlled mainly by:

- `chi_tc`: state-dilatancy and limiting image pressure;
- `N_coupling`: state dependence of image friction;
- `H0` and `Hy`: image-pressure hardening rate, `H=H0-Hy*psi_state`;
- initial image pressure/OCR.

One parameter set is used for both states: `chi_tc=4.5`, `N_coupling=0.30`,
`H0=37`, and `Hy=260`.  These are fitted to the two MCC curves together, not
independently.  The separate density effect comes from `psi_state` and the
mapped initial image pressure.

EVP uses the same MCC parameters and initial `pc'`.  Its viscosity index is
set to `psi_visc=1e-4` with `t0=1 day` at a strain rate of `0.001/day`.  This is
the previously demonstrated quasi-rate-independent limit; it should not be
confused with the NorSand state parameter.

## 6. Why the final critical states coincide

For constant cell pressure `sigma3'=p0'`, triaxial geometry gives

```text
p' = p0' + q/3.
```

At critical state `q=Mp'`, so the common analytical target is

```text
qcs = M p0'/(1-M/3)
pcs = p0' + qcs/3.
```

For `M=1.2` and `p0'=100 kPa`:

```text
qcs = 200.000 kPa
pcs = 166.667 kPa
Vcs = 1.769350.
```

Dense and loose specimens approach the same point from different directions:
the dense state peaks and dilates, whereas the loose state hardens and
contracts.  Matching critical state does not imply matching the entire path.

## 7. NorSand formulation and verification boundary

The implemented monotonic outer-surface model includes:

```text
eta/Mi = 1 - ln(p'/pi')
Mi,tc = Mtc - N_coupling chi_i |psi_i|
Dp = deps_v^p/deps_q^p = Mi - eta
dpi'/pi' = H g(theta)(p'/pi')^2(pi,max'/p' - pi'/p') deps_q^p.
```

The supplied algorithm-note Lode interpolation is now used for `g(theta)`,
and the optional contractive softening term is available through `Z`.  The
Marinelli et al. (2022) hardening form can be selected with
`softening_formulation="paper_2022"`; its `S` term is used only for undrained,
contractive loading.

The public Itasca verification parameters produce:

| State | q peak (kPa) | q final (kPa) | p' final (kPa) | Observed behavior |
|---|---:|---:|---:|---|
| psi0=-0.025 | 611.319 | 611.319 | 418.773 | dense hardening, rising p' |
| psi0=+0.10 | 261.452 | 174.952 | 125.504 | loose peak-softening, falling p' |

These reproduce the published qualitative curves and use the published
parameters exactly.  The source page provides plotted curves but no numerical
history files, so a pointwise numeric error against FLAC3D is not claimed.

Separately, the attached Marinelli et al. Figure 1 has been digitized and
reproduced with parameter set A.  Its six stress/volume curve comparisons have
normalized RMSE values from `0.244%` to `3.893%`; see `NORSAND_AUDIT.md` and
`results/norsand_figure1_summary.csv`.

Sources: [Itasca model equations](https://docs.itascacg.com/itasca930/common/models/norsand/doc/modelnorsand.html),
[Itasca verification example](https://docs.itascacg.com/itasca930/flac3d/zone/test3d/ConstitutiveModels/UndrainedTriaxialNorSand/undrainedtriaxialnorsand.html),
and [Cheng and Jefferies (2020)](https://doi.org/10.1061/9780784482810.002).

## 8. FE implementation boundary

`fem.py` is a displacement-based 2-D axisymmetric solver using four-node
quadrilaterals. Any material satisfying the common interface can be assigned
without changing the FE assembly. The triaxial cylinder has radial and axial
DOFs, includes hoop strain `u_r/r`, and uses a rigid membrane to preserve
uniform constant cell pressure.

At 30% axial strain, the dense FE curves have q-RMSE values versus MCC of
`0.72%` (EVP) and `0.89%` (NorSand). The loose values are `0.88%` and `2.85%`.
Every FE solution is also checked against its model's independent
material-point driver; q-RMSE remains below `0.82%`.

The legacy `fem.py` demonstrations are drained. The separate `hm_2d/` package
now supplies mixed displacement-pressure elements, fluid mass balance,
storage, and Darcy mobility for saturated analyses. Its first verified cases
are homogeneous closed-boundary undrained triaxial tests. See
`MODEL_EQUIVALENCE_REPORT.md` for the combined drained/undrained assessment and
the distinction between path-specific and unified NorSand calibration.
