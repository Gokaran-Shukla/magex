# 6 · Multiscale: first-principles exchange → micromagnetic parameters

```bash
magex params ../../validation_runs/CrI3_atomic_soc/results.json.gz --thickness 7.0
```

prints and returns (JSON) the continuum parameters obtained from a gradient expansion of the
atomistic energy around the ferromagnetic state:

| quantity | expression |
|----------|------------|
| exchange stiffness | A_ab = (1/2V_c) Σ_{i∈cell} Σ_j J_ij r_a r_b  (A = tr/3 in bulk, (A_xx+A_yy)/2 for films) |
| interfacial DMI | D_int = −(1/2V_c) Σ (ẑ × r̂_ij)·D_ij \|r_ij\| over in-plane bonds |
| bulk DMI | D_bulk = (1/3V_c) Σ r̂_ij·D_ij \|r_ij\| |
| uniaxial anisotropy | K_u = [Σ_i (A_i,xx − A_i,zz) + Σ_ij (J^S_zz − J^S_xx)]/V_c (single-ion + two-site) |
| saturation magnetisation | M_s = μ_B Σ_i \|μ_i\| / V_c |

V_c is the cell volume; for 2D materials (vacuum in the cell) V_c = area × `thickness_A` (default: the
nearest-neighbour distance).  The expressions are checked in the test-suite against the exact lattice
sum of long-wavelength spirals (agreement 0.05 %), and the DMI sign convention is the one used by the
micromagnetic solver, so the chirality is preserved.

Use them directly in a micromagnetic region:

```toml
[[region]]
from_atomistic = "../../validation_runs/CrI3_atomic_soc/results.json.gz"
J = "mft"
thickness_A = 7.0
shape = "disk"
diameter_nm = 120
alpha = 0.1
```

(`examples/micromag/cri3_from_atomistic.toml`: CrI3 flake relaxed from a random state, then an
out-of-plane hysteresis loop.)

## The complete workflow for a new material

```bash
magex run input.toml                  # 1. DFT+Wannier -> J_ij, DMI, SIA (MPI parallel)
magex mc tc.toml                      # 2. ordering temperature
magex llg textures.toml               # 3. atomistic textures, field on/off
magex params magex_results/results.json   # 4. continuum parameters
magex micromag device.toml            # 5. device-scale simulation
```

## Caveats

* the expansion assumes a ferromagnet; for antiferromagnets the sign-weighted stiffness of the
  staggered order is returned (useful for AFM textures, not for an FM micromagnetic model);
* temperature: these are T = 0 parameters; A(T) ∝ m(T)^~1.8 and K(T) ∝ m(T)^3 (Callen–Callen) can be
  obtained by combining with the Monte Carlo m(T);
* long-range couplings in metals: converge A with `rcut` (A is a second moment, so slowly decaying
  couplings matter more than for T_C).
