# 1 · The spin model: from `results.json` to a supercell

Every spin-dynamics and Monte Carlo run starts from a `[spinmodel]` table:

```toml
[spinmodel]
results     = "magex_results/results.json"   # .json or .json.gz written by `magex run`
J           = "mft"        # mft (magnetic force theorem) | bruno (renormalised) | downfold
anisotropic = true         # full 3x3 tensors (DMI + symmetric anisotropy); false -> isotropic J only
sia         = true         # single-ion anisotropy tensors A_i
rcut        = 8.0          # coupling cut-off in Å (default: 2 x nearest-neighbour distance)
jtol        = 1e-4         # drop couplings smaller than this (meV)
supercell   = [10, 10, 10] # in magnetic unit cells
pbc         = [true, true, true]
# moments = [2.2]          # override |μ| per magnetic site (μ_B)
# axis = [0, 0, 1]         # reference axis of the ordered state
```

The loader shares couplings that sit exactly on the boundary of the Green's-function supercell among
their equivalent images, so no bond is double counted.  At start-up a summary is printed:

```
spin model: 1 magnetic sites/cell (Fe), moments [2.218] mu_B, reference = ferromagnet
  couplings: J = mft, rcut = 8.000 A (NN 2.485 A), 10 shells, 168 couplings/cell, 89.9% of J(q=0) retained
  supercell (8, 8, 8), pbc (True, True, True): N = 512 spins, nnz = 774144
```

## Choosing the cut-off — check J(q)

Long-range RKKY-like couplings in metals oscillate.  Truncating them can make a **spiral** the
ground state of the truncated model even though the full model is ferromagnetic.  magex diagonalises
J(q) on a q-grid and warns when max_q λ(q) > λ(0):

```python
from magex.spinmodel import load_results, jq_stability
for rc in (3.0, 5.0, 6.0, 8.0, 11.0):
    uc = load_results("validation_runs/bccFe_weighted/results.json.gz", rcut=rc)
    lam0, lmax, q = jq_stability(uc)
    print(rc, lam0, lmax, q)        # rc = 5-6 Å: lmax > lam0 -> spiral
```

Stability is necessary but not sufficient: for bcc Fe rcut = 8 Å is stable but ends on a negative
shell and gives T_C ≈ 650 K, whereas the full range (11 Å, 20 shells) gives 1060 K (expt. 1043 K).
Use the full range for metals; insulators (SrMnO3, CrI3) converge with 2–3 shells.

## Model Hamiltonians without DFT

`magex model` (or `magex.models.make_model`) writes the same `results.json` format for textbook
lattices (`chain, square, triangular, honeycomb, kagome, sc, bcc, fcc`):

```bash
magex model tri.json --lattice triangular --a 3.0 --Jnn 2.0 --D 1.0 --dmi interfacial --moment 3
magex model afm.json --lattice sc --Jnn -1.0 --order afm
```

or inline in any input file:

```toml
[spinmodel]
model = {lattice = "triangular", a = 3.0, J = [2.0], D = 1.0, dmi = "interfacial", K = 0.1, moment = 3.0}
```

`J` is a list per neighbour shell (meV, per ordered pair, J > 0 ferromagnetic); `D` is the
nearest-neighbour DMI (interfacial `D (ẑ × r̂)` or bulk `D r̂`); `K` is an easy-axis single-ion
anisotropy along `K_axis`.

## Mean-field estimate

`magex.spinmodel.mean_field_tc(uc)` returns T_MF = (2/3) λ_max[J(0)]/k_B.  Monte Carlo uses it to
choose the temperature grid automatically; classical MC typically gives T_C ≈ 0.7–0.8 T_MF in 3D.

## Higher-order interactions: biquadratic and four-spin

    E_bq = −Σ_{i≠j} B_ij (e_i·e_j)²          (ordered pairs, like J)
    E_4  = −K₄ Σ_plaquettes [(e_i·e_j)(e_k·e_l) + (e_j·e_k)(e_l·e_i) − (e_i·e_k)(e_j·e_l)]

* **from first principles**: `biquadratic = "lloyd"` in `[spinmodel]` uses the finite-angle (Lloyd)
  fits stored in results.json (`lloyd_shells` in `magex run`): for every fitted shell the isotropic J
  is replaced by the bilinear constant of the fit and the biquadratic B is added;
* **model Hamiltonians**: `Bq = [B1, B2, ...]` per shell and `K4` (unique nearest-neighbour plaquettes:
  squares on the square lattice, the three rhombi per site on the triangular lattice) in `model = {...}`;
* LLG, GNEB and Monte Carlo include both terms (the Monte Carlo colouring adds the plaquette
  diagonals so simultaneous updates stay independent); gradients and local energies are checked
  against finite differences in the tests.
