# Examples

| file | system / route |
|---|---|
| `bccFe.toml` | bcc Fe, collinear QE + Wannier90 (s, p, d) |
| `SrMnO3.toml` | G-type AFM SrMnO3, collinear VASP + Wannier90 (Mn d, O p), ligand downfolding |
| `collinear_plus_soc.toml` | DMI, anisotropic exchange, SIA and MAE from one collinear run + atomic SOC |
| `spinor.toml` | fully relativistic (spinor) Wannier90 input |
| `slurm_40cores.sh` | SLURM job script for one 40-core node |

The Wannier90 files for bcc Fe, SrMnO3 and CrI3 are in the TB2J example repository:

```bash
git clone --depth 1 https://github.com/mailhexu/TB2J_examples
# bcc Fe:  TB2J_examples/Wannier/bccFe_QE_Wannier90/W90/
# SrMnO3:  TB2J_examples/Wannier/SrMnO3_VASP_Wannier/Wannier90/
# CrI3:    TB2J_examples/Wannier/CrI3_VASP_Wannier/z/
```

Copy an input file into the data folder (or edit its paths) and run `magex run <file>.toml`.

## Spin dynamics and Monte Carlo — `spindyn/`

These read the bundled `validation_runs/*/results.json.gz`, so they run without any DFT data.

| file | what it does | command |
|---|---|---|
| `skyrmion_llg.toml` | triangular lattice + interfacial DMI: spirals (B = 0) vs skyrmion lattice (2 T) | `magex llg` |
| `cri3_llg.toml` | CrI3 monolayer, thermal LLG at 15 K, field off / 1 T | `magex llg` |
| `fe_llg_relax.toml` | bcc Fe at 300 K: damped Larmor precession in 10 T | `magex llg` |
| `fe_mc.toml` | Curie temperature of bcc Fe | `magex mc` |
| `srmno3_mc.toml` | Néel temperature of G-type SrMnO3 | `magex mc` |
| `cri3_mc.toml` | Curie temperature of monolayer CrI3 | `magex mc` |

## Micromagnetics — `micromag/`

| file | what it does |
|---|---|
| `sp4.toml` | μMAG standard problem 4 (permalloy, s-state reversal) |
| `skyrmion_disk.toml` | isolated Néel skyrmion in a 100 nm Co/Pt disk, then +200 mT |
| `skyrmion_film_field.toml` | periodic PMA/DMI film: labyrinth (B = 0) → skyrmions (250 mT) |
| `pma_hysteresis.toml` | out-of-plane loop of a CoFeB/MgO nanomagnet |
| `vortex_disk.toml` | vortex in a Py disk, core displacement in an in-plane field |
| `sot_switching.toml` | spin–orbit-torque switching of a perpendicular dot with in-plane bias |
| `feGe_helix.toml` | bulk-DMI helix in B20 FeGe |
| `cri3_from_atomistic.toml` | CrI3 flake with Ms, A, K from the magex exchange (multiscale) |

Run any of them with `magex micromag <file>.toml`; see [`docs/tutorials`](../docs/tutorials/README.md).
