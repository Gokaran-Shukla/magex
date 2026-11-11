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
