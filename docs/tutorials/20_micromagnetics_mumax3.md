# 20 · mumax3-style micromagnetics

The micromagnetic engine (`magex micromag`) now covers the mumax3 feature set used in most scripts.
`examples/universal/micromag_mumax3_features.toml` uses all of the items below and exports to an `.mx3` script
with `magex export ... --to mumax3`.

| mumax3 | magex TOML |
|---|---|
| `Ball`, `Cylinder`, `Cuboid`, `Ellipsoid`, `Cone`, `.Add/.Intersect/.Sub/.Xor`, `.Transl`, `.RotZ` | `shape = "sphere" \| "cylinder" \| "cuboid" \| "ellipsoid" \| "cone" \| "union" \| "intersect" \| "difference" \| "xor"`, `shapes = [...]`, `center_nm = [x,y,z]`, `rotate_deg`, `rotate_x_deg`, `rotate_y_deg` |
| `EdgeSmooth = n` | `edge_smooth = n` in a region (sub-cell volume fractions in the demag field) |
| `Kc1, Kc2, Kc3` | `Kc1`, `Kc2` (E = Kc2 a₁²a₂²a₃²), `Kc3` (E = Kc3 (Σ a_i²a_j²)²) |
| `B1, B2, exx..eyz` | `B1`, `B2` per region, `[magnetoelastic] strain = [exx, eyy, ezz, exy, exz, eyz]` (or an .npy map) |
| `FrozenSpins` | `frozen = true` per region |
| `ext_InterDind(r1, r2, D)` | `[[inter_dmi]] regions = [r1, r2], D = ...` (link DMI constants, energy-consistent) |
| `ext_ScaleExchange` | `[[interlayer]] regions, scale` |
| `AddFieldTerm`, `AddEdensTerm` | `[[custom_field]] B = ["...", "...", "..."]`, `[[custom_energy]] density = "..."`, `constants = {...}` (numpy expressions in x, y, z, t, mx, my, mz) |
| `SetSolver(1..5)`, `FixDt` | `solver = "euler" \| "heun" \| "rk2" \| "rk4" \| "rk23" \| "rk45"`, `dt_fs` in a run stage |
| `ext_centerWall(c)` | `moving_frame = {component = "z", axis = "x"}` (lab-frame wall position and velocity are written) |
| `AutoSave(m, dt)`, `Save(B_demag)` | `autosave_ps = 5`, `save_fields = ["demag", "exchange", ...]` |

All energy terms are checked against finite differences of the energy (`tests/test_universal.py`).  The
moving frame keeps a field-driven wall centred and returns the 1D-model velocity γΔB/α within 3 %.

When D varies in space (several regions or `inter_dmi`), the interfacial DMI uses link constants (mean of
the two cells, or the `inter_dmi` value across a region boundary), so the field is the exact derivative of
the energy.  At free surfaces the chiral boundary condition is used, as in mumax3; there the discrete field
is not the derivative of the discrete energy, by construction.

Not implemented: GPU kernels for the full term set (the CuPy backend is experimental), mumax3's
`Crop`/`Expand` helpers and its scripting language, and the Slonczewski `Lambda` per region.
