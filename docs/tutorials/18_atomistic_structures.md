# 18 · Atomistic structures, extra energy terms and initial textures

Everything in this tutorial is driven by a `[structure]` table that `magex llg`, `magex mc`, `magex sqw`,
`magex gneb` and `magex program` all read.  It edits the supercell built from `[spinmodel]` *before* the
simulation starts, which is how VAMPIRE (`create:*`, `material:*`), UppASD (`posfile`, `momfile`, random
alloys) and Spirit (geometry, defects, pinning) set up their systems.

## Shapes, materials and defects

```toml
[structure]
shape = "sphere"              # sphere | cylinder | cube | ellipsoid | truncated_octahedron | film | none
size = 20.0                   # radius (A); [a, b, c] semi-axes for an ellipsoid; edge for a cube
center = [30, 30, 30]         # A (default: centre of the supercell)
core_shell = {radius = 12.0, core = "core", shell = "shell"}
layers = [{material = "Co", thickness = 3}, {material = "Pt", thickness = 1}]   # along z, unit cells
alloy = {host = "core", guest = "Ni", fraction = 0.2, seed = 1}
vacancies = 0.05              # or {fraction, seed, material}
roughness = {amplitude = 1, correlation = 15.0, surface = "top"}                 # Voronoi islands
materials = [
  {name = "core",  Kc = 0.02, cubic_axes = [[1,0,0],[0,1,0],[0,0,1]], alpha = 0.05},
  {name = "shell", moment_scale = 0.7, ks = 0.05, Ku = [0.1, [0, 0, 1]], K2 = 0.01},
]
exchange_scale = [[1, 1, 1], [1, 1, -0.3], [1, -0.3, 0.6]]    # J_ij x f[mat_i, mat_j] (bulk, core, shell)
neel_surface = 0.05           # Neel pair anisotropy on all under-coordinated sites (meV)
dipolar = true                # exact dipole-dipole field by FFT (or {images = 6} for periodic films)
quadruplets = {K = 0.1, list = [[0, 1, 2, 3]]}
pinned = [{region = "bottom", layers = 1, direction = [1, 0, 0]}]    # also top, xmin..ymax, material:<name>
temperature_profile = {axis = "x", T0 = 10, T1 = 300}               # or {spot = [x,y,z], sigma, T0, T1}
```

* **Carving** removes sites and all their couplings (exchange tensors, DMI, biquadratic, plaquettes).
* **Materials** are labels on sites.  `moment_scale`, `anisotropy_scale`, an extra uniaxial `Ku`, cubic
  `Kc` (E = Kc Σ a_i²a_j², Kc > 0: easy ⟨100⟩), higher uniaxial `K2, K3` (−K2 (e·u)⁴ − K3 (e·u)⁶), per-site
  damping `alpha` and a site temperature `T` can be set per material.
* **Interface exchange**: `exchange_scale` multiplies every coupling J_ij by f[material_i, material_j];
  a negative entry turns a ferromagnetic interface antiferromagnetic (exchange bias, synthetic AFMs).
* **Néel surface anisotropy** (VAMPIRE convention) E_i = (k_s/2) Σ_{j∈nn} (e_i·r̂_ij)², applied to sites with
  fewer nearest neighbours than the bulk; it is folded into the single-ion tensor, so every solver sees it.
* **Dipolar**: B_dip,i = (μ₀μ_B/4π) Σ_j μ_j [3(e_j·r̂)r̂ − e_j]/r³ for every pair, by FFT on the supercell grid
  (zero padding on open axes, explicit image sums on periodic ones).  It reproduces the direct sum to
  machine precision and the thin-film shape anisotropy μ₀M_s²/2 to < 1 % (`tests/test_universal.py`).
  In Monte Carlo the dipolar field is refreshed once per colour group (as in VAMPIRE's macrocell update).
* **Pinned spins** are excluded from Monte Carlo moves and have zero torque and noise in the LLG.
* **Temperature profiles** give every site its own temperature in Monte Carlo (local β) and in the
  stochastic LLG (local noise amplitude): thermal gradients, laser spots.

## Initial textures

`init` in `[llg]` / `[program]` (and `system.init_spins` in Python) accepts, besides
`random | ferro | reference | file`:

| kind | parameters |
|---|---|
| `spiral` | `q` (1/Å) or `q_frac` (r.l.u.), cycloid by default, `helix = true` |
| `skyrmion`, `skyrmions`, `bubble` | `radius`, `wall` (Å), `centers = [[x,y],…]`, `polarity = ±1`, `vorticity = ±1` (antiskyrmion), `helicity = "auto"` (picks the Néel/Bloch helicity favoured by the DMI) |
| `hopfion` | `radius`, `height`, `center` (3D) |
| `domain` | `normal = "x"`, `position`, `width`, `type = "neel" \| "bloch"` |

```toml
[llg]
init = "skyrmions"
texture = {centers = [[30, 30], [80, 60]], radius = 6.0}
```

## Example

`examples/universal/core_shell_particle.toml` field-cools a 2.4 nm-radius core–shell particle with an
antiferromagnetic interface, cubic core anisotropy, Néel surface anisotropy on the shell and full dipolar
interaction (`magex program core_shell_particle.toml`).  The core orders first along the field and the
shell then freezes antiparallel to it, which is the interface-frozen (exchange-bias) state, although the
reverse arrangement has the lower Zeeman energy.
