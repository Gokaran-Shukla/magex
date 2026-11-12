# 19 · Simulation programs, integrators, minimisers and harmonic analysis

`magex program input.toml` runs the VAMPIRE-style programs (`sim:program`), the Spirit methods and the
UppASD modes on any `[spinmodel]` + `[structure]`.  Every program writes CSV tables, a figure, and the final
spins as NPZ/CSV, VTK (ParaView: Glyph filter) and extended XYZ (OVITO).

| `type` | what it does | key options |
|---|---|---|
| `hysteresis` | field loop ±Bmax, every point relaxed (T = 0) or sampled by MC (T > 0) | `Bmax`, `npts`, `direction`, `method`, `temperature` |
| `field_cooling` | MC cooling in a field, m(T), E(T) | `T_start`, `T_end`, `nT`, `sweeps`, `field` |
| `anneal` | simulated annealing (geometric schedule) + minimiser | `T_start`, `T_end`, `nT`, `sweeps` |
| `minimize` | relax a texture | `method = lbfgs \| ncg \| vp \| llg`, `tol` |
| `fmr` | field step + undamped ring-down, FFT → resonance | `B_pulse`, `alpha`, `dt`, `steps` |
| `cmc` | constrained Monte Carlo, torque curve and K(T) | `temperatures`, `torque_curve` |
| `eigenmodes` | Hessian eigenvalues and LLG eigenfrequencies | `nmodes` |
| `lsf` | Monte Carlo with longitudinal spin fluctuations | `temperatures`, `b`, `measure` |

## Checks that come with the code

* **Hysteresis** of a uniaxial macrospin: coercive field = anisotropy field 2K/(μμ_B) (Stoner–Wohlfarth).
* **FMR** of a uniaxial bulk ferromagnet: f = γ(B + B_K)/2π (Kittel); `examples/universal/fmr.toml` gives
  76.55 GHz against 76.5 GHz.
* **Constrained MC**: K(T)/K(0) follows the Callen–Callen m³ law for single-ion anisotropy
  (`examples/universal/cmc_anisotropy.toml`, `tests/test_universal.py::test_constrained_mc_callen_callen`).

## Atomistic integrators (`[llg] integrator = ...`)

`heun` (stochastic Heun, default) · `depondt` (Heun on rotation vectors, exact norm, Depondt & Mertens 2009)
· `sib` (semi-implicit midpoint by Cayley transform, Mentink et al. 2010) · `rk4` · `euler`.
All five reproduce Larmor precession to 1e-4 rad after 1000 steps.

## Torques in the atomistic LLG

```toml
[llg]
stt = {u = [40.0, 0, 0], beta = 0.2}            # Zhang-Li, u in m/s (least-squares lattice gradients)
sot = {p = [0, 1, 0], B_DL = 0.05, B_FL = 0.0}  # damping-/field-like SOT or Slonczewski fields (T)
ttm = {T0 = 300, P0 = 1e21, t0 = 0.2e-12, tau = 0.05e-12}   # two-temperature model drives the noise
```

`examples/universal/zhang_li_wall.toml`: a wall with a hard axis moves at v = (β/α)u below Walker breakdown
(78.5 m/s for β/α·u = 80 m/s), and stops for β = 0 — the textbook result.

## Monte Carlo moves (`[mc] move = ...`)

`gaussian` (adaptive cone, default) · `uniform` · `hinzke` (VAMPIRE's mixture of cone, uniform and
spin-flip moves) · `heatbath` (exact sampling of exp(βH·e) for the linear local field, Metropolis-corrected
for anisotropy and nonlinear terms; acceptance ≈ 1).  All four give the same E(T) and m(T).

**Longitudinal spin fluctuations** (`type = "lsf"`, Ma & Dudarev 2011): moments M_i = s_i e_i with a Landau
term a s² + b s⁴ chosen so that s = 1 at T = 0; the moment survives above T_C.

## Minimisers (`magex.minimize`)

`lbfgs` (tangent-space L-BFGS with geodesic retraction, Spirit's LBFGS-OSO idea) · `ncg` (Polak–Ribière+) ·
`vp` (velocity projection) · `llg` (damped dynamics).  The line search falls back to a gradient test when
energy differences reach round-off, so torques of 1e-7 meV are reached.

## Harmonic analysis (`magex.htst`)

```python
from magex.htst import hessian, llg_frequencies, htst_rate, mmf
f = llg_frequencies(system, e_min)                     # magnon eigenfrequencies (GHz)
r = htst_rate(system, e_min, e_saddle, T=300)          # rate (1/s), prefactor, barrier
e_sp, info = mmf(system, e_min)                        # minimum-mode following to a saddle point
```

HTST follows Bessarab, Uzdin & Jónsson (2012) for systems without zero modes; for a biaxial macrospin the
prefactor equals the FMR frequency of the minimum, which is tested.  Eigenfrequencies of a ferromagnetic
ring match the analytic magnon dispersion to 1e-8.  The dense Hessian limits these tools to a few thousand
spins; combine them with `magex gneb` for the saddle point.

## Spin–lattice dynamics (`magex.sld.SpinLattice`)

Harmonic springs plus a bond-length dependent nearest-neighbour exchange J(r) = J₀ exp(−(r−r₀)/λ) (or
Ma's cubic form) on top of the full static spin model; velocity Verlet with an optional Langevin lattice
thermostat around one LLG step.  Without damping the total energy is conserved, with a drift that falls as
the time step is reduced; with both thermostats on, the lattice equilibrates at the set temperature.

## Observables (`magex.observables`)

Sublattice-, material- and layer-resolved magnetisation, scalar and vector chirality, S(q) (grid FFT or
arbitrary q), shell-averaged C(r), autocorrelation A(t), and VTK / extended-XYZ / OVF writers.
