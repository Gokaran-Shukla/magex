# Changelog

## 0.4.0 — universal engine

- atomistic structure builder `[structure]`: sphere / cylinder / cube / ellipsoid / truncated-octahedron
  particles, core–shell, multilayers, random alloys, vacancies, surface roughness, pinned spins, per-material
  moments, anisotropy, damping and temperature, interface exchange scaling, temperature profiles
- extra energy terms: exact FFT dipole–dipole interaction, cubic and higher-order uniaxial anisotropy,
  Néel surface anisotropy, generic four-spin quadruplets (LLG, Monte Carlo, minimisers, GNEB)
- initial textures: spirals, (anti)skyrmions and skyrmion lattices with automatic helicity, bubbles,
  hopfions, domain walls
- atomistic LLG: Depondt, SIB, RK4 and Euler integrators; Zhang–Li STT on arbitrary lattices; SOT /
  Slonczewski fields; site-resolved temperature and damping; two-temperature model
- Monte Carlo: heat-bath, uniform and Hinzke–Nowak moves; constrained Monte Carlo with torque curves and K(T);
  longitudinal spin fluctuations
- `magex program`: hysteresis, field cooling, annealing, minimisation, FMR, constrained MC, eigenmodes, LSF
- minimisers: L-BFGS (tangent space, geodesic retraction), nonlinear CG, velocity projection
- harmonic analysis: Hessian, LLG eigenfrequencies, HTST rates, minimum-mode following
- spin–lattice dynamics with bond-length dependent exchange and Langevin lattice thermostat
- observables: resolved magnetisation, chirality, S(q), C(r), autocorrelation; VTK, extended XYZ and
  atomistic OVF output
- micromagnetics (mumax3 feature set): CSG 3D shapes and boolean operations, edge smoothing, Kc2/Kc3,
  magnetoelastic coupling, frozen spins, inter-region DMI, custom field and energy expressions, RK23 /
  RK4 / RK2 / Euler / Heun solvers, moving frame, OVF autosave and field maps
- exporters translate `[structure]`, `[program]` and `[llg]` to VAMPIRE, UppASD and Spirit input; the mumax3
  exporter handles several regions, CSG shapes and the new terms
- docs: complete 96-slide Beamer deck of the formalism (docs/magex_complete_formalism.pdf); tutorials 18–20, feature matrix against VAMPIRE, UppASD, Spirit and mumax3
- speed: multithreaded numba kernels (fused Heun/Depondt steps, ziggurat RNG on chunked xorshift streams,
  compact symmetry-reduced coupling storage, parallel Monte Carlo colour sweeps, fused micromagnetic local
  fields, optional single-precision demag); faster per core than VAMPIRE, UppASD and Spirit on the same CPU
  (docs/benchmarks.md)
- cross-validation: identical m(T) from VAMPIRE, UppASD and magex and identical energies from Spirit for the
  same exported model; exporter keywords checked against the three parsers (UppASD anisotropy file added,
  VAMPIRE roughness/dipole keywords fixed)
- exact dipolar Monte Carlo (incremental field, self-image term); spherical image summation with analytic film
  tail; fixed the sublattice offset sign of the dipolar FFT kernel (multi-sublattice systems)
- sparse/matrix-free Hessians, shift-invert eigenmodes and LLG frequencies for 10⁴–10⁶ spins; HTST with zero
  modes and refined saddles (`[gneb] htst = true`)
- constrained Monte Carlo with dipolar and higher-order on-site terms; Langevin longitudinal spin dynamics
- spin–lattice dynamics: palindromic (second-order) splitting, symplectic spin sub-step, J(r) on all shells

## 0.3.0

- packaging: PyPI trusted-publishing workflow, conda recipe, ReadTheDocs/MkDocs site, executed notebooks,
  example smoke tests in CI, JOSS paper draft (paper/)

- `magex damping`: Gilbert damping tensor from the Wannier Hamiltonian (torque-correlation /
  Kambersky formula; SOC- and exchange-torque forms; intra- and interband parts; direction
  dependence; electronic temperature; MPI over k); alpha can be read by `magex llg` and
  `magex micromag` (`alpha = {from = "damping_n001.json", eta_meV = 50}`)
- quantum thermostat for the atomistic LLG (Planck-spectrum coloured noise, `thermostat = "quantum"`)
- RPA (Tyablikov) ordering temperature with sublattices and anisotropy gap, quantum (S+1)/S
  rescaling; `magex tc` and extra fields in the Monte Carlo `tc_estimate.json`
- `magex sqw`: dynamical structure factor S(q,w) from atomistic LLG with LSWT overlay
- micromagnetic stage `dispersion`: spin-wave dispersion (sinc source, 2D FFT) with analytic
  thin-film overlay incl. interfacial-DMI non-reciprocity; local time-dependent fields (`B_local`)
- `magex gneb`: geodesic nudged elastic band with climbing image for atomistic and micromagnetic
  systems; BB minimiser on the spheres; Arrhenius lifetimes; examples for skyrmion collapse and
  MRAM thermal stability
- two-sublattice micromagnetics (`[afm]`): antiferromagnets, ferrimagnets and d-wave altermagnets
  (sublattice-anisotropic stiffness), homogeneous/inhomogeneous inter-sublattice exchange, demag of
  the total magnetisation, SOT; chirality-resolved spin-wave dispersion; AFM domain-wall velocity example
- spin-transfer torques: Zhang-Li (current in plane, non-adiabatic beta) and Slonczewski (CPP, MTJ
  free layers, angle-dependent efficiency); Thiele stage (gyrovector, dissipation tensor, predicted
  velocity and skyrmion Hall angle); texture tracking in run stages
- SOT efficiency from spin/orbital Hall conductivities (orbital torque via eta_LS, transparency,
  finite thickness); spin pumping (Delta alpha, DC spin current, ISHE voltage)
- biquadratic and four-spin interactions in the atomistic spin model (LLG, GNEB, Monte Carlo);
  biquadratic constants from the Lloyd finite-angle fits (`biquadratic = "lloyd"`)
- finite temperature in micromagnetics: Callen-Callen scaling of Ms, A, K, D from mean field or a
  Monte Carlo m(T) (`temperature_K`, `mc_results`); Landau-Lifshitz-Bloch solver and two-temperature
  model; ultrafast demagnetisation example
- micromagnetic extras: Oersted fields (Biot-Savart FFT), RKKY bilinear + biquadratic interlayer
  coupling across spacers, exchange bias, Voronoi grains (Ku/Ms/axis dispersion, grain-boundary
  exchange), parameter maps from .npy/OVF
- Monte Carlo methods: parallel tempering (replica exchange) and Wang-Landau (1/t, numba kernel)
  with density of states, entropy and free energy; `method = ...` in [mc]
- micromagnetic checkpoint/restart (`checkpoint_ps`, `[output] restart = true`); experimental CuPy GPU
  backend (`Micromagnet.to_gpu()`, self-check `python -m magex.micromag.gpu`)
- interoperability: `magex import-tb2j` (pickle or exchange.out), `magex export --to vampire|uppasd|spirit|mumax3`

## 0.2.0 (2026-11-12)

Simulation engines on top of the exchange calculation.

- `magex llg`: atomistic stochastic LLG (Heun) with full 3x3 exchange tensors, DMI, single-ion anisotropy
  and Zeeman; field off/on comparison, annealing, real-space textures, topological charge on any lattice, GIFs
- `magex mc`: Metropolis Monte Carlo with adaptive moves, over-relaxation and graph-coloured vectorised
  sweeps; automatic FM/AFM/ferrimagnetic order parameter; Binder-cumulant T_C / T_N; parallel temperatures
- J(q) stability check (warns when a coupling cut-off makes a spiral the ground state); mean-field T_C
- `magex micromag`: finite-difference micromagnetics — Newell demag with FFT (open/periodic), exchange,
  interfacial/bulk DMI with chiral boundary conditions, Ku1/Ku2/cubic anisotropy, Zeeman (static, pulse,
  ac, sinc, non-uniform), spin-orbit torques, RK45 / stochastic Heun / Barzilai-Borwein / relax /
  hysteresis, regions and multilayers with inter-region exchange scaling, geometries incl. PNG masks,
  OVF 2.0 I/O, material library
- `magex params`: atomistic -> micromagnetic parameters (A, D_int, D_bulk, K_u, Ms)
- `magex model`: model Hamiltonians on standard lattices in results.json format
- publication figure style (vector PDF + 300 dpi PNG, Okabe-Ito palette, HSL texture colouring)
- results.json.gz support; bundled results for bcc Fe, SrMnO3 and CrI3
- tutorials (docs/tutorials) and example inputs (examples/spindyn, examples/micromag)
- tests: Larmor, energy conservation, Langevin, MC limits, topological charge, demag sum rules,
  Bloch and DMI wall energies, OVF round trip, SOT torque sign, atomistic-continuum spiral energies

## 0.1.0 (2026-11-11)

First public release.

- Wannier90 input: collinear (up/dn) and spinor hr.dat, .win, centres, POSCAR; home-cell shift of Wannier functions
- Atomic xi L.S on Wannier orbitals (s, p, d, f) with L built in the W90 real-harmonic basis
- On-site (LKAG/TB2J) and rotationally invariant weighted-partition Green's-function engines
- Ozaki pole and semicircle energy integration
- Full exchange tensor, DMI and single-ion anisotropy from one Hamiltonian (three references, joint fit)
- Band-energy MAE with time-reversal and spin-model cross-checks
- Space-group symmetrisation (spglib)
- Bruno renormalisation and ligand downfolding
- Exact finite-angle (Lloyd) pair energies with bilinear + biquadratic fits
- Orbital-resolved exchange
- Linear spin-wave theory with full tensors, DMI and SIA
- MPI parallelism over k-points and energy nodes
- Exactness test suite; validation on SrMnO3, bcc Fe and CrI3
