# Changelog

## 0.3.0 (in development, branch `dev-0.3`)

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
