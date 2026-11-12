# magex tutorials

magex covers the whole chain from an electronic-structure calculation to device-scale magnetisation
dynamics.  Every stage reads the output of the previous one, and every stage can also be started on its
own from a model Hamiltonian or a material library.

```
 Wannier90 H(k) ──magex run──▶ results.json ──magex llg──▶ atomistic spin dynamics, textures, Q
 (VASP / QE)                  (J_ij tensors,  ──magex mc───▶ M(T), χ, C, Binder → T_C / T_N
                               DMI, SIA, μ)   ──magex params─▶ Ms, A, D, K ──magex micromag──▶ thin films,
                                                                              nanostructures, loops, SOT
```

| # | tutorial | what you get | run time (2 cores) |
|---|----------|--------------|--------------------|
| 1 | [Spin model from results.json](01_spin_model.md) | loading couplings, cut-offs, J(q) stability check, model lattices | seconds |
| 2 | [Atomistic LLG and spin textures](02_llg_textures.md) | skyrmions with field off / on, thermal LLG, Larmor precession, GIFs | ~30 s per run |
| 3 | [Monte Carlo: Curie and Néel temperatures](03_monte_carlo_Tc.md) | T_C of bcc Fe and CrI3, T_N of SrMnO3, finite-size Binder analysis | minutes |
| 4 | [Micromagnetics: first simulation](04_micromagnetics_basics.md) | μMAG standard problem 4, mesh, materials, stages, outputs | ~2 min |
| 5 | [Thin-film recipes](05_thin_film_recipes.md) | skyrmion disk, labyrinth → skyrmions, PMA hysteresis, vortex, SOT switching, B20 helix, multilayers | 1–10 min |
| 6 | [Multiscale: DFT → micromagnetics](06_multiscale.md) | A, D, K, Ms from first-principles exchange; CrI3 flake | minutes |
| 7 | [Python API](07_python_api.md) | scripting all solvers, custom fields, custom geometries | – |
| 8 | [Gilbert damping from first principles](08_damping.md) | α tensor vs broadening from the Wannier Hamiltonian, feeding LLG and micromagnetics | minutes (MPI) |
| 9 | [Quantum corrections](09_quantum_statistics.md) | RPA T_C, (S+1)/S rescaling, quantum (Planck) thermostat for LLG | seconds |
| 10 | [Magnon spectra from dynamics](10_magnon_spectra.md) | S(q, ω) from LLG vs LSWT; micromagnetic spin-wave dispersion, DMI non-reciprocity | 1 min |
| 11 | [Energy barriers and lifetimes](11_energy_barriers.md) | GNEB: skyrmion collapse, MRAM thermal stability Δ, Arrhenius lifetimes | seconds–minutes |
| 12 | [Antiferro-, ferri- and altermagnets](12_antiferromagnets.md) | two-sublattice micromagnetics, AFMR, chiral magnon splitting, SOT-driven AFM walls | 1–2 min |
| 13 | [Spin-transfer and spin-orbit torques](13_spin_torques.md) | Zhang–Li, Slonczewski, SOT; skyrmion Hall angle vs Thiele | 1 min |
| 14 | [First-principles torques and spin pumping](14_first_principles_sot.md) | θ from spin/orbital Hall conductivities, orbital torque, Δα, ISHE voltage | seconds |
| – | [Conventions and units](conventions.md) | energy convention, units, sign of DMI, boundary conditions | – |

All inputs live in [`examples/spindyn`](../../examples/spindyn) and [`examples/micromag`](../../examples/micromag);
all figures are written as vector PDF plus 300 dpi PNG with colour-blind-safe palettes, ready for papers.

Validation (all in `tests/`, `pytest -q` runs in < 10 s):

| check | result |
|-------|--------|
| Larmor frequency, ω = γB | exact to 10⁻⁶ |
| energy conservation at α = 0 (Heun, dt = 0.2 fs) | 5 × 10⁻⁶ relative over 2000 steps |
| thermal LLG and MC vs Langevin function L(μB/k_BT) | within statistical error |
| SC classical Heisenberg FM, T_C (exact 33.49 K for J = 1 meV) | 33.29 K (−0.6 %) |
| SC classical Heisenberg AFM, T_N | 32.6 K (−2.7 %, L ≤ 8) |
| Newell demag tensor: cube N = 1/3, trace = 1, thin film N_zz → 1 | 10⁻¹² |
| Bloch wall energy 4√(AK) | 0.06 % |
| DMI wall energy 4√(AK) − π\|D\| (chiral boundary conditions) | 0.03 % |
| μMAG SP4 s-state ⟨m⟩ = (0.967, 0.125, 0) and first ⟨m_x⟩ zero crossing ≈ 0.137 ns | (0.9670, 0.1253, 0), 0.136 ns |
| atomistic → continuum A and D (spiral energies) | 0.05 % |
