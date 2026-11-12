# Feature matrix: magex and the established spin-simulation codes

magex aims to run, from one input format and one Python API, the simulations people usually split over
VAMPIRE (atomistic, Monte Carlo + LLG), UppASD (atomistic spin dynamics), Spirit (atomistic, energy
landscapes) and mumax3 (GPU micromagnetics) — and to start from the exchange parameters it computes from
Wannier functions itself.  ✓ = available, ~ = partial or simplified, – = not available.  Entries for the
other codes summarise their documented main features and are not exhaustive; versions differ.

| Capability | magex | VAMPIRE | UppASD | Spirit | mumax3 |
|---|---|---|---|---|---|
| Exchange tensors from first principles (Wannier90) | ✓ | – | – | – | – |
| TB2J import, exporters to the other four codes | ✓ | – | – | – | – |
| Full exchange tensor, DMI, single-ion anisotropy | ✓ | ✓ | ✓ | ~ (no symmetric anisotropic exchange) | n/a |
| Biquadratic, four-spin, quadruplets | ✓ | ~ | ✓ | ✓ | n/a |
| Cubic, higher-order uniaxial, Néel surface anisotropy | ✓ | ✓ | ~ | ~ | ✓ (Kc1–3) |
| Exact dipole–dipole (FFT) | ✓ | ✓ | ✓ | ✓ | ✓ (demag) |
| Shapes, core–shell, multilayers, alloys, vacancies, roughness | ✓ | ✓ | ~ (via input files) | ~ | ✓ (CSG) |
| Pinned spins, temperature gradients | ✓ | ✓ | ✓ | ✓ (pinning) | ✓ (frozen spins) |
| Stochastic LLG: Heun, Depondt, SIB, RK4 | ✓ | ✓ (Heun) | ✓ | ✓ | ✓ (micromag.) |
| Quantum (Planck) thermostat | ✓ | – | ✓ | – | – |
| Spin-transfer / spin-orbit torques | ✓ | ✓ | ✓ | ✓ | ✓ |
| Two-temperature model | ✓ | ✓ | ~ | – | – |
| Metropolis, heat-bath, Hinzke–Nowak moves | ✓ | ✓ | ✓ | ✓ (MC) | – |
| Parallel tempering, Wang–Landau | ✓ | – | ~ | – | – |
| Constrained Monte Carlo K(T) | ✓ | ✓ | – | – | – |
| Longitudinal spin fluctuations | ✓ (MC) | – | ✓ | – | – |
| Hysteresis, field cooling, FMR, annealing programs | ✓ | ✓ | ✓ | ~ | ✓ |
| Minimisers (L-BFGS, NCG, VP) | ✓ | – | ~ | ✓ | ✓ (BB) |
| GNEB with climbing image | ✓ | – | ✓ | ✓ | – |
| HTST rates, eigenmodes, minimum-mode following | ✓ (dense, small systems) | – | – | ✓ | – |
| S(q,ω), linear spin waves | ✓ | – | ✓ | ~ | ~ |
| Spin–lattice dynamics | ~ (harmonic + J(r)) | – | ✓ | – | – |
| RPA / mean-field T_C, Callen–Callen scaling, LLB | ✓ | ~ | ~ | – | – |
| Gilbert damping from first principles | ✓ | – | – | – | – |
| Micromagnetics: demag, exchange, DMI, anisotropy, Zeeman | ✓ | – | – | – | ✓ |
| Micromagnetics: two-sublattice AFM / altermagnets | ✓ | – | – | – | – |
| Micromagnetics: magnetoelastic, custom terms, moving frame, edge smoothing | ✓ | – | – | – | ✓ |
| GPU | ~ (experimental CuPy, micromagnetics) | ✓ (CUDA) | – | ✓ (CUDA) | ✓ |
| MPI / distributed spin dynamics | ~ (MPI for exchange; processes for MC) | ✓ | ~ (OpenMP) | ~ (OpenMP) | – |
| Graphical interface | – (figures, VTK/XYZ/OVF for ParaView/OVITO) | – | – | ✓ | ~ (web GUI) |

**Where the others are still ahead.** Raw speed for very large systems: VAMPIRE (MPI, CUDA), Spirit
(CUDA) and mumax3 (GPU) run 10⁶–10⁸ spins or cells; magex is vectorised NumPy/SciPy (+numba kernels) and is
comfortable up to ~10⁵–10⁶ spins on one node.  UppASD's spin–lattice dynamics and its quantum/LSF dynamics
are more complete, and Spirit's HTST handles zero modes.  These are the next items on the roadmap.
