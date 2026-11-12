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
| Exact dipole–dipole (FFT; exact incremental field in Monte Carlo) | ✓ | ✓ | ✓ | ✓ | ✓ (demag) |
| Shapes, core–shell, multilayers, alloys, vacancies, roughness | ✓ | ✓ | ~ (via input files) | ~ | ✓ (CSG) |
| Pinned spins, temperature gradients | ✓ | ✓ | ✓ | ✓ (pinning) | ✓ (frozen spins) |
| Stochastic LLG: Heun, Depondt, SIB, RK4 (multithreaded kernels) | ✓ | ✓ (Heun) | ✓ | ✓ | ✓ (micromag.) |
| Quantum (Planck) thermostat | ✓ | – | ✓ | – | – |
| Spin-transfer / spin-orbit torques | ✓ | ✓ | ✓ | ✓ | ✓ |
| Two-temperature model | ✓ | ✓ | ~ | – | – |
| Metropolis, heat-bath, Hinzke–Nowak moves | ✓ | ✓ | ✓ | ✓ (MC) | – |
| Parallel tempering, Wang–Landau | ✓ | – | ~ | – | – |
| Constrained Monte Carlo K(T) (incl. dipolar and higher-order on-site terms) | ✓ | ✓ | – | – | – |
| Longitudinal spin fluctuations (MC and Langevin dynamics) | ✓ | – | ✓ | – | – |
| Hysteresis, field cooling, FMR, annealing programs | ✓ | ✓ | ✓ | ~ | ✓ |
| Minimisers (L-BFGS, NCG, VP) | ✓ | – | ~ | ✓ | ✓ (BB) |
| GNEB with climbing image | ✓ | – | ✓ | ✓ | – |
| HTST rates (with zero modes), eigenmodes, minimum-mode following | ✓ (sparse eigenmodes; HTST dense, ≲ 3000 spins) | – | – | ✓ | – |
| S(q,ω), linear spin waves | ✓ | – | ✓ | ~ | ~ |
| Spin–lattice dynamics | ~ (harmonic springs + J(r) on all shells, symplectic) | – | ✓ | – | – |
| RPA / mean-field T_C, Callen–Callen scaling, LLB | ✓ | ~ | ~ | – | – |
| Gilbert damping from first principles | ✓ | – | – | – | – |
| Micromagnetics: demag, exchange, DMI, anisotropy, Zeeman | ✓ | – | – | – | ✓ |
| Micromagnetics: two-sublattice AFM / altermagnets | ✓ | – | – | – | – |
| Micromagnetics: magnetoelastic, custom terms, moving frame, edge smoothing | ✓ | – | – | – | ✓ |
| GPU | ~ (experimental CuPy, micromagnetics) | ✓ (CUDA) | – | ✓ (CUDA) | ✓ |
| MPI / distributed spin dynamics | ~ (MPI for exchange; processes for MC) | ✓ | ~ (OpenMP) | ~ (OpenMP) | – |
| Graphical interface | – (figures, VTK/XYZ/OVF for ParaView/OVITO) | – | – | ✓ | ~ (web GUI) |

**Speed and agreement.** On the same CPU, magex's atomistic LLG and Monte Carlo kernels are faster per core
than serial VAMPIRE, OpenMP UppASD and CPU Spirit, and the same exported model gives the same m(T) in VAMPIRE,
UppASD and magex and the same energy in Spirit ([benchmarks](benchmarks.md)).

**Where the others are still ahead.** GPUs (VAMPIRE and Spirit CUDA back ends, mumax3) and distributed-memory
MPI (VAMPIRE) for 10⁷–10⁸ spins; magex runs on the cores of one node.  UppASD's spin–lattice dynamics offers
more lattice potentials.
