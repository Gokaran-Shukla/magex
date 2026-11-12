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
| HTST rates (with zero modes), eigenmodes, minimum-mode following | ✓ (sparse LU + Lanczos: 40 000 spins in 37 s) | – | – | ✓ | – |
| S(q,ω), linear spin waves | ✓ | – | ✓ | ~ | ~ |
| Spin–lattice dynamics | ✓ (harmonic, FPU, Morse, Lennard-Jones; J(r) on all shells; symplectic) | – | ✓ | – | – |
| RPA / mean-field T_C, Callen–Callen scaling, LLB | ✓ | ~ | ~ | – | – |
| Gilbert damping from first principles | ✓ | – | – | – | – |
| Micromagnetics: demag, exchange, DMI, anisotropy, Zeeman | ✓ | – | – | – | ✓ |
| Micromagnetics: two-sublattice AFM / altermagnets | ✓ | – | – | – | – |
| Micromagnetics: magnetoelastic, custom terms, moving frame, edge smoothing | ✓ | – | – | – | ✓ |
| GPU | ~ (CUDA kernels for atomistic LLG/MC and cuFFT demag; verified in the CUDA simulator, not yet timed on a GPU) | ✓ (CUDA) | – | ✓ (CUDA) | ✓ |
| MPI / distributed spin dynamics | ✓ (slab decomposition with halo exchange for LLG and MC; set-up replicated) | ✓ | ~ (OpenMP) | ~ (OpenMP) | – |
| Graphical interface | – (figures, VTK/XYZ/OVF for ParaView/OVITO) | – | – | ✓ | ~ (web GUI) |

**Speed and agreement.** On the same CPU, magex's atomistic LLG and Monte Carlo kernels are faster per core
than serial VAMPIRE, OpenMP UppASD and CPU Spirit, and the same exported model gives the same m(T) in VAMPIRE,
UppASD and magex and the same energy in Spirit ([benchmarks](benchmarks.md)).

**Where the others are still ahead.** Proven GPU performance: VAMPIRE and Spirit have production CUDA back ends
and mumax3 is GPU-native, while magex's CUDA kernels are verified but not yet benchmarked on hardware.
VAMPIRE's MPI version distributes the system set-up as well and has been run on thousands of cores; magex
replicates the set-up on every rank.
