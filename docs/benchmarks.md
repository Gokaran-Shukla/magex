# Benchmarks and cross-validation against VAMPIRE, UppASD and Spirit

All numbers were measured on the same machine (2 CPU cores, no GPU) with the codes built from their GitHub
sources (VAMPIRE `make serial`, UppASD CMake Release with OpenMP, Spirit core CMake Release, CPU build).
The scripts are in [`benchmarks/`](https://github.com/Gokaran-Shukla/magex/tree/main/benchmarks).
Timings vary by ±10 % between runs on this shared machine; the model is the simple-cubic Heisenberg
ferromagnet J = 1 meV (magex convention), μ = 2 μ_B, exported with `magex export`.

## Speed

| task (simple cubic, 32³ = 32 768 spins) | magex, 1 thread | magex, 2 threads | VAMPIRE (serial) | UppASD (2 OpenMP threads) | Spirit (CPU) |
|---|---|---|---|---|---|
| stochastic LLG, Heun / Depondt, ns per spin per step | **55–64** | **29** | 74 | 99 | 204 |
| Metropolis Monte Carlo, ns per spin per sweep | **80** | **50** | 110 (13³) | ~160 (13³, incl. set-up) | – |

What makes the difference: the coupling matrix is stored once per symmetry-distinct bond (isotropic,
isotropic + DMI or full 3×3 blocks), a whole Heun/Depondt step is two fused multithreaded loops, the thermal
field comes from a ziggurat Gaussian generator on per-chunk xorshift streams (bit-reproducible for any number
of threads), and Monte Carlo colour groups are swept in parallel.  Small systems (< 2048 sites per group) use
serial twins of the kernels to avoid thread start-up costs.

Micromagnetics: the local fields (exchange, DMI with chiral boundary conditions, uniaxial and cubic anisotropy)
are one fused kernel, about 15× faster than the previous NumPy stencil; the demag convolution is FFT-bound
(256×256×1: 34 ms in double, 18 ms with `[demag] precision = "single"`, relative error 2×10⁻⁷, which is
what mumax3 uses on GPUs).

## Accuracy: the same model in four codes

Magnetisation m(T) of the 13³ simple-cubic ferromagnet (3000 thermalisation + 6000 measurement sweeps),
exported from one magex `results.json`:

| T (K) | VAMPIRE | UppASD | magex |
|---|---|---|---|
| 20 | 0.751 | 0.751 | 0.750 |
| 28 | 0.578 | 0.569 | 0.578 |
| 32 | 0.419 | 0.399 | 0.411 |
| 36 | 0.134 | 0.125 | 0.124 |
| 40 | 0.048 | 0.069 | 0.068 |

The three codes agree within the critical fluctuations of a 2197-spin system, which confirms the exchange
conversions of the exporters (VAMPIRE J = 2 J_magex in joules; UppASD J = J_magex in mRy).

* **Spirit**: the total energy of a perturbed skyrmion configuration (J, DMI, anisotropy, tilted field)
  agrees with magex to 3×10⁻⁸ relative, after removing the constant −K/3 per spin that comes from magex
  storing the single-ion tensor traceless.
* **LLG relaxation** at T = 10 K, α = 1 (32³): m after 0.2 ps is 0.917 in magex and 0.918 in UppASD.
* **Anisotropy signs**: a VAMPIRE LLG run of the exported uniaxial model relaxes onto the magex easy
  axis; the UppASD kfile sign (K1 < 0 = easy axis) was taken from UppASD's field routine.
* **Keywords**: every keyword the exporters write was checked against the parsers in the VAMPIRE, UppASD and
  Spirit sources (this fixed `create:interfacial-roughness` and added the UppASD anisotropy file).

## Exactness upgrades in this release

* Dipolar Monte Carlo is exact: single-spin updates with the full dipolar field kept up to date
  incrementally (O(N) per accepted move), including each spin's coupling to its own periodic images.
  It matches brute-force Metropolis on full energies (tests/test_fast.py).
* Periodic dipolar sums use spherical image summation, with an analytic far-field tail for films, so the
  shape anisotropy converges to 3×10⁻⁴ at 2 images and 10⁻⁷ at 4.  A sign error in the sublattice offset
  of the FFT kernel (multi-sublattice systems only) was found by the direct-sum test and fixed.
* Spin–lattice dynamics: the splitting is now palindromic (the earlier version was only first order), and the
  spin sub-step is a symplectic colour-group sweep, so the energy error is second order and bounded.
* HTST handles zero (Goldstone/translational) modes, with translational zero-mode volumes computed from
  dE/dx; saddle points are polished by minimum-mode following before the harmonic analysis
  (`[gneb] htst = true`).
* Hessians are sparse (local terms) plus matrix-free (dipolar, higher order); eigenmodes and LLG
  frequencies use shift-invert Lanczos: the k = 0 mode of a 22 500-spin film comes out at the exact Larmor
  frequency in 10 s.

## Where the others still lead

* **GPUs**: VAMPIRE and Spirit have CUDA back ends and mumax3 is GPU-only; magex has no tested GPU path
  (the CuPy micromagnetics backend is experimental).  On a single GPU, mumax3 will outrun magex's CPU
  micromagnetics for large meshes.
* **Distributed memory**: VAMPIRE's MPI decomposition scales to thousands of cores; magex is
  shared-memory (threads) for spin dynamics and Monte Carlo.
* **Mature workflows**: UppASD's spin–lattice dynamics has more lattice potentials, Spirit's GUI is
  interactive, and all three have years of production use.
