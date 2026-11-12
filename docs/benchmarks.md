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
  monolayer shape anisotropy is converged to 10⁻⁴ at 2 image shells and 4×10⁻⁶ at 8 (10⁻² without the tail).  A sign error in the sublattice offset
  of the FFT kernel (multi-sublattice systems only) was found by the direct-sum test and fixed.
* Spin–lattice dynamics: the splitting is now palindromic (the earlier version was only first order), and the
  spin sub-step is a symplectic colour-group sweep, so the energy error is second order and bounded.
* HTST handles zero (Goldstone/translational) modes, with translational zero-mode volumes computed from
  dE/dx; saddle points are polished by minimum-mode following before the harmonic analysis
  (`[gneb] htst = true`).
* Hessians are sparse (local terms) plus matrix-free (dipolar, higher order); eigenmodes and LLG
  frequencies use shift-invert Lanczos: the k = 0 mode of a 22 500-spin film comes out at the exact Larmor
  frequency in 10 s.

## Distributed memory (MPI)

Slab decomposition with one halo exchange per Heun stage (`magex.mpi_spin`), 64×64×32 = 131 072 spins,
one numba thread per rank:

| ranks | LLG (ns/spin/step) | Monte Carlo (ns/spin/sweep) |
|---|---|---|
| 1 | 53.8 | 71.9 |
| 2 | 30.4 (1.77×) | 39.1 (1.84×) |

The distributed T = 0 trajectory equals the serial one to 2×10⁻¹⁵, and Monte Carlo energies agree within
statistical errors.

## HTST without dense diagonalisation

Skyrmion collapse in films of growing size: log-determinants from sparse LU factors, the unstable and zero
modes by shift-invert Lanczos, the dynamical factor from the unstable mode only, saddle polished by
Newton steps on the sparse Hessian:

| spins | prefactor (Hz) | time |
|---|---|---|
| 400 | 3.87×10¹³ (dense: same to 10⁻⁸) | 0.1 s |
| 3 600 | 3.763×10¹³ | 0.6 s |
| 14 400 | 3.763×10¹³ | 5.5 s |
| 40 000 | 3.763×10¹³ | 37 s |

The prefactor is independent of the box size once the lattice-pinned saddle is counted once per lattice
translation (the minimum has two translational zero modes, the saddle none).

## CUDA

`magex.cuda_backend` (numba.cuda: fused Heun/Depondt kernels, Metropolis/heat-bath colour sweeps, per-thread
xoroshiro128+) and the cuFFT demag path of the CuPy micromagnetic backend reproduce the CPU kernels exactly
at T = 0 and the exact thermal averages, in numba's CUDA simulator.  No GPU was available, so there are no
GPU timings yet.

## Where the others still lead

* **GPUs**: VAMPIRE and Spirit have production CUDA back ends and mumax3 is GPU-native; magex's CUDA kernels
  are verified in the simulator but have not been run or timed on a physical GPU.
* **Very large MPI runs**: VAMPIRE also distributes the system set-up and has been used on thousands of
  cores; magex distributes the dynamics but builds the full system on every rank.
* **Maturity**: all three have years of production use and Spirit has an interactive GUI.
