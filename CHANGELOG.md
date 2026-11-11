# Changelog

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
