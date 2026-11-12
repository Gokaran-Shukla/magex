---
title: 'magex: from Wannier Hamiltonians to magnetic textures, ordering temperatures and micromagnetics'
tags:
  - Python
  - magnetism
  - exchange interactions
  - spin dynamics
  - Monte Carlo
  - micromagnetics
  - skyrmions
  - spintronics
authors:
  - name: Gokaran Shukla
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
  - name: Physical Science and Engineering Division, King Abdullah University of Science and Technology (KAUST), Thuwal, Saudi Arabia
    index: 1
date: 12 November 2026
bibliography: paper.bib
---

# Summary

`magex` is a Python package that connects first-principles electronic structure to device-scale
magnetisation dynamics in one consistent framework. From a Wannier90 tight-binding Hamiltonian it
computes the full exchange tensors, Dzyaloshinskii–Moriya interactions, single-ion anisotropy and the
Gilbert damping with Green's-function methods; the same Hamiltonian then drives atomistic
Landau–Lifshitz–Gilbert (LLG) spin dynamics, Monte Carlo simulations of ordering temperatures, energy
barriers by the geodesic nudged elastic band method, and a finite-difference micromagnetic solver for
thin films and nanostructures, including antiferromagnets, altermagnets, spin-transfer and spin–orbit
torques and the Landau–Lifshitz–Bloch equation. Every step reads the output of the previous one, uses
the same sign conventions, and is validated against analytic limits in an automated test-suite.

# Statement of need

Computational magnetism is usually done with a chain of separate codes: a DFT code, a Wannier
interpolation, a code for exchange parameters such as TB2J [@He2021], an atomistic spin-dynamics
package such as VAMPIRE [@Evans2014], UppASD [@Skubic2008] or Spirit [@Muller2019], and a
micromagnetic solver such as OOMMF [@Donahue1999] or mumax³ [@Vansteenkiste2014]. Each step has its
own conventions for the sign and double counting of the exchange, the DMI vector and the anisotropy,
and converting between them is a frequent source of error. Some parameters — the Gilbert damping in
particular — are normally typed in by hand rather than computed.

`magex` removes these interfaces. Its exchange engine uses a rotationally invariant partition of the
exchange field that restores the Goldstone sum rule broken by the on-site rotation of the standard
Liechtenstein formula [@Liechtenstein1987], obtains the full tensor and DMI from a single
spin-polarised calculation, and computes the Gilbert damping from the torque–torque correlation
function [@Kambersky2007; @Gilmore2007]. The resulting `results.json` is read directly by all
simulation engines, and a gradient expansion gives the micromagnetic exchange stiffness, DMI and
anisotropy, so a single first-principles calculation parametrises atomistic and continuum
simulations alike. TB2J results can be imported, and models can be exported to VAMPIRE, UppASD,
Spirit and mumax³.

# Features and validation

* **Exchange and damping:** full 3×3 exchange tensors, DMI and single-ion anisotropy from one
  Hamiltonian; Bruno renormalisation [@Bruno2003]; ligand downfolding; finite-angle (Lloyd) pair
  energies with biquadratic fits; linear spin-wave theory; Gilbert damping tensors resolved into
  intra- and interband contributions.
* **Atomistic engines:** stochastic LLG with classical or quantum (Planck-spectrum) thermostats
  [@Barker2019]; dynamical structure factors $S(\mathbf q,\omega)$; vectorised Metropolis Monte
  Carlo with graph colouring, parallel tempering and Wang–Landau sampling; RPA ordering temperatures
  [@Pajda2001]; biquadratic and four-spin interactions; geodesic nudged elastic band [@Bessarab2015].
* **Micromagnetics:** Newell demagnetising tensor [@Newell1993] with FFT convolution and periodic
  images, interfacial and bulk DMI with chiral boundary conditions, Zhang–Li and Slonczewski
  spin-transfer torques, spin–orbit torques with efficiencies from spin- and orbital-Hall
  conductivities, spin pumping, two-sublattice antiferro-, ferri- and altermagnets, Oersted fields,
  RKKY coupling, grains, the Landau–Lifshitz–Bloch equation with a two-temperature model,
  hysteresis, spin-wave dispersions, OVF 2.0 output, checkpointing and an experimental GPU backend.

The test-suite checks, among others: the Larmor frequency and energy conservation; Langevin
statistics of the classical and quantum thermostats; the simple-cubic Heisenberg $T_C$; demagnetising
sum rules; Bloch and DMI domain-wall energies (0.06 % and 0.03 %); μMAG standard problem 4; the
antiferromagnetic resonance frequency (0.04 %); the atomistic-to-continuum exchange stiffness and
DMI (0.05 %); the Stoner–Wohlfarth barrier; Zhang–Li wall velocities and the Thiele skyrmion Hall
angle; Oersted fields and RKKY coupling. With the bundled first-principles exchange, classical Monte
Carlo gives $T_C = 1060$ K for bcc Fe (experiment 1043 K) and the RPA reproduces the 950 K of
@Pajda2001.

# Acknowledgements

The author thanks the developers of TB2J for the public example data used in the validation.

# References
