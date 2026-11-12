# 9 · Quantum corrections: RPA T_C and the quantum thermostat

Classical spins obey equipartition: every magnon mode holds k_BT regardless of its energy.  This
underestimates the order at low temperature (no T^{3/2} Bloch law) and, for small spins, the ordering
temperature.  magex offers three corrections.

## 1. Fast estimates without Monte Carlo — `magex tc`

```bash
magex tc validation_runs/CrI3_atomic_soc/results.json.gz
magex tc validation_runs/bccFe_weighted/results.json.gz --rcut 11
```

| material | T_MF | T_RPA | T_RPA (S+1)/S | MC (classical) | experiment |
|----------|------|-------|----------------|----------------|------------|
| bcc Fe | 1442 K | 952 K | – (itinerant) | 1060 K | 1043 K |
| SrMnO3 | 327 K | 169 K | 289 K | 192 K | 233–260 K |
| CrI3 monolayer | 30 K | 25 K | 41 K | 20 K | 45 K |

RPA (Tyablikov; Pajda et al. PRB 64, 174402) uses the sign-gauged J(q) of all sublattices and the
uniaxial anisotropy gap (single-ion + two-site), so 2D magnets give a finite T_C.  For bcc Fe it
reproduces the 950 K of Pajda et al.  The (S+1)/S rescaling, with S = |μ|/2, is the quantum
correction for **localized** moments (CrI3, oxides); do not use it for itinerant metals.

`magex mc` also prints these numbers and stores them in `tc_estimate.json` (`T_RPA`,
`T_quantum_rescaled`, `T_RPA_quantum`).

## 2. Quantum thermostat in the LLG

```toml
[llg]
temperature = 20.0
thermostat = "quantum"      # default "classical"
```

The white thermal noise is replaced by coloured noise with the Planck spectrum
S(ω) ∝ x/(eˣ − 1), x = ħω/k_BT (Barker & Bauer, PRL 122, 186601 (2019)), so that magnons of energy
ħω ≫ k_BT are frozen out.  The spectrum is represented by 23 damped-oscillator and Lorentzian
Ornstein–Uhlenbeck processes (max. error 2 × 10⁻³), each integrated exactly over the time step.

Check (free spins, μ = 2 μ_B, B = 1 T, T = 1 K, ħω_L/k_BT = 1.34):

| thermostat | ⟨m_z⟩ from LLG | prediction |
|------------|----------------|------------|
| classical | 0.409 ± 0.002 | L(μB/k_BT) = 0.402 |
| quantum | 0.683 ± 0.003 | L(μB/k_BT_eff) = 0.654, T_eff = T·x/(eˣ−1) |

The remaining 4 % comes from the damping-broadened response of the spin to frequencies around ω_L.
Use it for M(T) curves of insulating magnets, magnon-driven effects at low temperature, and
anything where classical statistics overestimates the thermal disorder.
