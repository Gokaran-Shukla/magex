# Conventions and units

## Atomistic spin Hamiltonian (magex / TB2J convention)

    E = Σ_i e_i·A_i·e_i  −  Σ_{i≠j} e_i·J_ij·e_j  −  μ_B Σ_i μ_i e_i·B

* the exchange sum runs over **ordered** pairs, so a Heisenberg bond `−J' Σ_<ij> e_i·e_j` corresponds to `J = J'/2`;
  `J > 0` is ferromagnetic;
* `J_ij` is a full 3×3 tensor: isotropic part, symmetric anisotropic part and DMI vector
  `D = (J^A_yz, J^A_zx, J^A_xy)`, so `e_i·J^A·e_j = D·(e_i × e_j)`;
* single-ion anisotropy `e·A·e` (easy axis = smallest eigenvalue);
* units: meV, tesla, ps, kelvin, Å.  μ_B = 5.7883818060×10⁻² meV/T, k_B = 8.617333262×10⁻² meV/K,
  γ = 0.1760859630 rad ps⁻¹ T⁻¹.

Effective field  `B_i = −(1/(μ_B μ_i)) ∂E/∂e_i`  (tesla).

## Atomistic LLG

    de_i/dt = −γ/(1+α²) [ e_i × B_i + α e_i × (e_i × B_i) ],   ⟨b b⟩ = 2αk_BT/(γ μ_i μ_B Δt)

Stochastic Heun integrator (Stratonovich), renormalised after each stage (default; also Depondt, SIB, RK4, Euler).

Extra atomistic terms (`[structure]`): cubic `Kc Σ a_i²a_j²`, uniaxial `−K2 (e·u)⁴ − K3 (e·u)⁶`, Néel surface
`(k_s/2) Σ_nn (e·r̂)²`, quadruplets `−K[(e_i·e_j)(e_k·e_l) + (e_j·e_k)(e_l·e_i) − (e_i·e_k)(e_j·e_l)]`, and dipolar
`−(1/2) Σ_i μ_i μ_B e_i·B_dip,i` with all pairs counted once.

## Monte Carlo

Metropolis with adaptive Gaussian trial moves (acceptance ≈ 50 %), plus energy-conserving
over-relaxation; sites that do not interact are updated simultaneously (greedy graph colouring), which
makes the sweep a handful of vectorised operations.  Order parameter: magnetisation (FM) or staggered
vector built from the magex reference signs (AFM, ferrimagnet).  T_C / T_N from the Binder-cumulant
crossing of two sizes, otherwise the susceptibility peak.

## Micromagnetics (SI units)

    w = A|∇m|² + D_int [m_z ∇·m − (m·∇) m_z] + D_bulk m·(∇×m) − K_u1 (m·u)² − K_u2 (m·u)⁴
        + K_c1 (α₁²α₂² + α₂²α₃² + α₃²α₁²) − M_s m·B − (μ₀/2) M_s m·H_d

* B_eff = −(1/M_s) ∂w/∂m (T); γ = 1.7595×10¹¹ rad s⁻¹ T⁻¹ (same as mumax/OOMMF);
* free surfaces: `2A ∂_n m = D_int[(m·n) ẑ − m_z n] + D_bulk (n × m)` (ghost cells);
* interface exchange: harmonic mean of A, optional scaling per region pair (`[[interlayer]]`);
* demag: Newell cell-averaged tensor, FFT convolution; zero padding on open axes, macro-geometry
  image sums on periodic axes;
* SOT: `B_SOT = B_DL (m × σ) + B_FL σ`, `B_DL = ħ θ_SH J /(2 e M_s t)`, giving the damping-like torque
  `−γ B_DL m × (m × σ)`, which pushes m towards σ for B_DL > 0.

With these signs a positive interfacial D favours the Néel wall whose centre points along −x when
m_z goes from +1 to −1 along +x (checked in `tests/test_micromag.py::test_dmi_wall_energy`).  The atomistic → continuum bridge uses
the same signs, so chirality is preserved across scales.
