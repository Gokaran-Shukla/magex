# 14 · Torque efficiencies from first principles, and spin pumping

## SOT efficiency from spin and orbital Hall conductivities

Instead of a spin Hall angle typed by hand, the damping-like efficiency can be built from the Hall
conductivities that Wannier-based spin/orbital Hall codes compute:

```toml
[sot]
J = 1.0e12
sigma = [0, 1, 0]
thickness_nm = 1.0
theta = {sigma_xx = 2.0e4,      # S/cm (charge conductivity of the heavy metal)
         sigma_SH = -30,        # (hbar/e) S/cm, spin Hall conductivity
         sigma_OH = 5000,       # (hbar/e) S/cm, orbital Hall conductivity
         eta_LS = 0.1,          # orbital-to-spin conversion in the ferromagnet (orbital torque)
         transparency = 0.5,    # interface spin transparency
         nm_thickness_nm = 5, lambda_s_nm = 1.5}
# or: theta = {from = "hall_conductivities.json", eta_LS = 0.1}   (keys sigma_xx, sigma_SH, sigma_OH)
```

    θ_SH = 2σ_SH/σ_xx,  θ_OH = 2σ_OH/σ_xx,  θ_eff = T_int (θ_SH + η_LS θ_OH)(1 − sech(d/λ_s))

A Pt-like σ_SH = 2000 (ħ/e) S/cm with σ_xx = 5 × 10⁴ S/cm gives θ_SH = 0.08.  The orbital term
describes orbital torques from light metals (Ti, Cr, Cu/CuOx) on ferromagnets with large
orbital-to-spin conversion (Ni > Co > Fe); its value is printed at start-up.

## Spin pumping and the inverse spin Hall effect — `spin_pumping_fmr.toml`

```toml
[spin_pumping]
g_mix = 2.5e19        # 1/m^2  real part of the spin-mixing conductance
theta_SH = 0.08       # of the normal metal
rho = 2.0e-7          # Ohm m
lambda_s_nm = 1.5
nm_thickness_nm = 6
length_um = 1000
```

* the damping is enhanced by Δα = gμ_B g_r/(4πM_s t_FM) (0.0092 for 5 nm Py/Pt) — added to α;
* every `run` stage reports the pumped DC spin current j_s = (ħ/4π) g_r ⟨m × ṁ⟩ and, with the
  normal-metal parameters, the open-circuit ISHE voltage
  V = ρ θ_SH (2e/ħ) j_s (λ_s/d) tanh(d/2λ_s) L (no shunting by the ferromagnet).

Drive the FMR with `[field] ac = {B = [0, 5e-4, 0], f_GHz = 9.3}` at the Kittel frequency; scanning
`f_GHz` or the bias field gives the FMR line and the voltage spectrum.  The uniform-precession limit
of j_s is checked analytically in the tests.
