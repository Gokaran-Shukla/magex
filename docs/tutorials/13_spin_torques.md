# 13 · Spin-transfer and spin-orbit torques

| torque | input | physics |
|--------|-------|---------|
| Zhang–Li STT | `[stt] type = "zhang-li"`, `J = [Jx, Jy, Jz]`, `P`, `beta` | current in plane through textures: −(u·∇)m + β m×(u·∇)m, u = Pgμ_BJ/(2eM_s(1+β²)) |
| Slonczewski STT | `[stt] type = "slonczewski"`, `J`, `P`, `Lambda`, `p`, `eps_prime`, `thickness_nm` | current perpendicular (spin valves, MTJ free layers), angle-dependent efficiency ε(m·p) |
| SOT | `[sot]` `J`, `theta`, `sigma`, `fl_ratio`, `thickness_nm`, `pulse_ns` | damping-like ħθJ/(2eM_st) and field-like torques from a heavy metal |

All accept `pulse_ns = [t0, width]` and work with the two-sublattice solver.

## Checks (`tests/test_micromag.py`)

* Zhang–Li domain wall with β = α moves at v = u (3 %); for β ≠ α in a 1D wire without hard-axis
  anisotropy the wall precesses and v = u(1 + αβ)/(1 + α²), as expected;
* Slonczewski macrospin with perpendicular anisotropy switches above and not below the critical
  current J_c = 2eαM_s t B_K /(ħ ε);
* skyrmion velocity and skyrmion Hall angle from LLG agree with the Thiele equation.

## Skyrmion racetrack — `skyrmion_racetrack_stt.toml`

```toml
[stt]
type = "zhang-li"
J = [5e11, 0, 0]
P = 0.5
beta = 0.15

[[stage]]
type = "thiele"          # prediction from the relaxed texture: G x (v - u) + D (alpha v - beta u) = 0

[[stage]]
type = "run"
time_ns = 0.3
track = true             # follows the texture centre; prints velocity and Hall angle
```

| | v_x (m/s) | v_y (m/s) | Hall angle |
|--|-----------|-----------|------------|
| Thiele (from the relaxed skyrmion) | 22.96 | −3.94 | −9.74° |
| LLG simulation | 22.71 | −3.97 | −9.91° |

For β = α the skyrmion follows the current exactly (no Hall effect); β ≠ α produces the transverse
motion that pushes skyrmions to racetrack edges.  The Thiele stage also reports Q, the gyrovector and
the dissipation tensor, so it can be used to scan materials quickly before running dynamics.

## Spin-orbit torque from first principles

See [tutorial 14](14_first_principles_sot.md): the damping-like efficiency can be computed from spin-
and orbital-Hall conductivities instead of entering θ by hand.
