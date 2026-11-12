"""Spin-transfer torques and Thiele analysis.

Zhang-Li (current in plane, continuous textures), Gilbert form

    dm/dt|_STT = -(u.grad) m + beta m x (u.grad) m,     u = P g mu_B J / (2 e Ms (1 + beta^2))

(converted to the Landau-Lifshitz form used by the integrators).  Domain walls below Walker
breakdown move at v = (beta/alpha) u; skyrmions follow the Thiele equation.

Slonczewski (current perpendicular, spin-valve / MTJ free layer with polariser p):

    B_STT = b_J [ eps(m) (m x p) + eps' p ],   b_J = hbar J / (2 e Ms t),
    eps(m) = P Lambda^2 / ( (Lambda^2 + 1) + (Lambda^2 - 1) m.p )

(the damping-like part has the same form as the SOT field).  Positive J pulls m towards p.

Thiele:  G x (v - u) + D (alpha v - beta u) = 0 with gyrovector G = 4 pi Q z and dissipation
tensor D_ij = int d_i m . d_j m dA / (per unit thickness, cell-area weighted); returns the
predicted texture velocity and Hall angle for a given u.
"""
from __future__ import annotations

import numpy as np

from .constants import E_CHARGE, HBAR, MU_B


def gradients(system, m):
    """[d_x m, d_y m, d_z m] (1/m) with the solver's boundary conditions."""
    out = []
    for axis, (p, Ap, q, Aq) in enumerate(system._stencil(m)):
        if Ap is None:
            out.append(np.zeros_like(m))
        else:
            out.append((p - q) / (2 * system.mesh.d[axis]))
    return out


def zhang_li_u(system, J, P=0.5, beta=0.0):
    """Drift velocity u (m/s) per cell for current density J (A/m^2, 3-vector)."""
    Ms = system.Ms
    inv = np.divide(1.0, Ms, out=np.zeros_like(Ms), where=Ms > 0)
    pre = P * 2.0 * MU_B / (2 * E_CHARGE * (1 + beta**2))        # g = 2
    return pre * inv[..., None] * np.asarray(J, float)


def stt_rhs(system, m, t=None):
    """Extra dm/dt (1/s) in Landau-Lifshitz form from system.stt (Zhang-Li)."""
    s = system.stt
    if not s or s.get("type", "zhang-li") != "zhang-li":
        return 0.0
    if "pulse_ns" in s:
        t0, w = s["pulse_ns"]
        tt = system.t if t is None else t
        if not (t0 * 1e-9 <= tt < (t0 + w) * 1e-9):
            return 0.0
    beta = float(s.get("beta", 0.0))
    u = zhang_li_u(system, s["J"], s.get("P", 0.5), beta)
    g = gradients(system, m)
    ugm = u[..., 0:1] * g[0] + u[..., 1:2] * g[1] + u[..., 2:3] * g[2]
    tau = -ugm + beta * np.cross(m, ugm)
    a = system.alpha[..., None]
    out = (tau + a * np.cross(m, tau)) / (1 + a * a)
    return np.where(system.magnetic[..., None], out, 0.0)


def slonczewski_field(system, m, t=None):
    s = system.stt
    if not s or s.get("type") != "slonczewski":
        return 0.0
    if "pulse_ns" in s:
        t0, w = s["pulse_ns"]
        tt = system.t if t is None else t
        if not (t0 * 1e-9 <= tt < (t0 + w) * 1e-9):
            return 0.0
    p = np.asarray(s.get("p", [0, 0, 1]), float)
    p = p / np.linalg.norm(p)
    Ms = system.Ms
    inv = np.divide(1.0, Ms, out=np.zeros_like(Ms), where=Ms > 0)
    tfl = s.get("thickness", system.mesh.n[2] * system.mesh.d[2])
    bJ = HBAR * s["J"] / (2 * E_CHARGE * tfl) * inv
    L2 = float(s.get("Lambda", 1.0)) ** 2
    mp = np.einsum("...a,a->...", m, p)
    eps = s.get("P", 0.5) * L2 / ((L2 + 1) + (L2 - 1) * mp)
    B = (bJ * eps)[..., None] * np.cross(m, p) + (bJ * s.get("eps_prime", 0.0))[..., None] * p
    return np.where(system.magnetic[..., None], B, 0.0)


# ----------------------------------------------------------------------------- Thiele
def texture_center(system, m, layer=0):
    """Centre (m) of the reversed core (weight (1 - m.m_bg)/2), background = majority direction."""
    X, Y, Z = system.mesh.coords()
    mm = m[:, :, layer]
    bg = np.sign(np.sum(mm[..., 2])) or 1.0
    w = np.clip(0.5 * (1 - bg * mm[..., 2]), 0, 1) ** 4
    w = np.where(system.magnetic[:, :, layer], w, 0)
    return np.array([np.sum(w * X[:, :, layer]), np.sum(w * Y[:, :, layer])]) / max(np.sum(w), 1e-300)


def thiele(system, m, u=(1.0, 0.0), alpha=None, beta=0.0, layer=0):
    """Gyrovector, dissipation tensor and the Thiele velocity for Zhang-Li drive u (m/s, in plane)."""
    from .plot import topological_charge
    g = gradients(system, m)
    dA = system.mesh.d[0] * system.mesh.d[1]
    mag = system.magnetic[:, :, layer]
    gx, gy = g[0][:, :, layer], g[1][:, :, layer]
    D = np.array([[np.sum((gx * gx).sum(-1)[mag]), np.sum((gx * gy).sum(-1)[mag])],
                  [np.sum((gy * gx).sum(-1)[mag]), np.sum((gy * gy).sum(-1)[mag])]]) * dA
    Q = topological_charge(system, m, layer)
    Gz = 4 * np.pi * Q              # sign fixed against direct LLG simulations (tests)
    a = float(np.mean(system.alpha[system.magnetic])) if alpha is None else float(alpha)
    u = np.asarray(u, float)
    # G x (v - u) with G = Gz z:  Gz (-(v-u)_y, (v-u)_x)
    Mg = np.array([[0.0, -Gz], [Gz, 0.0]])
    v = np.linalg.solve(Mg + a * D, Mg @ u + beta * D @ u)
    hall = float(np.degrees(np.arctan2(v[1], v[0]) - np.arctan2(u[1], u[0])))
    return dict(Q=float(Q), G=Gz, D=D.tolist(), v=v.tolist(), hall_angle_deg=hall)
