"""Temperature-dependent micromagnetics: Callen-Callen parameter scaling and the Landau-Lifshitz-Bloch
(LLB) equation with an optional two-temperature model (ultrafast demagnetisation).

Scaling with the equilibrium magnetisation m_e(T) (from classical mean field, or from a magex
Monte Carlo m(T) curve):   Ms(T) = Ms0 m_e,  A(T) = A0 m_e^a (a ~ 1.66-2),  K(T) = K0 m_e^3
(uniaxial single-ion, l(l+1)/2 = 3),  D(T) = D0 m_e^a.

LLB (Garanin 1997; Atxitia et al. PRB 75, 064420 (2007)) for the reduced magnetisation m (|m| free):

  dm/dt = -gamma m x B + gamma a_par (m.B) m / m^2 - gamma a_perp m x (m x B) / m^2
  a_par = lambda 2T/(3Tc),  a_perp = lambda (1 - T/(3Tc))  (T < Tc),   a_perp = a_par (T >= Tc)
  B = B_micromag[A(T)/m_e^2, K(T)/m_e^2, D(T)/m_e^2, Ms0] + B_long
  B_long = (1/(2 chi_par)) (1 - m^2/m_e^2) m                       (T < Tc)
         = -(1/chi_par) (1 + (3/5) Tc m^2/(T - Tc)) m              (T >= Tc)

with classical mean-field m_e = L(3 Tc m_e / T) and chi_par = (mu_s/k_B T) L'(x)/(1 - (3Tc/T) L'(x)),
x = 3 Tc m_e/T (per tesla).  Two-temperature model:
  C_e(Te) dTe/dt = -G (Te - Tl) + P(t),  C_l dTl/dt = G (Te - Tl),  C_e = gamma_e Te,
  P(t) = P0 exp(-(t - t0)^2 / (2 tau^2));  the spin system sees T = Te(t).
"""
from __future__ import annotations

import numpy as np

from .constants import GAMMA_LL, K_B, MU_B

# ----------------------------------------------------------------------------- mean field


def langevin(x):
    x = np.asarray(x, float)
    small = np.abs(x) < 1e-4
    return np.where(small, x / 3, 1 / np.tanh(np.where(small, 1, x)) - 1 / np.where(small, 1, x))


def dlangevin(x):
    x = np.asarray(x, float)
    small = np.abs(x) < 1e-3
    xs = np.where(small, 1, x)
    xs = np.clip(xs, -300, 300)
    return np.where(small, 1 / 3 - x**2 / 15, 1 / xs**2 - 1 / np.sinh(xs) ** 2)


def m_eq(T, Tc):
    """Classical mean-field equilibrium magnetisation m_e(T): root of m = L(3 Tc m / T)."""
    if T <= 0:
        return 1.0
    if T >= Tc:
        return 0.0
    from scipy.optimize import brentq
    return float(brentq(lambda m: m - float(langevin(3 * Tc * m / T)), 1e-12, 1.0, xtol=1e-15, rtol=1e-14))


def chi_par(T, Tc, mu_s):
    """Longitudinal susceptibility (1/T) in classical mean field; mu_s in mu_B."""
    beta = mu_s * MU_B / (K_B * max(T, 1e-9))
    if T < Tc:
        me = m_eq(T, Tc)
        x = 3 * Tc * me / T
        Lp = float(dlangevin(x))
        return beta * Lp / (1 - 3 * Tc / T * Lp)
    return beta / 3 / (1 - Tc / T) if T > Tc else 1e3


class MCScaling:
    """m_e(T) interpolated from a magex Monte Carlo mc_results.csv (largest size)."""

    def __init__(self, csv_path, Tc=None):
        import csv
        rows = list(csv.DictReader(open(csv_path)))
        N = max(int(r["N"]) for r in rows)
        rs = sorted([r for r in rows if int(r["N"]) == N], key=lambda r: float(r["T"]))
        self.T = np.array([float(r["T"]) for r in rs])
        self.m = np.array([float(r["m"]) for r in rs])
        self.Tc = Tc

    def __call__(self, T):
        if T <= self.T[0]:
            # Bloch-like extrapolation to m(0) = 1
            return float(1 - (1 - self.m[0]) * (T / self.T[0]) ** 1.5)
        return float(np.interp(T, self.T, self.m))


def scaled_parameters(params, T, Tc, me=None, a_exp=1.8, k_exp=3.0):
    """Callen-Callen scaled Ms, A, Ku1, D at temperature T (params: dict of T = 0 values)."""
    me = m_eq(T, Tc) if me is None else me
    out = dict(params)
    out["Ms"] = params.get("Ms", 0.0) * me
    for k in ("A", "D"):
        if k in params:
            out[k] = params[k] * me**a_exp
    for k in ("Ku1", "Kc1"):
        if k in params:
            out[k] = params[k] * me**k_exp
    out["m_e"] = me
    return out


# ----------------------------------------------------------------------------- LLB
class LLB:
    """Deterministic LLB integrator (RK4) on top of a Micromagnet whose parameters are the T = 0 values.

    The Micromagnet's A, Ku1, D are rescaled internally at every temperature change.
    """

    def __init__(self, system, Tc, lam=0.02, mu_s=2.0, a_exp=1.8, me_func=None):
        self.sys = system
        self.Tc = float(Tc)
        self.lam = float(lam)
        self.mu_s = float(mu_s)
        self.a_exp = a_exp
        self.me_func = me_func or (lambda T: m_eq(T, self.Tc))
        self._p0 = {k: getattr(system, k).copy() for k in ("A", "Ku1", "Kc1", "D_int", "D_bulk")}
        self.T = None
        self.set_temperature(0.0)

    def set_temperature(self, T):
        T = max(float(T), 1e-6)
        if self.T is not None and abs(T - self.T) < 1e-9:
            return
        self.T = T
        me = max(self.me_func(T), 1e-3)
        self.me = me
        s = self.sys
        s.A[:] = self._p0["A"] * me ** (self.a_exp - 2)          # A(T)/m_e^2
        s.D_int[:] = self._p0["D_int"] * me ** (self.a_exp - 2)
        s.D_bulk[:] = self._p0["D_bulk"] * me ** (self.a_exp - 2)
        s.Ku1[:] = self._p0["Ku1"] * me                          # K(T)/m_e^2 = K0 m_e
        s.Kc1[:] = self._p0["Kc1"] * me
        s.invalidate()
        self.chi = chi_par(T, self.Tc, self.mu_s)
        if T < self.Tc:
            self.a_par = self.lam * 2 * T / (3 * self.Tc)
            self.a_perp = self.lam * (1 - T / (3 * self.Tc))
        else:
            self.a_par = self.a_perp = self.lam * 2 * T / (3 * self.Tc)

    def field(self, m, t=None):
        B = self.sys.effective_field(m, t)
        m2 = np.einsum("...a,...a->...", m, m)[..., None]
        if self.T < self.Tc:
            Bl = (1 / (2 * self.chi)) * (1 - m2 / self.me**2) * m
        else:
            Bl = -(1 / self.chi) * (1 + 0.6 * self.Tc * m2 / (self.T - self.Tc)) * m
        return B + np.where(self.sys.magnetic[..., None], Bl, 0.0)

    def rhs(self, m, t=None):
        B = self.field(m, t)
        m2 = np.maximum(np.einsum("...a,...a->...", m, m), 1e-12)[..., None]
        mxB = np.cross(m, B)
        out = -GAMMA_LL * mxB + GAMMA_LL * self.a_par * (np.einsum("...a,...a->...", m, B)[..., None]) * m / m2 \
            - GAMMA_LL * self.a_perp * np.cross(m, mxB) / m2
        return np.where(self.sys.magnetic[..., None], out, 0.0)

    def step(self, m, dt):
        t = self.sys.t
        k1 = self.rhs(m, t)
        k2 = self.rhs(m + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = self.rhs(m + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = self.rhs(m + dt * k3, t + dt)
        self.sys.t = t + dt
        return m + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def run(self, m, duration, dt, T_of_t=None, record_every=None, callback=None):
        n = int(round(duration / dt))
        rec = max(1, int(round((record_every or duration / 200) / dt)))
        rows = []
        for k in range(n + 1):
            if T_of_t is not None:
                self.set_temperature(T_of_t(self.sys.t))
            if k % rec == 0:
                mm = m[self.sys.magnetic]
                rows.append([self.sys.t, self.T, *mm.mean(0), float(np.linalg.norm(mm, axis=1).mean()), self.me])
                if callback:
                    callback(self.sys.t, m)
            if k < n:
                m = self.step(m, dt)
        return m, np.array(rows)


# ----------------------------------------------------------------------------- two-temperature model
class TwoTemperature:
    """Electron/lattice temperatures under a Gaussian laser pulse (SI units)."""

    def __init__(self, T0=300.0, gamma_e=6e3, C_l=3.8e6, G=8e17, P0=1e21, t0=0.2e-12, tau=0.05e-12):
        self.gamma_e, self.C_l, self.G = gamma_e, C_l, G
        self.P0, self.t0, self.tau = P0, t0, tau
        self.Te = self.Tl = float(T0)
        self.t = 0.0

    def P(self, t):
        return self.P0 * np.exp(-((t - self.t0) ** 2) / (2 * self.tau**2))

    def advance(self, t):
        """Integrate to time t with small explicit sub-steps."""
        while self.t < t - 1e-21:
            h = min(1e-16, t - self.t)
            dTe = (-self.G * (self.Te - self.Tl) + self.P(self.t)) / (self.gamma_e * self.Te)
            dTl = self.G * (self.Te - self.Tl) / self.C_l
            self.Te += h * dTe
            self.Tl += h * dTl
            self.t += h
        return self.Te
