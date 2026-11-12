"""Time integrators and energy minimisers for the micromagnetic system.

LLG (Landau-Lifshitz form, gamma = GAMMA_LL):
    dm/dt = -gamma/(1+alpha^2) [ m x B + alpha m x (m x B) ],   B = B_eff + B_SOT (+ b_th)

* RK45      adaptive Dormand-Prince 5(4) with FSAL, error = max |m5 - m4|
* Heun      fixed-step stochastic Heun with the Brown thermal field
            <b b> = 2 alpha k_B T / (gamma Ms V dt)
* minimize  Barzilai-Borwein steepest descent on the sphere (as mumax 'minimize')
* relax     damping-only RK45 (precession off) until the torque stops decreasing, then minimize
* hysteresis  field sweep with minimisation at every step
"""
from __future__ import annotations

import time

import numpy as np

from .constants import GAMMA_LL, K_B


def llg_rhs(system, m, t, precession=True, thermal=None):
    B = system.effective_field(m, t)
    if system.sot:
        B = B + system.sot_field(m)
    stt = getattr(system, "stt", None)
    if stt and stt.get("type") == "slonczewski":
        from .stt import slonczewski_field
        B = B + slonczewski_field(system, m, t)
    if thermal is not None:
        B = B + thermal
    a = system.alpha[..., None]
    mxB = np.cross(m, B)
    rhs = a * np.cross(m, mxB)
    if precession:
        rhs = rhs + mxB
    out = -GAMMA_LL / (1 + a * a) * rhs
    if stt and stt.get("type", "zhang-li") == "zhang-li":
        from .stt import stt_rhs
        out = out + stt_rhs(system, m, t)
    return out, B


class _Recorder:
    def __init__(self, system, every, callback):
        self.sys = system
        self.every = every
        self.next = 0.0
        self.rows = []
        self.cb = callback

    def __call__(self, t, m, B, force=False):
        if force or t >= self.next - 1e-18:
            E = self.sys.energies(m)
            tq = float(np.max(np.linalg.norm(np.cross(m, B), axis=-1)))
            mm = self.sys.average(m)
            Bx = self.sys.B_ext + (self.sys.B_func(t) if self.sys.B_func else 0)
            self.rows.append([t, *mm, E["total"], E.get("exchange", 0), E.get("dmi", 0), E.get("anisotropy", 0)
                              + E.get("cubic", 0), E.get("demag", 0), E.get("zeeman", 0), tq, *np.ravel(Bx)])
            if self.cb:
                self.cb(t, m)
            while self.next <= t + 1e-18:
                self.next += self.every


TABLE_HEADER = ["t_s", "mx", "my", "mz", "E_total_J", "E_exchange_J", "E_dmi_J", "E_anis_J", "E_demag_J",
                "E_zeeman_J", "max_torque_T", "Bx_T", "By_T", "Bz_T"]

# Dormand-Prince coefficients
_C = [0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1]
_A = [[], [1 / 5], [3 / 40, 9 / 40], [44 / 45, -56 / 15, 32 / 9],
      [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
      [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
      [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84]]
_B5 = [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0]
_B4 = [5179 / 57600, 0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40]


class RK45:
    def __init__(self, system, tol=1e-5, dt=1e-13, dt_min=1e-16, dt_max=5e-12, precession=True):
        self.sys = system
        self.tol = tol
        self.dt = dt
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.precession = precession
        self.nsteps = 0
        self.nreject = 0

    def run(self, m, duration, record_every=None, callback=None, stop_torque=None, log=None, max_steps=10**7):
        """Integrate for ``duration`` seconds.  Returns (m, table)."""
        sysm = self.sys
        t0, tend = sysm.t, sysm.t + duration
        rec = _Recorder(sysm, record_every or duration / 100, callback)
        k1, B = llg_rhs(sysm, m, sysm.t, self.precession)
        rec(sysm.t, m, B, force=True)
        wall = time.time()
        lastlog = wall
        while sysm.t < tend - 1e-21 and self.nsteps < max_steps:
            dt = min(self.dt, tend - sysm.t)
            ks = [k1]
            for s in range(1, 7):
                ms = m + dt * sum(a * k for a, k in zip(_A[s], ks))
                ms = sysm.normalize(ms)
                k, Bs = llg_rhs(sysm, ms, sysm.t + _C[s] * dt, self.precession)
                ks.append(k)
            m5 = sysm.normalize(m + dt * sum(b * k for b, k in zip(_B5, ks)))
            m4 = m + dt * sum(b * k for b, k in zip(_B4, ks))
            err = float(np.max(np.linalg.norm(m5 - sysm.normalize(m4), axis=-1)))
            if err <= self.tol or dt <= self.dt_min:
                sysm.t += dt
                m = m5
                k1, B = ks[6], Bs                    # FSAL: stage 7 is evaluated at (t+dt, m5)
                self.nsteps += 1
                rec(sysm.t, m, B)
                if stop_torque is not None:
                    tq = float(np.max(np.linalg.norm(np.cross(m, B), axis=-1)))
                    if tq < stop_torque:
                        break
            else:
                self.nreject += 1
            fac = 0.9 * (self.tol / max(err, 1e-30)) ** 0.2
            self.dt = float(np.clip(dt * np.clip(fac, 0.2, 5.0), self.dt_min, self.dt_max))
            if log and time.time() - lastlog > 15:
                lastlog = time.time()
                mm = sysm.average(m)
                log(f"    t = {sysm.t * 1e9:9.4f} ns  dt = {self.dt * 1e15:7.2f} fs  "
                    f"<m> = ({mm[0]: .4f},{mm[1]: .4f},{mm[2]: .4f})  steps {self.nsteps}")
        rec(sysm.t, m, B, force=True)
        if log:
            log(f"    RK45: {self.nsteps} steps ({self.nreject} rejected) in {time.time() - wall:.1f} s, "
                f"t = {sysm.t * 1e9:.4f} ns")
        return m, np.array(rec.rows)


class Heun:
    """Fixed-step stochastic Heun integrator (finite temperature)."""

    def __init__(self, system, dt=1e-14, temperature=0.0, seed=None, precession=True):
        self.sys = system
        self.dt = dt
        self.T = temperature
        self.rng = np.random.default_rng(seed)
        self.precession = precession

    def thermal(self):
        if self.T <= 0:
            return None
        s = self.sys
        Ms = np.where(s.magnetic, s.Ms, 1.0)
        sig = np.sqrt(2 * s.alpha * K_B * self.T / (GAMMA_LL * Ms * s.mesh.V * self.dt))
        b = sig[..., None] * self.rng.normal(size=s.mesh.n + (3,))
        return np.where(s.magnetic[..., None], b, 0.0)

    def run(self, m, duration, record_every=None, callback=None, log=None):
        s = self.sys
        n = int(round(duration / self.dt))
        rec = _Recorder(s, record_every or duration / 100, callback)
        wall = time.time()
        for k in range(n):
            b = self.thermal()
            k1, B = llg_rhs(s, m, s.t, self.precession, b)
            if k == 0:
                rec(s.t, m, B, force=True)
            mp = s.normalize(m + self.dt * k1)
            k2, _ = llg_rhs(s, mp, s.t + self.dt, self.precession, b)
            m = s.normalize(m + 0.5 * self.dt * (k1 + k2))
            s.t += self.dt
            rec(s.t, m, B)
        _, B = llg_rhs(s, m, s.t, self.precession)
        rec(s.t, m, B, force=True)
        if log:
            log(f"    Heun: {n} steps at T = {self.T} K in {time.time() - wall:.1f} s")
        return m, np.array(rec.rows)


def minimize(system, m, tol=1e-5, max_iter=50000, log=None):
    """Barzilai-Borwein descent.  Stops when max |m x B| < tol (T).  Returns (m, iterations, torque)."""
    # descent direction g = B_perp = -m x (m x B)
    B = system.effective_field(m)
    g = -np.cross(m, np.cross(m, B))
    tq = float(np.max(np.linalg.norm(g, axis=-1)))
    tau = 1e-2 / max(tq, 1e-12)
    wall = time.time()
    it = 0
    for it in range(1, max_iter + 1):
        if tq < tol:
            break
        mn = system.normalize(m + tau * g)
        Bn = system.effective_field(mn)
        gn = -np.cross(mn, np.cross(mn, Bn))
        s = (mn - m).ravel()
        y = -(gn - g).ravel()
        sy = float(s @ y)
        if sy > 0:
            tau = float(s @ s) / sy if it % 2 else sy / float(y @ y)
        else:
            tau = 1e-2 / max(tq, 1e-12)
        tau = min(tau, 1e3 / max(tq, 1e-12))
        m, g = mn, gn
        tq = float(np.max(np.linalg.norm(g, axis=-1)))
    if log:
        log(f"    minimize: {it} iterations, max torque {tq:.2e} T, {time.time() - wall:.1f} s")
    return m, it, tq


def relax(system, m, tol=1e-5, max_time=0.05e-9, chunk=10e-12, log=None):
    """Damping-only dynamics (robust descent away from saddle points) followed by BB minimisation.

    The damping-only stage runs in chunks of ``chunk`` seconds until the energy decrease per
    chunk drops below 1e-5 of its total change or ``max_time`` is reached.
    """
    a0 = system.alpha.copy()
    system.alpha[system.magnetic] = 1.0
    t0 = system.t
    solver = RK45(system, tol=1e-4, precession=False, dt_max=1e-11)
    E_start = E_prev = system.energies(m)["total"]
    tq = np.inf
    while system.t - t0 < max_time:
        m, tab = solver.run(m, chunk, record_every=chunk)
        E = tab[-1, 4]
        tq = tab[-1, 10]
        if tq < max(100 * tol, 1e-3) or abs(E_prev - E) < 1e-5 * max(abs(E_start - E), 1e-30):
            break
        E_prev = E
    if log:
        log(f"    relax (damping only): {(system.t - t0) * 1e12:.0f} ps, {solver.nsteps} steps, torque {tq:.2e} T")
    system.alpha[:] = a0
    system.t = t0
    m, it, tq = minimize(system, m, tol, log=log)
    return m


def hysteresis(system, m, Bmax, npts=41, direction=(0, 0, 1), branch="full", tol=1e-4, callback=None, log=None):
    """Quasi-static loop B = b u, b: +Bmax -> -Bmax (-> +Bmax).  Returns table [b, m.u, mx, my, mz]."""
    u = np.asarray(direction, float)
    u = u / np.linalg.norm(u)
    down = np.linspace(Bmax, -Bmax, npts)
    bs = np.concatenate([down, down[::-1][1:]]) if branch == "full" else down
    rows = []
    B0 = system.B_ext.copy()
    for k, b in enumerate(bs):
        system.set_field(B0 + b * u)
        m, it, tq = minimize(system, m, tol)
        mm = system.average(m)
        rows.append([b, mm @ u, *mm])
        if callback:
            callback(k, b, m)
        if log and (k % max(1, len(bs) // 10) == 0):
            log(f"    B = {b: .4f} T   m.u = {mm @ u: .4f}   ({it} it, torque {tq:.1e} T)")
    system.set_field(B0)
    return m, np.array(rows)
