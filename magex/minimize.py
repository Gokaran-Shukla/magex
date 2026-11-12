"""Energy minimisation of unit-spin configurations on the product of spheres (S^2)^N.

  lbfgs_oso   limited-memory BFGS in the tangent space with geodesic (rotation) retraction,
              in the spirit of Ivanov et al., Comput. Phys. Commun. 260, 107749 (2021) (the Spirit
              "LBFGS-OSO" minimiser); vectors are transported by tangent projection
  ncg         nonlinear conjugate gradients (Polak-Ribiere+) with the same retraction
  vp          velocity projection (quick-min, Spirit "VP")
  llg         damped LLG relaxation (alpha = 1, no precession) -- the most robust, slowest choice

All take a callable ``energy_grad(e) -> (E, dE/de)`` (or a SpinSystem) and return (e, info).
The convergence criterion is the maximum torque |e x dE/de| (energy units).
"""
from __future__ import annotations

import numpy as np


def _norm(e):
    return e / np.linalg.norm(e, axis=-1, keepdims=True)


def _proj(e, v):
    return v - np.einsum("ia,ia->i", v, e)[:, None] * e


def _retract(e, d):
    """Geodesic step e -> exp_e(d) for tangent d (per-spin rotation by |d|)."""
    th = np.linalg.norm(d, axis=1, keepdims=True)
    safe = np.where(th > 1e-14, th, 1.0)
    return _norm(np.where(th > 1e-14, e * np.cos(th) + d / safe * np.sin(th), e + d))


def _fg(obj):
    if callable(obj):
        return obj
    return lambda e: (obj.energy(e), obj.gradient(e))


def _line_search(fg, e, E, g, d, step, max_angle):
    dmax = float(np.max(np.linalg.norm(d, axis=1)))
    if dmax * step > max_angle:
        step = max_angle / dmax
    slope = float(np.sum(g * d))
    if slope >= 0:
        d = -g
        slope = -float(np.sum(g * g))
        dmax = float(np.max(np.linalg.norm(d, axis=1)))
        step = min(step, max_angle / max(dmax, 1e-300))
    for _ in range(30):
        en = _retract(e, step * d)
        En, gn = fg(en)
        if En <= E + 1e-4 * step * slope:
            return en, En, gn, step
        if abs(En - E) < 1e-11 * max(abs(E), 1.0):
            # energy differences below round-off: accept if the tangent gradient decreased
            if np.sum(_proj(en, gn) ** 2) < np.sum(g * g):
                return en, En, gn, step
        step *= 0.5
    return en, En, gn, step


def lbfgs_oso(system, e, tol=1e-6, max_iter=10000, memory=10, max_angle=0.3, log=None):
    fg = _fg(system)
    e = _norm(np.asarray(e, float).copy())
    E, G = fg(e)
    g = _proj(e, G)
    S, Y = [], []
    step = 1.0
    for it in range(max_iter):
        gmax = float(np.max(np.linalg.norm(g, axis=1)))
        if gmax < tol:
            break
        q = g.copy()
        al = []
        for s, y in zip(reversed(S), reversed(Y)):
            rho = 1.0 / float(np.sum(y * s))
            a = rho * float(np.sum(s * q))
            al.append((a, rho, s, y))
            q -= a * y
        if S:
            gam = float(np.sum(S[-1] * Y[-1]) / np.sum(Y[-1] * Y[-1]))
            q *= gam
        else:
            q *= 1e-2 / max(gmax, 1e-300)
        for a, rho, s, y in reversed(al):
            b = rho * float(np.sum(y * q))
            q += (a - b) * s
        d = _proj(e, -q)
        en, En, Gn, st = _line_search(fg, e, E, g, d, 1.0, max_angle)
        gn = _proj(en, Gn)
        s = _proj(en, st * d)
        y = gn - _proj(en, g)
        if float(np.sum(s * y)) > 1e-14:
            S.append(s)
            Y.append(y)
            if len(S) > memory:
                S.pop(0)
                Y.pop(0)
        else:
            S, Y = [], []
        S = [_proj(en, v) for v in S]
        Y = [_proj(en, v) for v in Y]
        e, E, g = en, En, gn
        if log and it % 200 == 0:
            log(f"  lbfgs {it:6d}  E = {E:.8f}  max torque = {gmax:.2e}")
    return e, dict(iterations=it, energy=float(E), max_torque=float(np.max(np.linalg.norm(g, axis=1))))


def ncg(system, e, tol=1e-6, max_iter=20000, max_angle=0.3, log=None):
    fg = _fg(system)
    e = _norm(np.asarray(e, float).copy())
    E, G = fg(e)
    g = _proj(e, G)
    d = -g
    step = 1e-2 / max(float(np.max(np.linalg.norm(g, axis=1))), 1e-300)
    for it in range(max_iter):
        gmax = float(np.max(np.linalg.norm(g, axis=1)))
        if gmax < tol:
            break
        en, En, Gn, step = _line_search(fg, e, E, g, d, min(step * 2.0, 1e3 * step + 1.0), max_angle)
        gn = _proj(en, Gn)
        go = _proj(en, g)
        beta = max(0.0, float(np.sum(gn * (gn - go)) / max(np.sum(g * g), 1e-300)))
        d = -gn + beta * _proj(en, d)
        if float(np.sum(d * gn)) >= 0 or (it + 1) % 50 == 0:      # descent check + periodic restart
            d = -gn
        e, E, g = en, En, gn
        if log and it % 500 == 0:
            log(f"  ncg {it:6d}  E = {E:.8f}  max torque = {gmax:.2e}")
    return e, dict(iterations=it, energy=float(E), max_torque=float(np.max(np.linalg.norm(g, axis=1))))


def vp(system, e, tol=1e-6, max_iter=50000, dt=None, log=None):
    """Velocity projection (quick-min) with unit mass."""
    fg = _fg(system)
    e = _norm(np.asarray(e, float).copy())
    E, G = fg(e)
    F = -_proj(e, G)
    dt = dt or 0.1 / max(float(np.max(np.linalg.norm(F, axis=1))), 1e-30) ** 0.5
    v = np.zeros_like(e)
    for it in range(max_iter):
        fmax = float(np.max(np.linalg.norm(F, axis=1)))
        if fmax < tol:
            break
        vF = float(np.sum(v * F))
        FF = float(np.sum(F * F))
        v = (vF / FF) * F if vF > 0 else np.zeros_like(v)
        v = v + dt * F
        d = dt * v
        dm = float(np.max(np.linalg.norm(d, axis=1)))
        if dm > 0.2:
            d *= 0.2 / dm
        e = _retract(e, _proj(e, d))
        E, G = fg(e)
        F = -_proj(e, G)
        v = _proj(e, v)
    return e, dict(iterations=it, energy=float(E), max_torque=float(np.max(np.linalg.norm(F, axis=1))))


def minimize(system, e, method="lbfgs", tol=1e-6, max_iter=None, log=None):
    """Dispatch: method in lbfgs | ncg | vp | llg.  ``system`` is a SpinSystem or energy_grad callable."""
    if method in ("lbfgs", "lbfgs_oso"):
        out = lbfgs_oso(system, e, tol, max_iter or 10000, log=log)
    elif method == "ncg":
        out = ncg(system, e, tol, max_iter or 20000, log=log)
    elif method == "vp":
        out = vp(system, e, tol, max_iter or 50000, log=log)
    elif method == "llg":
        from .llg import LLGSolver
        s = LLGSolver(system, 1.0, 2e-3, 0.0, precession=False)
        e = e.copy()
        for it in range(max_iter or 100000):
            if it % 50 == 0 and s.max_torque(e) * 5.7883818060e-2 * float(np.max(system.mu)) < tol:
                break
            e = s.step(e)
        out = (e, dict(iterations=it, energy=float(system.energy(e))))
    else:
        raise ValueError(f"unknown minimiser '{method}'")
    if getattr(system, "pinned", None) is not None and not callable(system):
        out[0][system.pinned] = system.pinned_dir[system.pinned]
    return out
