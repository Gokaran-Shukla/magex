"""Harmonic analysis on the spin manifold: Hessian, eigenmodes, harmonic transition-state theory and
minimum-mode following (Spirit "HTST", "EMA", "MMF").

Tangent coordinates: two orthonormal vectors t_i1, t_i2 perpendicular to every spin e_i.  The
Riemannian Hessian is

    H_(ia),(jb) = t_ia . (d^2E/de_i de_j) . t_jb  -  delta_ij delta_ab (e_i . dE/de_i)

and the undamped linearised LLG is  dq/dt = Gamma J H q,  Gamma_i = gamma/(mu_i mu_B), J = per-spin
[[0, 1], [-1, 0]], whose eigenvalues +-i omega are the magnon (eigenmode) frequencies.

HTST rate (Bessarab, Uzdin, Jonsson, PRB 85, 184409 (2012)), no zero modes:

    k = (v / 2 pi) sqrt( prod eps_min / prod' eps_sp ) exp(-dE / k_B T),
    v = sqrt( sum_{i>=2} a_i^2 / eps_i^sp ),   a_i = eps_i  q_1 . Gamma J q_i

(q_i the eigenvectors at the saddle point, q_1 the unstable mode).  For a single biaxial spin this
reduces to k = (f_FMR,min) exp(-dE/k_BT), which is used as a test.
"""
from __future__ import annotations

import numpy as np

from .spinmodel import GAMMA, K_B, MU_B


def tangent_basis(e):
    a = np.where(np.abs(e[:, :1]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))
    t1 = np.cross(e, a)
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
    t2 = np.cross(e, t1)
    return t1, t2


def _G(system, e):
    return system.gradient(e) if hasattr(system, "gradient") else system(e)[1]


def hessian(system, e, eps=1e-5):
    """Dense 2N x 2N tangent Hessian (meV/rad^2) by central differences of the gradient."""
    N = len(e)
    t1, t2 = tangent_basis(e)
    T = np.stack([t1, t2], 1)                               # (N, 2, 3)
    g0 = _G(system, e)
    H = np.zeros((2 * N, 2 * N))
    for j in range(N):
        for b in range(2):
            v = np.zeros_like(e)
            v[j] = T[j, b]
            dg = (_G(system, e + eps * v) - _G(system, e - eps * v)) / (2 * eps)
            H[:, 2 * j + b] = np.einsum("nka,na->nk", T, dg).ravel()
    H = 0.5 * (H + H.T)
    H -= np.diag(np.repeat(np.einsum("ia,ia->i", e, g0), 2))
    return H, T


def _gammaJ(system, N):
    Gm = GAMMA / (system.mu * MU_B)                         # rad/(ps meV)
    GJ = np.zeros((2 * N, 2 * N))
    for i in range(N):
        GJ[2 * i, 2 * i + 1] = Gm[i]
        GJ[2 * i + 1, 2 * i] = -Gm[i]
    return GJ


def llg_frequencies(system, e, H=None):
    """Positive eigenfrequencies (GHz) of the undamped linearised LLG about state e."""
    if H is None:
        H, _ = hessian(system, e)
    M = _gammaJ(system, len(e)) @ H
    w = np.linalg.eigvals(M)
    f = np.sort(np.abs(w.imag[w.imag > 1e-9])) / (2 * np.pi) * 1e3
    return f


def htst_rate(system, e_min, e_sp, T, zero_tol=1e-6):
    """HTST escape rate (1/s) from minimum e_min over saddle e_sp at temperature T (K).

    Returns dict(rate, prefactor (1/s), dE (meV), eps_min, eps_sp)."""
    N = len(e_min)
    Hm, _ = hessian(system, e_min)
    Hs, _ = hessian(system, e_sp)
    em = np.linalg.eigvalsh(Hm)
    es, Q = np.linalg.eigh(Hs)
    if np.sum(es < -zero_tol) != 1:
        raise ValueError(f"saddle point has {int(np.sum(es < -zero_tol))} negative modes (expected 1)")
    if np.any(np.abs(em) < zero_tol) or np.any(np.abs(es[1:]) < zero_tol):
        raise ValueError("zero (Goldstone) modes present: HTST with zero modes is not implemented")
    GJ = _gammaJ(system, N)
    q1 = Q[:, 0]
    a = es[1:] * (q1 @ GJ @ Q[:, 1:])
    v = np.sqrt(np.sum(a**2 / es[1:]))                      # 1/(ps sqrt(meV))
    logdet = 0.5 * (np.sum(np.log(em)) - np.sum(np.log(es[1:])))
    pref = v / (2 * np.pi) * np.exp(logdet) * 1e12           # 1/s
    dE = system.energy(e_sp) - system.energy(e_min)
    return dict(rate=float(pref * np.exp(-dE / (K_B * T))), prefactor=float(pref), dE=float(dE),
                eps_min=em, eps_sp=es)


def mmf(system, e, max_iter=5000, tol=1e-6, step=0.05, kick=0.05, seed=1, log=None):
    """Minimum-mode following from a minimum to a first-order saddle point (dense Hessian, small systems).

    Force inversion along the lowest Hessian mode v: F_eff = F - 2 (F.v) v if lambda_1 < 0, else -(F.v) v
    (climb along the softest mode).  Returns (saddle spins, info)."""
    from .minimize import _proj, _retract
    rng = np.random.default_rng(seed)
    e = e.copy()
    H, T = hessian(system, e)
    lam, V = np.linalg.eigh(H)
    v = np.einsum("nk,nka->na", V[:, 0].reshape(-1, 2), T)
    e = _retract(e, kick * v / np.max(np.linalg.norm(v, axis=1)) + 1e-4 * _proj(e, rng.normal(size=e.shape)))
    for it in range(max_iter):
        H, T = hessian(system, e)
        lam, V = np.linalg.eigh(H)
        v = np.einsum("nk,nka->na", V[:, 0].reshape(-1, 2), T)
        F = -_proj(e, _G(system, e))
        Fv = float(np.sum(F * v))
        Feff = F - 2 * Fv * v if lam[0] < 0 else -Fv * v / max(abs(Fv), 1e-300) * max(abs(Fv), 1e-2)
        fmax = float(np.max(np.linalg.norm(F, axis=1)))
        if lam[0] < 0 and fmax < tol:
            break
        d = Feff / max(abs(lam[0]), 1e-3) if lam[0] < 0 else Feff
        dm = float(np.max(np.linalg.norm(d, axis=1)))
        if dm > step:
            d *= step / dm
        e = _retract(e, _proj(e, d))
        if log and it % 100 == 0:
            log(f"  mmf {it:5d}  E = {system.energy(e):.6f}  lambda_1 = {lam[0]:+.3e}  |F| = {fmax:.2e}")
    return e, dict(iterations=it, lambda_1=float(lam[0]), max_force=fmax, energy=float(system.energy(e)))
