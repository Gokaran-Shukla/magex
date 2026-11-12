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
        # zero (Goldstone / translational) modes: use the zero-mode formulation
        return htst_rate_zero_modes(system, e_min, e_sp, T)
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
    if kick > 0:
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


# ============================================================================ scalable versions
def sparse_hessian(system, e, eps=1e-5):
    """Tangent Hessian as (H_local sparse 2N x 2N, H_nonlocal LinearOperator or None, T (N,2,3)).

    Local part analytic: -2 J (exchange tensors) + 2 A (single-ion) + on-site terms (cubic, higher uniaxial,
    by per-component finite differences, which is exact for on-site terms), minus the curvature term
    diag(e.g).  Non-local terms (dipolar, biquadratic, four-spin, quadruplets) enter matrix-free through
    central differences of their gradient."""
    import scipy.sparse as sp
    from scipy.sparse.linalg import LinearOperator
    from .terms import CubicAnisotropy, UniaxialHigher
    N = len(e)
    t1, t2 = tangent_basis(e)
    T = np.stack([t1, t2], 1)
    rows = np.repeat(np.arange(N), 6)
    Tm = sp.csr_matrix((T.transpose(0, 2, 1).reshape(-1), (np.repeat(3 * np.arange(N), 6) + np.tile([0, 0, 1, 1, 2, 2], N),
                                                          np.repeat(2 * np.arange(N), 6) + np.tile([0, 1], 3 * N))),
                       shape=(3 * N, 2 * N))
    K = -(system.Jsp + system.Jsp.T)
    if system.has_A:
        blocks = 2 * system.A
        K = K + sp.block_diag(list(blocks), format="csr") if N < 2 else K + _blockdiag(blocks)
    onsite = [t for t in system.terms if isinstance(t, (CubicAnisotropy, UniaxialHigher))]
    nonloc = [t for t in system.terms if not isinstance(t, (CubicAnisotropy, UniaxialHigher))]
    if onsite:
        B = np.zeros((N, 3, 3))
        for a in range(3):
            d = np.zeros_like(e)
            d[:, a] = eps
            gp = sum(t.gradient(e + d) for t in onsite)
            gm = sum(t.gradient(e - d) for t in onsite)
            B[:, :, a] = (gp - gm) / (2 * eps)
        K = K + _blockdiag(0.5 * (B + B.transpose(0, 2, 1)))
    g0 = system.gradient(e)
    Hl = (Tm.T @ K @ Tm).tocsr()
    Hl = Hl - sp.diags(np.repeat(np.einsum("ia,ia->i", e, g0), 2))
    Hn = None
    if nonloc or system.has_higher:
        def grad_nl(x):
            g = np.zeros_like(x)
            if system.has_higher:
                g += system.higher_gradient(x)
            for t in nonloc:
                g += t.gradient(x)
            return g

        def mv(v):
            d = (Tm @ np.asarray(v).ravel()).reshape(N, 3)
            dg = (grad_nl(e + eps * d) - grad_nl(e - eps * d)) / (2 * eps)
            return Tm.T @ dg.ravel()
        Hn = LinearOperator((2 * N, 2 * N), matvec=mv, dtype=float)
    if system.pinned is not None:           # pinned spins: no degrees of freedom (stiff modes)
        P = np.repeat(system.pinned, 2)
        Hl = Hl + sp.diags(np.where(P, 1e6, 0.0))
    return Hl, Hn, T


def _blockdiag(blocks):
    import scipy.sparse as sp
    N = len(blocks)
    r = np.repeat(3 * np.arange(N), 9) + np.tile(np.repeat(np.arange(3), 3), N)
    c = np.repeat(3 * np.arange(N), 9) + np.tile(np.tile(np.arange(3), 3), N)
    return sp.csr_matrix((np.asarray(blocks).reshape(-1), (r, c)), shape=(3 * N, 3 * N))


def hessian_operator(system, e):
    from scipy.sparse.linalg import LinearOperator, aslinearoperator
    Hl, Hn, T = sparse_hessian(system, e)
    if Hn is None:
        return Hl, T
    A = aslinearoperator(Hl) + Hn
    return LinearOperator(A.shape, matvec=A.matvec, dtype=float), T


def lowest_modes(system, e, k=10, tol=1e-10):
    """k lowest Hessian eigenpairs (meV) of state e; sparse shift-invert when the Hessian is local,
    LOBPCG/Lanczos otherwise.  Returns (eigenvalues, eigenvectors (2N, k), T)."""
    from scipy.sparse.linalg import eigsh
    N = len(e)
    if 2 * N <= 1200:
        H, T = hessian(system, e)
        ev, V = np.linalg.eigh(H)
        return ev[:k], V[:, :k], T
    H, T = hessian_operator(system, e)
    import scipy.sparse as sp
    if sp.issparse(H):
        shift = -1e-3 - abs(float(H.diagonal().min()))
        try:
            ev, V = eigsh(H, k=k, sigma=shift, which="LM", tol=tol)
        except Exception:
            ev, V = eigsh(H, k=k, which="SA", tol=tol)
    else:
        ev, V = eigsh(H, k=k, which="SA", tol=tol, maxiter=20000)
    o = np.argsort(ev)
    return ev[o], V[:, o], T


def llg_frequencies_sparse(system, e, k=10):
    """k lowest LLG eigenfrequencies (GHz) of a stable state from the generalised symmetric problem
    (G^T H G) y = w^2 H^-1 y,  G = Gamma J  (shift-invert, scales to 10^5-10^6 spins for local Hessians)."""
    import scipy.sparse as sp
    from scipy.sparse.linalg import LinearOperator, eigsh, splu, cg
    N = len(e)
    H, T = hessian_operator(system, e)
    Gm = np.repeat(GAMMA / (system.mu * MU_B), 2)

    def Gv(v):                                   # G = Gamma J, J = per-spin [[0, 1], [-1, 0]]
        v = v.reshape(-1, 2)
        return (Gm.reshape(-1, 2) * np.stack([v[:, 1], -v[:, 0]], 1)).ravel()

    def GTv(v):
        v = v.reshape(-1, 2)
        return (np.stack([-(Gm.reshape(-1, 2)[:, 1] * v[:, 1]), Gm.reshape(-1, 2)[:, 0] * v[:, 0]], 1)).ravel()

    def Ginv(v):                                 # G^-1 = J^T Gamma^-1
        v = v.reshape(-1, 2) / Gm.reshape(-1, 2)
        return np.stack([-v[:, 1], v[:, 0]], 1).ravel()

    def GTinv(v):
        v = v.reshape(-1, 2) / Gm.reshape(-1, 2)
        return np.stack([v[:, 1], -v[:, 0]], 1).ravel()

    if sp.issparse(H):
        lu = splu(H.tocsc())
        Hinv = lu.solve
    else:
        def Hinv(v):
            x, info = cg(H, v, rtol=1e-12, maxiter=20000)
            return x
    Hm = (lambda v: H @ v)
    A = LinearOperator((2 * N, 2 * N), matvec=lambda v: GTv(Hm(Gv(v))), dtype=float)
    Ainv = LinearOperator((2 * N, 2 * N), matvec=lambda v: Ginv(Hinv(GTinv(v))), dtype=float)
    Mop = LinearOperator((2 * N, 2 * N), matvec=Hinv, dtype=float)
    Minv = LinearOperator((2 * N, 2 * N), matvec=Hm, dtype=float)
    # every frequency appears twice (the two quadratures of a mode): ask for 2k and keep one of each pair
    w2 = eigsh(A, k=min(2 * k, 2 * N - 2), M=Mop, sigma=0.0, OPinv=Ainv, which="LM", return_eigenvectors=False)
    f = np.sort(np.sqrt(np.abs(w2))) / (2 * np.pi) * 1e3
    return f[::2][:k]


def translation_generators(system, e, axes=(0, 1)):
    """Tangent vectors (2N, n) of rigid translations along the periodic axes, d e / d x_a (rad/A)."""
    from .structure import gradient_operators
    D = gradient_operators(system)
    t1, t2 = tangent_basis(e)
    out = []
    for a in axes:
        g = np.stack([D[a] @ e[:, c] for c in range(3)], 1)
        out.append(np.stack([np.einsum("ia,ia->i", g, t1), np.einsum("ia,ia->i", g, t2)], 1).ravel())
    return np.stack(out, 1)


def zero_mode_volume(system, e, axes=(0, 1)):
    """Volume of the translational zero-mode manifold: (area swept) x sqrt(det Gram(d e/d x_a))."""
    G = translation_generators(system, e, axes)
    L = [np.linalg.norm(system.L[a] * system.uc.cell[a]) for a in axes]
    return float(np.prod(L) * np.sqrt(abs(np.linalg.det(G.T @ G))))


def htst_rate_zero_modes(system, e_min, e_sp, T, n_zero=None, volumes=None, axes=(0, 1), alpha=0.0, zero_tol=1e-5):
    """HTST rate (1/s) with zero modes (Bessarab, Uzdin & Jonsson 2012; as in Spirit):

        k = (v/2pi) sqrt(prod' eps_min / prod'' eps_sp) (2 pi k_B T)^((n0_min - n0_sp)/2) (V_sp/V_min) e^(-dE/kT)

    prod' / prod'' skip the zero modes (and the unstable mode), v = sqrt(sum a_i^2/eps_i) over the
    non-zero stable modes of the saddle, a_i = eps_i q_1 . (Gamma/(1+alpha^2)) J q_i.  ``volumes`` =
    (V_min, V_sp) for arbitrary zero modes; by default translational zero modes along ``axes`` are assumed
    and their volumes are computed from d e / d x.  Dense eigen-decomposition (N up to a few thousand)."""
    N = len(e_min)
    Hm, _ = hessian(system, e_min)
    Hs, _ = hessian(system, e_sp)
    em = np.linalg.eigvalsh(Hm)
    es, Q = np.linalg.eigh(Hs)
    scale = max(np.abs(em).max(), np.abs(es).max())
    zm = np.abs(em) < zero_tol * scale
    zs = np.abs(es) < zero_tol * scale
    neg = es < -zero_tol * scale
    if neg.sum() != 1:
        raise ValueError(f"saddle point has {int(neg.sum())} negative modes (expected 1)")
    n0m, n0s = int(zm.sum()), int(zs.sum())
    if n_zero is not None and (n0m, n0s) != tuple(n_zero):
        raise ValueError(f"found {n0m}/{n0s} zero modes at minimum/saddle, expected {tuple(n_zero)}")
    if volumes is None and (n0m or n0s):
        Vm = zero_mode_volume(system, e_min, axes) if n0m else 1.0
        Vs = zero_mode_volume(system, e_sp, axes) if n0s else 1.0
    else:
        Vm, Vs = volumes if volumes is not None else (1.0, 1.0)
    GJ = _gammaJ(system, N) / (1 + alpha**2)
    i1 = int(np.where(neg)[0][0])
    stab = ~neg & ~zs
    q1 = Q[:, i1]
    a = es[stab] * (q1 @ GJ @ Q[:, stab])
    v = np.sqrt(np.sum(a**2 / es[stab]))
    logdet = 0.5 * (np.sum(np.log(em[~zm])) - np.sum(np.log(es[stab])))
    kT = K_B * T
    pref = v / (2 * np.pi) * np.exp(logdet) * (2 * np.pi * kT) ** ((n0m - n0s) / 2) * (Vs / Vm) * 1e12
    dE = system.energy(e_sp) - system.energy(e_min)
    return dict(rate=float(pref * np.exp(-dE / kT)), prefactor=float(pref), dE=float(dE), n_zero=(n0m, n0s),
                volumes=(Vm, Vs), eps_min=em, eps_sp=es)


def mmf_sparse(system, e, max_iter=5000, tol=1e-6, step=0.05, kick=0.05, seed=1, log=None):
    """Minimum-mode following with the lowest mode from Lanczos on the (sparse / matrix-free) Hessian."""
    from .minimize import _proj, _retract
    rng = np.random.default_rng(seed)
    e = e.copy()
    if kick > 0:
        lam, V, T = lowest_modes(system, e, k=2)
        v = np.einsum("nk,nka->na", V[:, 0].reshape(-1, 2), T)
        e = _retract(e, kick * v / np.max(np.linalg.norm(v, axis=1)) + 1e-4 * _proj(e, rng.normal(size=e.shape)))
    for it in range(max_iter):
        lam, V, T = lowest_modes(system, e, k=2)
        v = np.einsum("nk,nka->na", V[:, 0].reshape(-1, 2), T)
        F = -_proj(e, _G(system, e))
        Fv = float(np.sum(F * v))
        fmax = float(np.max(np.linalg.norm(F, axis=1)))
        if lam[0] < 0 and fmax < tol:
            break
        Feff = F - 2 * Fv * v if lam[0] < 0 else -np.sign(Fv if Fv != 0 else 1.0) * v * max(abs(Fv), 1e-2)
        d = Feff / max(abs(lam[0]), 1e-3) if lam[0] < 0 else Feff
        dm = float(np.max(np.linalg.norm(d, axis=1)))
        if dm > step:
            d *= step / dm
        e = _retract(e, _proj(e, d))
        if log and it % 50 == 0:
            log(f"  mmf {it:5d}  E = {system.energy(e):.6f}  lambda_1 = {lam[0]:+.3e}  |F| = {fmax:.2e}")
    return e, dict(iterations=it, lambda_1=float(lam[0]), max_force=fmax, energy=float(system.energy(e)))


def htst_refined(system, e_min, e_sp_guess, T, tol=1e-9, log=None):
    """HTST rate after polishing both stationary points: L-BFGS for the minimum, minimum-mode following
    from the (climbing-image) saddle guess, so that the harmonic eigenvalues are converged."""
    from .minimize import minimize
    e_min, _ = minimize(system, e_min, "lbfgs", tol=tol)
    e_sp, info = (mmf if system.N <= 1500 else mmf_sparse)(system, e_sp_guess, tol=max(tol, 1e-8), kick=0.0)
    if log:
        log(f"  saddle refined by minimum-mode following: |F| = {info['max_force']:.1e}, lambda_1 = {info['lambda_1']:.3e}")
    return htst_rate_zero_modes(system, e_min, e_sp, T)
