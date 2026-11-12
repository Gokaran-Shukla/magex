"""Quantum corrections to classical spin thermodynamics.

1. Quantum thermostat for the stochastic LLG (Barker & Bauer, PRB 100, 140401(R) (2019)).  The white Brown noise of power 2D is replaced by coloured noise with the
   Planck spectrum (no zero-point term)

       S(omega) = 2 D  x / (e^x - 1),      x = hbar omega / k_B T,

   so magnons of energy hbar omega are populated with Bose rather than equipartition statistics.  The
   spectrum is represented as a positive mixture of Ornstein-Uhlenbeck processes,

       x/(e^x - 1)  ~=  sum_k w_k g_k^2 / (g_k^2 + x^2),     w_k >= 0  (non-negative least squares)

   each integrated exactly (no time-step error in the noise), with relaxation rates
   gamma_k = g_k k_B T / hbar.  The fit is accurate to ~1e-3 for 0 <= x <= 20.

2. RPA (Tyablikov) ordering temperature from the exchange of results.json (Pajda et al., PRB 64,
   174402 (2001); Rusz et al., PRB 71, 184433 (2005) for several sublattices):

       k_B T_RPA = (2/3) [ (1/(n N_q)) sum_q Tr( D - J~(q) )^{-1} ]^{-1},  D_ij = delta_ij sum_l J~_il(0)

   with J~_ij = s_i s_j J_ij (sublattice signs of the reference state).  For one sublattice this is
   Pajda's formula; it lies between the mean-field and the classical Monte Carlo values and is the
   standard first-principles estimate for localized moments.

3. Quantum spin-length rescaling of classical temperatures, T_q = T_cl (S+1)/S with S = |mu|/2
   (localized-moment systems; not appropriate for itinerant metals such as Fe).
"""
from __future__ import annotations

import numpy as np

from .spinmodel import K_B

HBAR_MEV_PS = 0.6582119569     # meV ps


def planck(x):
    x = np.asarray(x, float)
    out = np.ones_like(x)
    nz = x > 1e-8
    out[nz] = x[nz] / np.expm1(x[nz])
    return out


def fit_planck(xmax=30.0, npts=4000):
    """Positive mixture fit of x/(e^x - 1) (max abs error ~2e-3 on [0, 30]).

    Basis: damped oscillators O^4/((O^2 - x^2)^2 + G^2 x^2) (decay ~ x^-4)
    and Lorentzians g^2/(g^2 + x^2).  Returns list of (kind, O_or_g, G, weight).
    """
    from scipy.optimize import nnls
    x = np.linspace(0, xmax, npts)
    cols, lab = [], []
    for O in np.geomspace(0.5, 15, 16):
        for G in (0.5, 1, 2, 4, 8):
            cols.append(O**4 / ((O**2 - x**2) ** 2 + G**2 * x**2))
            lab.append(("osc", O, G))
    for g in np.geomspace(0.2, 20, 10):
        cols.append(g**2 / (g**2 + x**2))
        lab.append(("lor", g, 0.0))
    Phi = np.array(cols).T
    wt = np.sqrt(1 / (0.02 + planck(x)))
    w, _ = nnls(Phi * wt[:, None], planck(x) * wt)
    return [(k, o, G, float(wi)) for (k, o, G), wi in zip(lab, w) if wi > 1e-8]


class QuantumNoise:
    """Coloured thermal field with the Planck spectrum for N spins (3 components each).

    sigma2_white  variance per step of the classical white-noise field, 2D/dt (T^2), per spin
    The field b(t) has spectral density 2D x/(e^x - 1), x = hbar omega/k_B T, i.e. the classical
    noise at low frequency and Bose-suppressed noise for hbar omega >> k_B T.  Every mode is a
    linear SDE integrated exactly over dt.
    """

    def __init__(self, N, sigma2_white, dt, T, seed=None):
        from scipy.linalg import expm
        self.rng = np.random.default_rng(seed)
        self.dt = float(dt)
        self.T = float(T)
        w0 = K_B * self.T / HBAR_MEV_PS                    # k_B T / hbar in rad/ps
        D = np.asarray(sigma2_white, float) * self.dt / 2.0 * np.ones(N)
        self.modes = fit_planck()
        self.lor, self.osc = [], []
        for kind, o, G, w in self.modes:
            if kind == "lor":
                gam = o * w0
                d = np.exp(-gam * self.dt)
                # unit-variance OU, amplitude a^2 = D w gamma
                self.lor.append((d, np.sqrt(1 - d * d), np.sqrt(D * w * gam)))
            else:
                Om, Ga = o * w0, G * w0
                A = np.array([[0.0, 1.0], [-Om**2, -Ga]])
                F = expm(A * self.dt)
                S = np.diag([1 / (2 * Ga * Om**2), 1 / (2 * Ga)])   # stationary covariance for c = 1
                Q = S - F @ S @ F.T
                ev, U = np.linalg.eigh(0.5 * (Q + Q.T))
                L = U * np.sqrt(np.clip(ev, 0, None))
                c = np.sqrt(2 * D * w * Om**4)                        # per-spin amplitude
                self.osc.append((F, L, c, S))
        self.xl = self.rng.normal(size=(len(self.lor), N, 3))
        self.xo = np.zeros((len(self.osc), 2, N, 3))
        for k, (F, L, c, S) in enumerate(self.osc):
            self.xo[k, 0] = np.sqrt(S[0, 0]) * self.rng.normal(size=(N, 3))
            self.xo[k, 1] = np.sqrt(S[1, 1]) * self.rng.normal(size=(N, 3))

    def __call__(self):
        b = 0.0
        for k, (d, kick, a) in enumerate(self.lor):
            self.xl[k] = d * self.xl[k] + kick * self.rng.normal(size=self.xl[k].shape)
            b = b + a[:, None] * self.xl[k]
        for k, (F, L, c, S) in enumerate(self.osc):
            x = self.xo[k]
            z = self.rng.normal(size=x.shape)
            self.xo[k] = np.einsum("ab,bnc->anc", F, x) + np.einsum("ab,bnc->anc", L, z)
            b = b + c[:, None] * self.xo[k, 0]
        return b


# ----------------------------------------------------------------------------- RPA
def _Jq(uc, q):
    """Sign-gauged isotropic J~(q) (ns x ns) for fractional q (3,)."""
    ns = uc.nsites
    M = np.zeros((ns, ns), complex)
    for (i, j, R, Jt) in uc.couplings:
        Jiso = np.trace(np.asarray(Jt)) / 3 * uc.signs[i] * uc.signs[j]
        M[i, j] += Jiso * np.exp(2j * np.pi * np.dot(q, R))
    return 0.5 * (M + M.conj().T)


def anisotropy_gap(uc):
    """Uniaxial gap K_i (meV) per site about the reference axis: single-ion + two-site anisotropy."""
    ns = uc.nsites
    n = np.asarray(uc.ref_dirs[0], float)
    n /= np.linalg.norm(n)
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(n, a)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    K = np.zeros(ns)
    for i in range(ns):
        A = np.asarray(uc.A[i])
        K[i] += 0.5 * (e1 @ A @ e1 + e2 @ A @ e2) - n @ A @ n
    for (i, j, R, Jt) in uc.couplings:
        JS = 0.5 * (np.asarray(Jt) + np.asarray(Jt).T) * uc.signs[i] * uc.signs[j]
        K[i] += n @ JS @ n - 0.5 * (e1 @ JS @ e1 + e2 @ JS @ e2)
    return K


def rpa_tc(uc, nq=24, anisotropy=True, log=None):
    """Tyablikov RPA ordering temperature (K) of a UnitCellModel; returns (T_RPA, T_MF).

    With ``anisotropy`` the uniaxial gap (single-ion + two-site) is added to the magnon
    denominator, which keeps T_RPA finite for 2D magnets (Mermin-Wagner otherwise gives 0).
    """
    from .spinmodel import mean_field_tc
    ns = uc.nsites
    J0 = _Jq(uc, np.zeros(3)).real
    diag = J0.sum(1)
    if anisotropy:
        diag = diag + np.maximum(anisotropy_gap(uc), 0.0)
    D = np.diag(diag)
    dims = [k for k in range(3) if any(abs(R[k]) > 0 for (_, _, R, _) in uc.couplings)]
    g = (np.arange(nq) + 0.5) / nq
    axes = [g if k in dims else np.zeros(1) for k in range(3)]
    qs = np.array(np.meshgrid(*axes, indexing="ij")).reshape(3, -1).T
    acc = 0.0
    bad = 0
    for q in qs:
        ev = np.linalg.eigvalsh(D - _Jq(uc, q))
        if np.any(ev <= 1e-9):
            bad += 1
            continue
        acc += np.sum(1.0 / ev)
    if bad and log:
        log(f"  RPA: {bad} q-points with non-positive magnon energy skipped (reference not stable?)")
    mean = acc / (ns * (len(qs) - bad))
    T_rpa = (2.0 / 3.0) / (mean * K_B)
    return float(T_rpa), float(mean_field_tc(uc))


def quantum_rescale(T, moments):
    """T (S+1)/S with S = |mu|/2 averaged over sites."""
    S = np.mean(np.abs(np.asarray(moments, float))) / 2.0
    return float(T * (S + 1) / S), float(S)
