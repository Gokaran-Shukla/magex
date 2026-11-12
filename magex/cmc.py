"""Constrained Monte Carlo (Asselin, Evans, Chantrell et al., PRB 82, 054415 (2010)).

The direction of the total magnetisation is held along a constraint axis n (polar angle theta from
the reference axis, azimuth phi) by moving pairs of spins: a primary spin i takes a Gaussian trial
move, a compensating spin j absorbs the change of the transverse magnetisation, and the pair move is
accepted with

    P = min[1, (M_z'/M_z)^2 |e_jz| / |e_jz'| exp(-beta dE)]

(z along n).  The thermal average of the total internal torque T = sum_i e_i x (-dE/de_i) along the
rotation axis gives the free-energy derivative dF/dtheta = -<T>, so that the temperature-dependent
anisotropy K(T) is obtained from the torque curve (uniaxial: <T> = -K(T) sin 2 theta per spin).
Exchange tensors, DMI, single-ion anisotropy and Zeeman energy are included (numba kernel).
"""
from __future__ import annotations

import numpy as np

from .spinmodel import K_B, MU_B, normalize

try:
    from numba import njit
except Exception:  # pragma: no cover
    def njit(*a, **k):
        def deco(f):
            return f
        return deco if not (a and callable(a[0])) else a[0]


@njit(cache=True)
def _local_field(i, e, indptr, indices, data):
    h = np.zeros(3)
    for a in range(3):
        r = 3 * i + a
        for k in range(indptr[r], indptr[r + 1]):
            c = indices[k]
            h[a] += data[k] * e[c // 3, c % 3]
    return h


@njit(cache=True)
def _site_dE(i, eo, en, e, indptr, indices, data, A, mu, B, muB):
    h = _local_field(i, e, indptr, indices, data)
    dE = 0.0
    for a in range(3):
        d = en[a] - eo[a]
        dE += -2.0 * d * h[a] - muB * mu[i] * d * B[a]
        for b in range(3):
            dE += A[i, a, b] * (en[a] * en[b] - eo[a] * eo[b])
    return dE


@njit(cache=True)
def _cmc_sweeps(e, nsweeps, beta, sigma, frame, indptr, indices, data, A, mu, B, muB, seed):
    np.random.seed(seed)
    N = e.shape[0]
    t1, t2, n = frame[0], frame[1], frame[2]
    Mz = 0.0
    for i in range(N):
        Mz += e[i, 0] * n[0] + e[i, 1] * n[1] + e[i, 2] * n[2]
    acc = 0
    for s in range(nsweeps * N):
        i = np.random.randint(N)
        j = np.random.randint(N - 1)
        if j >= i:
            j += 1
        ei = e[i].copy()
        ej = e[j].copy()
        ni = ei + sigma * np.random.standard_normal(3)
        ni /= np.sqrt(ni[0] ** 2 + ni[1] ** 2 + ni[2] ** 2)
        xi, yi, zi = ei @ t1, ei @ t2, ei @ n
        xn, yn, zn = ni @ t1, ni @ t2, ni @ n
        xj, yj, zj = ej @ t1, ej @ t2, ej @ n
        xjn = xj + xi - xn
        yjn = yj + yi - yn
        r = 1.0 - xjn * xjn - yjn * yjn
        if r < 0:
            continue
        zjn = np.sqrt(r) if zj >= 0 else -np.sqrt(r)
        Mzn = Mz + zn - zi + zjn - zj
        if Mzn <= 0:
            continue
        nj = xjn * t1 + yjn * t2 + zjn * n
        dE = _site_dE(i, ei, ni, e, indptr, indices, data, A, mu, B, muB)
        e[i] = ni
        dE += _site_dE(j, ej, nj, e, indptr, indices, data, A, mu, B, muB)
        p = (Mzn / Mz) ** 2 * abs(zj) / max(abs(zjn), 1e-300) * np.exp(-min(beta * dE, 700.0))
        if p >= 1.0 or np.random.random() < p:
            e[j] = nj
            Mz = Mzn
            acc += 1
        else:
            e[i] = ei
    return acc / (nsweeps * N)


def _frame(theta, phi, ref=(0, 0, 1)):
    z = np.asarray(ref, float) / np.linalg.norm(ref)
    a = np.array([1.0, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1.0, 0])
    x = normalize(np.cross(a, z)[None])[0]
    x = np.cross(z, np.cross(x, z))
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    n = np.cos(theta) * z + np.sin(theta) * (np.cos(phi) * x + np.sin(phi) * y)
    rot_axis = -np.sin(phi) * x + np.cos(phi) * y            # d n / d theta = rot_axis x n
    t1 = np.cross(rot_axis, n)
    t1 /= np.linalg.norm(t1)
    return np.array([t1, np.cross(n, t1), n]), rot_axis


def constrained_mc(system, T, theta, phi=0.0, ref=(0, 0, 1), therm=2000, meas=4000, every=5, sigma=0.3,
                   seed=1, e0=None):
    """Thermal averages at fixed magnetisation direction.  Returns dict(m, torque (meV per spin), E, acc)."""
    if system.terms or system.has_higher:
        raise NotImplementedError("constrained MC supports exchange, DMI, single-ion anisotropy and field")
    frame, rax = _frame(theta, phi, ref)
    J = system.Jsp.tocsr()
    J.sort_indices()
    args = (J.indptr.astype(np.int64), J.indices.astype(np.int64), J.data.astype(float),
            np.ascontiguousarray(system.A, float), system.mu.astype(float), np.asarray(system.B, float), MU_B)
    e = np.tile(frame[2], (system.N, 1)) if e0 is None else e0.copy()
    beta = 1.0 / (K_B * max(T, 1e-9))
    rng = np.random.default_rng(seed)
    a = _cmc_sweeps(e, therm, beta, sigma, frame, *args, int(rng.integers(1 << 30)))
    ms, tq, Es = [], [], []
    for k in range(meas // every):
        a = _cmc_sweeps(e, every, beta, sigma, frame, *args, int(rng.integers(1 << 30)))
        g = system.gradient(e)
        tau = np.cross(e, -g).sum(0)
        tq.append(float(tau @ rax) / system.N)
        ms.append(float(e.mean(0) @ frame[2]))
        Es.append(system.energy(e, per_spin=True))
    tq = np.array(tq)
    return dict(m=float(np.mean(ms)), torque=float(tq.mean()), torque_err=float(tq.std() / np.sqrt(len(tq) / 10)),
                E=float(np.mean(Es)), acc=float(a), spins=e)


def anisotropy_vs_T(system, Ts, theta=np.pi / 4, ref=(0, 0, 1), log=None, **kw):
    """K(T) (meV per spin) of a uniaxial magnet from the constrained torque at theta (<T> = -K sin 2theta)."""
    rows = []
    for T in Ts:
        r = constrained_mc(system, T, theta, ref=ref, **kw)
        K = -r["torque"] / np.sin(2 * theta)
        rows.append((T, r["m"], K, r["torque_err"] / abs(np.sin(2 * theta))))
        if log:
            log(f"  CMC T = {T:7.2f} K: m = {r['m']:.4f}, K = {K:.5f} +- {rows[-1][3]:.5f} meV/spin, acc = {r['acc']:.2f}")
    return np.array(rows)
