"""Advanced Monte Carlo: parallel tempering (replica exchange) and Wang-Landau sampling.

Parallel tempering (Swendsen & Wang 1986; Hukushima & Nemoto 1996): replicas at temperatures
T_1 < ... < T_M evolve with the vectorised Metropolis sweeps of magex.montecarlo; every
``swap_every`` sweeps neighbouring replicas exchange configurations with probability
min(1, exp[(beta_i - beta_j)(E_i - E_j)]).  Essential for frustrated magnets, spin glasses and
first-order transitions where single-temperature runs freeze.

Wang-Landau (Wang & Landau PRL 86, 2050 (2001)) with the 1/t refinement (Belardinelli & Pereyra,
PRE 75, 046701 (2007)): a random walk in energy with acceptance min(1, g(E_old)/g(E_new)) builds the
density of states g(E); all canonical averages at any temperature follow by reweighting:
Z = sum g e^{-beta E}, <E>, C(T) = (<E^2> - <E>^2)/k_B T^2, entropy and free energy.  The kernel is
compiled with numba when available (pip install numba), otherwise runs in pure Python (small
systems only).  Bilinear exchange, single-ion anisotropy and Zeeman energies are included.
"""
from __future__ import annotations

import time

import numpy as np

from .montecarlo import MetropolisMC
from .spinmodel import K_B, MU_B

try:  # optional JIT
    from numba import njit
    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    HAVE_NUMBA = False

    def njit(*a, **k):
        def deco(f):
            return f
        return deco if not (a and callable(a[0])) else a[0]


# ----------------------------------------------------------------------------- parallel tempering
def parallel_tempering(system, temperatures, therm_sweeps=2000, meas_sweeps=4000, sample_every=2, swap_every=5,
                       seed=1, init="reference", init_axis=(0, 0, 1), sigma=0.6, overrelax=1, log=None):
    """Returns (rows like magex.montecarlo._point, swap acceptance per neighbouring pair)."""
    Ts = np.array(sorted(float(t) for t in temperatures))
    M = len(Ts)
    rng = np.random.default_rng(seed)
    reps = [MetropolisMC(system, T, seed + 17 * k, sigma, overrelax) for k, T in enumerate(Ts)]
    conf = [system.init_spins(init, init_axis, seed + k) for k in range(M)]
    E = np.array([system.energy(c) for c in conf])
    swaps_try = np.zeros(M - 1)
    swaps_acc = np.zeros(M - 1)
    samples = [[] for _ in range(M)]
    t0 = time.time()
    total = therm_sweeps + meas_sweeps
    for n in range(total):
        for k in range(M):
            conf[k] = reps[k].sweep(conf[k])
            if n < therm_sweeps and (n + 1) % 50 == 0:
                a = reps[k].acceptance()
                reps[k].sigma = float(np.clip(reps[k].sigma * 0.5 / max(1 - a, 0.05), 0.02, 4.0))
        if (n + 1) % swap_every == 0:
            E = np.array([system.energy(c) for c in conf])
            start = (n // swap_every) % 2
            for k in range(start, M - 1, 2):
                d = (1 / (K_B * Ts[k]) - 1 / (K_B * Ts[k + 1])) * (E[k] - E[k + 1])
                swaps_try[k] += 1
                if d >= 0 or rng.random() < np.exp(d):
                    conf[k], conf[k + 1] = conf[k + 1], conf[k]
                    E[k], E[k + 1] = E[k + 1], E[k]
                    swaps_acc[k] += 1
        if n >= therm_sweeps and (n - therm_sweeps) % sample_every == 0:
            for k in range(M):
                v = (system.signs[:, None] * conf[k]).sum(0) / system.N
                samples[k].append((system.energy(conf[k]), np.linalg.norm(v)))
    N = system.N
    rows = []
    for k, T in enumerate(Ts):
        Es = np.array([s[0] for s in samples[k]])
        ms = np.array([s[1] for s in samples[k]])
        kT = K_B * T

        def obs(E, m):
            return (np.mean(E) / N, (np.mean(E**2) - np.mean(E) ** 2) / (N * kT**2), np.mean(m),
                    N * (np.mean(m**2) - np.mean(m) ** 2) / kT, 1 - np.mean(m**4) / (3 * np.mean(m**2) ** 2))
        full = obs(Es, ms)
        nb = 10
        L = len(Es) // nb
        err = (np.array([obs(Es[b * L:(b + 1) * L], ms[b * L:(b + 1) * L]) for b in range(nb)]).std(0, ddof=1)
               / np.sqrt(nb)) if L > 2 else np.zeros(5)
        rows.append(dict(L=list(system.L), N=N, T=float(T), E=full[0], C=full[1], m=full[2], chi=full[3], U4=full[4],
                         dE=err[0], dC=err[1], dm=err[2], dchi=err[3], dU4=err[4], acceptance=reps[k].acceptance(),
                         sigma=reps[k].sigma, ncolors=reps[k].ncolors, mvec=[0, 0, 0], seconds=time.time() - t0))
    swap = (swaps_acc / np.maximum(swaps_try, 1)).tolist()
    if log:
        log(f"  parallel tempering L={system.L}: {M} replicas, swap acceptance "
            f"{np.round(swap, 2).tolist()}, {time.time() - t0:.1f} s")
    return rows, swap


# ----------------------------------------------------------------------------- Wang-Landau
@njit(cache=True)
def _wl_kernel(indptr, indices, data, A, mu, B, muB, e, E, Emin, Emax, nb, lng, H, lnf_final, flat, sigma,
               max_moves, seed, check_every):
    np.random.seed(seed)
    N = e.shape[0]
    width = (Emax - Emin) / nb
    lnf = 1.0
    moves = 0
    phase_t = False
    while moves < max_moves:
        i = np.random.randint(N)
        eo0, eo1, eo2 = e[i, 0], e[i, 1], e[i, 2]
        n0 = eo0 + sigma * np.random.randn()
        n1 = eo1 + sigma * np.random.randn()
        n2 = eo2 + sigma * np.random.randn()
        nn = np.sqrt(n0 * n0 + n1 * n1 + n2 * n2)
        n0 /= nn
        n1 /= nn
        n2 /= nn
        h0 = 0.0
        h1 = 0.0
        h2 = 0.0
        for c in range(3):
            r = 3 * i + c
            s = 0.0
            for p in range(indptr[r], indptr[r + 1]):
                col = indices[p]
                s += data[p] * e[col // 3, col % 3]
            if c == 0:
                h0 = s
            elif c == 1:
                h1 = s
            else:
                h2 = s
        d0, d1, d2 = n0 - eo0, n1 - eo1, n2 - eo2
        dE = -2.0 * (d0 * h0 + d1 * h1 + d2 * h2) - muB * mu[i] * (d0 * B[0] + d1 * B[1] + d2 * B[2])
        an = 0.0
        ao = 0.0
        nv = (n0, n1, n2)
        ov = (eo0, eo1, eo2)
        for a in range(3):
            for b in range(3):
                an += nv[a] * A[i, a, b] * nv[b]
                ao += ov[a] * A[i, a, b] * ov[b]
        dE += an - ao
        En = E + dE
        bo = int((E - Emin) / width)
        if bo >= nb:
            bo = nb - 1
        if En >= Emin and En < Emax:
            bn = int((En - Emin) / width)
            if bn >= nb:
                bn = nb - 1
            if lng[bo] - lng[bn] >= 0 or np.random.random() < np.exp(lng[bo] - lng[bn]):
                e[i, 0], e[i, 1], e[i, 2] = n0, n1, n2
                E = En
                bo = bn
        lng[bo] += lnf
        H[bo] += 1
        moves += 1
        if moves % check_every == 0:
            t = moves / nb
            if phase_t:
                lnf = 1.0 / t
            else:
                hm = 0.0
                hmin = 1e300
                cnt = 0
                for k in range(nb):
                    if H[k] > 0:
                        hm += H[k]
                        cnt += 1
                        if H[k] < hmin:
                            hmin = H[k]
                if cnt > 0 and hmin > flat * hm / cnt and cnt > nb // 2:
                    lnf *= 0.5
                    for k in range(nb):
                        H[k] = 0
                    if lnf < 1.0 / t:
                        phase_t = True
                        lnf = 1.0 / t
            if lnf < lnf_final:
                break
    return E, moves, lnf


def wang_landau(system, nbins=120, E_window=None, lnf_final=1e-7, flatness=0.7, sigma=0.5, max_sweeps=2000000,
                seed=1, log=None):
    """Density of states ln g(E) on an energy window (meV).  Returns dict with E (bin centres), lng."""
    e = system.init_spins("reference").copy()
    E0 = system.energy(e)
    if E_window is None:            # skip the vanishing density of states right at the ground state
        E_window = (E0 + 0.03 * abs(E0), E0 + 0.95 * abs(E0))
    Emin, Emax = map(float, E_window)
    E = E0
    if not (Emin <= E < Emax):              # bring the walker into the window by cooling from high T
        from .spinmodel import mean_field_tc
        Tmf = max(mean_field_tc(system.uc), 1.0)
        e = system.init_spins("random", seed=seed)
        for T in np.geomspace(3 * Tmf, 0.02 * Tmf, 60):
            mc = MetropolisMC(system, T, seed)
            for _ in range(10):
                e = mc.sweep(e)
            E = system.energy(e)
            if Emin <= E < Emax:
                break
        if not (Emin <= E < Emax):
            raise RuntimeError("could not reach the Wang-Landau energy window; give E_window explicitly")
    e = np.ascontiguousarray(e, dtype=np.float64)
    J = system.Jsp.tocsr()
    lng = np.zeros(nbins)
    H = np.zeros(nbins)
    t0 = time.time()
    E, moves, lnf = _wl_kernel(J.indptr.astype(np.int64), J.indices.astype(np.int64), J.data.astype(np.float64),
                               system.A.astype(np.float64), system.mu.astype(np.float64), system.B.astype(np.float64),
                               MU_B, e, float(E), Emin, Emax, nbins, lng, H, lnf_final, flatness, sigma,
                               int(max_sweeps * system.N), seed, int(system.N * 10))
    Ec = Emin + (np.arange(nbins) + 0.5) * (Emax - Emin) / nbins
    visited = lng > 0
    lng = lng - lng[visited].min()
    if log:
        log(f"  Wang-Landau: {moves} moves ({moves / system.N:.0f} sweeps), final ln f = {lnf:.2e}, "
            f"{visited.sum()}/{nbins} bins visited, {time.time() - t0:.1f} s ({'numba' if HAVE_NUMBA else 'python'})")
    return dict(E=Ec[visited], lng=lng[visited], N=system.N, lnf=lnf, moves=int(moves))


def wl_thermodynamics(wl, temperatures):
    """<E>/N, C/N (k_B), entropy S/N (k_B) and free energy F/N (meV) from ln g(E)."""
    E, lng, N = np.asarray(wl["E"]), np.asarray(wl["lng"]), wl["N"]
    out = []
    for T in np.atleast_1d(temperatures):
        b = 1 / (K_B * T)
        w = lng - b * E
        w0 = w.max()
        p = np.exp(w - w0)
        Z = p.sum()
        Em = (p * E).sum() / Z
        E2 = (p * E**2).sum() / Z
        C = (E2 - Em**2) * b**2 / N
        lnZ = np.log(Z) + w0
        F = -K_B * T * lnZ
        S = (Em - F) / (K_B * T)
        out.append(dict(T=float(T), E=Em / N, C=C, S=S / N, F=F / N))
    return out
