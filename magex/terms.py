"""Additional energy terms for the atomistic spin model (``SpinSystem.terms``).

Each term implements
    energy(e) -> float (meV)
    gradient(e) -> (N, 3) dE/de (meV)
    local_energy(idx, en, e) -> (n,) energy of all contributions containing site idx[k] with e[idx[k]] -> en[k]
    bonds() -> extra (M, 2) site pairs that must not be updated simultaneously in Monte Carlo (or None)
    long_range -> True if the term couples all sites (its field is refreshed once per Monte Carlo group)

Terms
  CubicAnisotropy   E_i = Kc (a1^2 a2^2 + a2^2 a3^2 + a3^2 a1^2), a_k = e_i . c_k   (Kc > 0: easy <100>)
  UniaxialHigher    E_i = -K2 (e_i.u)^4 - K3 (e_i.u)^6
  Quadruplets       E = -sum_q K_q [(e_i.e_j)(e_k.e_l) + (e_j.e_k)(e_l.e_i) - (e_i.e_k)(e_j.e_l)]  (generic list)
  Dipolar           E = -(1/2) sum_i mu_i mu_B e_i . B_dip,i,  B_dip,i = (mu0 mu_B / 4 pi) sum_j mu_j
                    [3 (e_j.r)r - e_j] / r^3; FFT convolution on the supercell grid per sublattice pair,
                    open boundaries by zero padding, periodic ones by explicit image sums
"""
from __future__ import annotations

import itertools

import numpy as np

MU_B = 5.7883818060e-2          # meV/T
DIP = 0.927400968               # mu0 mu_B / 4 pi  in  T A^3


class _Term:
    long_range = False

    def bonds(self):
        return None


class CubicAnisotropy(_Term):
    def __init__(self, Kc, axes=None):
        self.Kc = np.asarray(Kc, float)                    # (N,)
        self.C = np.eye(3) if axes is None else np.asarray(axes, float)

    def _a(self, e):
        return e @ self.C.T                                 # (N,3) direction cosines

    def energy(self, e):
        a2 = self._a(e) ** 2
        return float(np.sum(self.Kc * (a2[:, 0] * a2[:, 1] + a2[:, 1] * a2[:, 2] + a2[:, 2] * a2[:, 0])))

    def gradient(self, e):
        a = self._a(e)
        a2 = a**2
        ga = 2 * a * np.stack([a2[:, 1] + a2[:, 2], a2[:, 0] + a2[:, 2], a2[:, 0] + a2[:, 1]], 1)
        return self.Kc[:, None] * (ga @ self.C)

    def local_energy(self, idx, en, e):
        a2 = (en @ self.C.T) ** 2
        return self.Kc[idx] * (a2[:, 0] * a2[:, 1] + a2[:, 1] * a2[:, 2] + a2[:, 2] * a2[:, 0])


class UniaxialHigher(_Term):
    def __init__(self, K2, K3=0.0, axis=(0, 0, 1)):
        self.K2 = np.asarray(K2, float)
        self.K3 = np.asarray(K3, float) * np.ones_like(self.K2)
        u = np.asarray(axis, float)
        self.u = u / np.linalg.norm(u)

    def energy(self, e):
        c = e @ self.u
        return float(np.sum(-self.K2 * c**4 - self.K3 * c**6))

    def gradient(self, e):
        c = e @ self.u
        return (-4 * self.K2 * c**3 - 6 * self.K3 * c**5)[:, None] * self.u[None, :]

    def local_energy(self, idx, en, e):
        c = en @ self.u
        return -self.K2[idx] * c**4 - self.K3[idx] * c**6


class Quadruplets(_Term):
    def __init__(self, quads, K, N):
        self.Q = np.asarray(quads, int).reshape(-1, 4)
        self.K = np.asarray(K, float) * np.ones(len(self.Q))
        lists = [[] for _ in range(N)]
        for p, q in enumerate(self.Q):
            for r, s in enumerate(q):
                lists[s].append((p, r))
        mx = max([len(x) for x in lists] + [1])
        self.pid = -np.ones((N, mx), int)
        self.role = np.zeros((N, mx), int)
        for s, x in enumerate(lists):
            for c, (p, r) in enumerate(x):
                self.pid[s, c], self.role[s, c] = p, r

    @staticmethod
    def _f(a, b, c, d):
        dot = lambda x, y: np.einsum("...a,...a->...", x, y)
        return dot(a, b) * dot(c, d) + dot(b, c) * dot(d, a) - dot(a, c) * dot(b, d)

    def energy(self, e):
        Q = self.Q
        return float(-np.sum(self.K * self._f(e[Q[:, 0]], e[Q[:, 1]], e[Q[:, 2]], e[Q[:, 3]])))

    def gradient(self, e):
        Q = self.Q
        a, b, c, d = (e[Q[:, k]] for k in range(4))
        dot = lambda x, y: np.einsum("ia,ia->i", x, y)[:, None]
        K = -self.K[:, None]
        g = np.zeros_like(e)
        np.add.at(g, Q[:, 0], K * (dot(c, d) * b + dot(b, c) * d - dot(b, d) * c))
        np.add.at(g, Q[:, 1], K * (dot(c, d) * a + dot(d, a) * c - dot(a, c) * d))
        np.add.at(g, Q[:, 2], K * (dot(a, b) * d + dot(d, a) * b - dot(b, d) * a))
        np.add.at(g, Q[:, 3], K * (dot(a, b) * c + dot(b, c) * a - dot(a, c) * b))
        return g

    def local_energy(self, idx, en, e):
        pid, role = self.pid[idx], self.role[idx]
        valid = pid >= 0
        P = self.Q[np.where(valid, pid, 0)]
        S = e[P]
        for r in range(4):
            m = valid & (role == r)
            S[m, r] = np.broadcast_to(en[:, None, :], S.shape[:2] + (3,))[m]
        f = self._f(S[..., 0, :], S[..., 1, :], S[..., 2, :], S[..., 3, :])
        K = self.K[np.where(valid, pid, 0)]
        return -np.sum(np.where(valid, K * f, 0.0), axis=1)

    def bonds(self):
        Q = self.Q
        return np.concatenate([Q[:, [0, 1]], Q[:, [1, 2]], Q[:, [2, 3]], Q[:, [3, 0]], Q[:, [0, 2]], Q[:, [1, 3]]])


class Dipolar(_Term):
    """Exact dipole-dipole interaction by FFT on the supercell grid (per sublattice pair)."""
    long_range = True

    def __init__(self, system, images=4, rcut_images=None):
        self.sys = system
        L = np.array(system.L)
        ns = system.uc.nsites
        self.ns = ns
        self.L = L
        self.pbc = system.pbc
        self.fshape = tuple(int(L[a]) if system.pbc[a] else (2 * int(L[a]) if L[a] > 1 else 1) for a in range(3))
        cell = system.uc.cell
        basis = system.uc.positions
        offs = []
        for a in range(3):
            M = self.fshape[a]
            k = np.arange(M)
            offs.append(np.where(k <= M // 2, k, k - M) if system.pbc[a] else np.where(k < L[a], k, k - M))
        O = np.stack(np.meshgrid(*offs, indexing="ij"), -1).reshape(-1, 3)
        imgs = [range(-images, images + 1) if system.pbc[a] else [0] for a in range(3)]
        self.K = {}
        for s, t in itertools.product(range(ns), range(ns)):
            W = np.zeros((len(O), 3, 3))
            for p in itertools.product(*imgs):
                R = (O + np.array(p) * L) @ cell + basis[t] - basis[s]
                r = np.linalg.norm(R, axis=1)
                ok = r > 1e-8
                rh = np.zeros_like(R)
                rh[ok] = R[ok] / r[ok, None]
                inv3 = np.zeros_like(r)
                inv3[ok] = 1.0 / r[ok] ** 3
                W += DIP * inv3[:, None, None] * (3 * rh[:, :, None] * rh[:, None, :] - np.eye(3)[None])
            W = W.reshape(self.fshape + (3, 3))
            for a in range(3):          # unused Nyquist offset of open axes
                if not system.pbc[a] and self.fshape[a] > 1:
                    sl = [slice(None)] * 5
                    sl[a] = int(L[a])
                    W[tuple(sl)] = 0.0
            self.K[(s, t)] = np.fft.rfftn(W, axes=(0, 1, 2))
        # full-grid bookkeeping (supports carved systems)
        self.grid = getattr(system, "grid_index", np.arange(system.N))
        self.Nfull = int(np.prod(L)) * ns
        self._cache = None

    def field(self, e):
        """B_dip (T), shape (N,3)."""
        ns, L = self.ns, self.L
        full = np.zeros((self.Nfull, 3))
        full[self.grid] = self.sys.mu[:, None] * e
        M = full.reshape(tuple(L) + (ns, 3))
        Mk = [np.fft.rfftn(M[..., t, :], s=self.fshape, axes=(0, 1, 2)) for t in range(ns)]
        B = np.zeros(tuple(L) + (ns, 3))
        for s in range(ns):
            acc = 0
            for t in range(ns):
                acc = acc + np.einsum("...ab,...b->...a", self.K[(s, t)], Mk[t])
            B[..., s, :] = np.fft.irfftn(acc, s=self.fshape, axes=(0, 1, 2))[:L[0], :L[1], :L[2]]
        return B.reshape(-1, 3)[self.grid]

    def energy(self, e):
        return float(-0.5 * MU_B * np.sum(self.sys.mu * np.einsum("ia,ia->i", e, self.field(e))))

    def gradient(self, e):
        return -MU_B * self.sys.mu[:, None] * self.field(e)

    def refresh(self, e):
        self._cache = self.field(e)

    def local_energy(self, idx, en, e):
        B = self._cache if self._cache is not None else self.field(e)
        return -MU_B * self.sys.mu[idx] * np.einsum("ia,ia->i", en, B[idx])
