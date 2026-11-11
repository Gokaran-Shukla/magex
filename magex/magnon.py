"""Linear spin-wave theory for collinear (or general) ordered states with full exchange
tensors, DMI and single-ion anisotropy.

Model:  E = sum_i e_i.A_i.e_i - sum_{i!=j} e_i.J_ij.e_j ,  spins S_i = |M_i|/2 (g = 2).
Holstein-Primakoff in local frames (u_i = x_i + i y_i, n_i = local z):
    A_ij(k) = sqrt(S_i S_j) u_i^T Jt_ij(k) conj(u_j) - delta_ij 2 sum_l S_l n_i.Jt_il(0).n_l
    B_ij(k) = sqrt(S_i S_j) u_i^T Jt_ij(k) u_j
with Jt_ij(R) = -J_ij(R)/(S_i S_j),  Jt_ii(0) = A_i/S_i^2,  Jt(k) = sum_R Jt(R) e^{2 pi i k.R}.
    H = 1/2 sum_k X_k^+ h(k) X_k ,  h = [[A(k), B(k)], [B(k)^+, A(-k)^T]]
Energies: positive eigenvalues of g.h (paraunitary / Colpa).  For a single FM sublattice
this gives  w(k) = (4/M) [J(0) - J(k)]  in the paper's convention.
"""
from __future__ import annotations

import numpy as np

from .model import rotation_to


class LSWT:
    def __init__(self, spins, directions, couplings, aniso=None):
        """spins: (n,) S_i;  directions: (n,3) unit n_i;  couplings: list of (i, j, R, J3x3 meV)
        for every ordered pair (both i->j and j->i present);  aniso: (n,3,3) A_i in meV."""
        self.S = np.asarray(spins, float)
        self.n = np.array([d / np.linalg.norm(d) for d in directions])
        self.ns = len(self.S)
        self.cpl = couplings
        self.Aion = np.zeros((self.ns, 3, 3)) if aniso is None else np.asarray(aniso)
        self.u = []
        for d in self.n:
            Rm = rotation_to(d)
            self.u.append(Rm[:, 0] + 1j * Rm[:, 1])
        self.u = np.array(self.u)
        # Jt(0) contraction for the Ising-like diagonal term
        self.diag = np.zeros(self.ns)
        for (i, j, R, J) in self.cpl:
            Jt = -J / (self.S[i] * self.S[j])
            self.diag[i] += self.S[j] * self.n[i] @ Jt @ self.n[j]
        for i in range(self.ns):
            Jt = self.Aion[i] / self.S[i] ** 2
            self.diag[i] += self.S[i] * self.n[i] @ Jt @ self.n[i]

    def Jt(self, k):
        out = np.zeros((self.ns, self.ns, 3, 3), complex)
        for (i, j, R, J) in self.cpl:
            out[i, j] += -J / (self.S[i] * self.S[j]) * np.exp(2j * np.pi * np.dot(k, R))
        for i in range(self.ns):
            out[i, i] += self.Aion[i] / self.S[i] ** 2
        return out

    def hmat(self, k):
        k = np.asarray(k, float)
        Jk, Jmk = self.Jt(k), self.Jt(-k)
        sq = np.sqrt(np.outer(self.S, self.S))
        u, uc = self.u, self.u.conj()
        A = sq * np.einsum("ia,ijab,jb->ij", u, Jk, uc) - np.diag(2 * self.diag)
        Am = sq * np.einsum("ia,ijab,jb->ij", u, Jmk, uc) - np.diag(2 * self.diag)
        B = sq * np.einsum("ia,ijab,jb->ij", u, Jk, u)
        return np.block([[A, B], [B.conj().T, Am.T]])

    def energies(self, k):
        h = self.hmat(k)
        g = np.diag(np.r_[np.ones(self.ns), -np.ones(self.ns)])
        w = np.linalg.eigvals(g @ h)
        if np.max(np.abs(w.imag)) > 1e-6 * max(1.0, np.max(np.abs(w.real))):
            self.unstable = True
        w = np.sort(w.real)
        return w[self.ns:]            # positive branch (meV)

    def band(self, kpath):
        return np.array([self.energies(k) for k in kpath])

    def dos(self, mesh, emax=None, nbins=400, sigma=None):
        g = [np.arange(n) / n for n in mesh]
        ks = np.stack(np.meshgrid(*g, indexing="ij"), -1).reshape(-1, 3)
        W = np.concatenate([self.energies(k) for k in ks])
        emax = emax or 1.1 * W.max()
        e = np.linspace(0, emax, nbins)
        sigma = sigma or emax / 100
        dos = np.exp(-((e[:, None] - W[None]) ** 2) / (2 * sigma**2)).sum(1) / (len(ks) * sigma * np.sqrt(2 * np.pi))
        return e, dos


def kpath_from_labels(labels, special, npts=60):
    pts = [np.array(special[l]) for l in labels]
    path, ticks = [], [0]
    for a, b in zip(pts[:-1], pts[1:]):
        seg = [a + (b - a) * t for t in np.linspace(0, 1, npts, endpoint=False)]
        path += seg
        ticks.append(len(path))
    path.append(pts[-1])
    ticks[-1] = len(path) - 1
    return np.array(path), ticks
