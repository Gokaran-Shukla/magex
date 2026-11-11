"""Beyond the plain magnetic force theorem (isotropic, collinear reference).

Notation (q-space matrices over spin sites, transverse channel, raw rotation angles):
    Kb(q)  bubble  -(1/pi) Im int f Tr[W G W G]          (incl. on-site bubble)
    D      diag first-order term  T_i = Tr[rho_ii d^2H_i]
    rotational invariance (no SOC):  sum_{j,R} Kb_ij(R) + D_i = 0

  MFT (Liechtenstein):      K       = Kb + D
  Bruno-renormalised MFT:   K~      = -D Kb^{-1} D - D            [PRL 90, 087205 (2003)]
      (energy as a function of *moment* directions; for one FM sublattice
       w~(q) = w(q) / (1 - w(q)/Delta), the spin-wave energy is increased where it is
       comparable to the exchange splitting Delta).
  Ligand downfolding:       K_eff   = K_MM - K_ML K_LL^{-1} K_LM  (ligand angles relaxed)

Real-space isotropic J_ij(R) = -K_ij(R) / (2 s_i s_j).
With rigid="sumrule" (default) D_i = -sum_jR Kb_ij(R), which enforces the Goldstone mode
(K(q=0) has the global-rotation zero eigenvector) even when off-site exchange fields exist.
"""
from __future__ import annotations

import numpy as np


def _transverse_axes(d):
    d = np.asarray(d, float) / np.linalg.norm(d)
    a = np.array([1.0, 0, 0]) if abs(d[0]) < 0.9 else np.array([0, 1.0, 0])
    u1 = np.cross(d, a)
    u1 /= np.linalg.norm(u1)
    u2 = np.cross(d, u1)
    return u1, u2


def scalar_channels(rd, d):
    u1, u2 = _transverse_axes(d)
    kb = 0.5 * (np.einsum("a,...ab,b->...", u1, rd["K"], u1) + np.einsum("a,...ab,b->...", u2, rd["K"], u2))
    D = 0.5 * (np.einsum("a,iab,b->i", u1, rd["T1"], u1) + np.einsum("a,iab,b->i", u2, rd["T1"], u2))
    return kb, D       # kb: (ns,ns,N1,N2,N3)


def renormalize(rd, d, mag, lig=(), bruno=False, downfold=False, units=1000.0, rcond=1e-8,
                rigid="sumrule"):
    """Return dict with J (ns_mag, ns_mag, N1,N2,N3) in meV plus diagnostics.

    mag, lig: positions (in the GF site list) of magnetic and ligand sites.
    """
    kb, D = scalar_channels(rd, d)
    mesh = kb.shape[2:]
    nk = int(np.prod(mesh))
    ns = kb.shape[0]
    s = np.sign(rd["moments"] @ (np.asarray(d, float) / np.linalg.norm(d)))
    s[s == 0] = 1
    sumrule = kb.sum(axis=(1, 2, 3, 4)) + D
    if rigid == "sumrule":
        # Off-site spin-dependent hoppings are not rotated by a local on-site rotation, which
        # breaks the Goldstone sum rule.  Restore it by defining the rigid term from the bubble.
        D = -kb.sum(axis=(1, 2, 3, 4))
    Kbq = np.fft.ifftn(kb, axes=(2, 3, 4)) * nk          # sum_R k(R) e^{+2 pi i q.R}
    Kbq = np.moveaxis(Kbq.reshape(ns, ns, nk), 2, 0)     # (nq, ns, ns)
    Kbq = 0.5 * (Kbq + np.conj(np.transpose(Kbq, (0, 2, 1))))
    keep = list(mag)
    if bruno:
        # Bruno renormalisation within the magnetic subspace: weakly polarised ligands make the
        # moment-direction parametrisation ill-defined, so they are excluded and the Goldstone
        # mode is enforced inside the magnetic block (D_i = -sum_{j in mag, R} Kb_ij(R)).
        M = list(mag)
        Kb_m = Kbq[:, M][:, :, M]
        Dmag = -kb[np.ix_(M, M)].sum(axis=(1, 2, 3, 4))
        Dm = np.diag(Dmag)
        Kq = np.array([-Dm @ np.linalg.pinv(k, rcond=rcond, hermitian=True) @ Dm - Dm for k in Kb_m])
    else:
        Kq = Kbq + np.diag(D)[None]
        if downfold and len(lig):
            M, L = list(mag), list(lig)
            out = []
            for k in Kq:
                KLL = k[np.ix_(L, L)]
                out.append(k[np.ix_(M, M)] - k[np.ix_(M, L)] @ np.linalg.pinv(KLL, rcond=rcond, hermitian=True)
                           @ k[np.ix_(L, M)])
            Kq = np.array(out)
        else:
            Kq = Kq[:, keep][:, :, keep]
    nm = len(keep)
    Kr = np.moveaxis(Kq, 0, 2).reshape((nm, nm) + tuple(mesh))
    Kr = np.fft.fftn(Kr, axes=(2, 3, 4)).real / nk
    sm = s[list(mag)]
    J = -Kr / (2.0 * sm[:, None, None, None, None] * sm[None, :, None, None, None]) * units
    return dict(J=J, sumrule=sumrule * units, D=D * units, signs=sm, Kq=Kq * units)
