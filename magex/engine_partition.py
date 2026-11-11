"""Rotationally invariant rigid-spin rotation ("weighted" partition of the exchange field).

A Wannier Hamiltonian carries spin-dependent matrix elements X(R) not only on site but
also between atoms.  Rotating only the on-site block of atom i (TB2J / LKAG practice)
leaves the inter-atomic part un-rotated, so a uniform rotation of all sites is *not* a
global spin rotation and the Goldstone sum rule  sum_jR K_ij(R) + T_i = 0  fails.

Here the exchange field is partitioned among atoms,

    X = sum_i X^(i),     X^(i) = Pi_i X_L + X_L^+ Pi_i ,
    (X_L)_{mn}(R) = f_{a(m) a(n)} X_{mn}(R),   f_ab = w_a / (w_a + w_b),  w_a = ||p_a||,

so magnetic atoms carry the spin-dependent hopping to non-magnetic neighbours and
equal atoms share it half/half.  Rotating every X^(i) by the same angle is an exact global
rotation, so the sum rule holds identically (no SOC).  The perturbation for site i,

    W_ia = Pi_i Y_a + Y_a^+ Pi_i ,      Y_a = (e_a x X_L).sigma ,

is non-local; with  U_a(k) = Y_a(k)[sel,:] Psi(k)  every trace reduces to products
A diag(1/(z-E)) B^+  with A,B in {Psi_sel, U_x, U_y, U_z}:

    Tr[W_ia G W_jb G] = Tr[(Y_aG)_ij (Y_bG)_ji] + Tr[(Y_aGY_b^+)_ij G_ji]
                      + Tr[G_ij (Y_bGY_a^+)_ji] + Tr[(GY_b^+)_ij (GY_a^+)_ji]
"""
from __future__ import annotations

import time

import numpy as np

from . import parallel as par
from .engine import EPS, kmesh_frac, rodrigues
from .model import PAULI


def partition_weights(model, mode="weighted"):
    """Return f (norb, norb) atom-pair weights for the row atom (R-independent except same-atom)."""
    nat = model.natoms
    w = np.array([np.linalg.norm(model.onsite_exchange(a)) for a in range(nat)])
    fa = np.zeros((nat, nat))
    for a in range(nat):
        for b in range(nat):
            if a == b:
                fa[a, b] = 0.5
            elif mode == "weighted":
                s = w[a] + w[b]
                fa[a, b] = w[a] / s if s > 0 else 0.5
            else:                       # 'onsite': inter-atomic X is never rotated
                fa[a, b] = 0.0
    return fa[np.ix_(model.orb_atom, model.orb_atom)], w


def _spinor(q):
    """q (..., 3, n, n) spin-vector of orbital matrices -> (..., 2n, 2n)."""
    n = q.shape[-1]
    out = np.zeros(q.shape[:-3] + (2 * n, 2 * n), complex)
    out[..., :n, :n] = q[..., 2, :, :]
    out[..., n:, n:] = -q[..., 2, :, :]
    out[..., :n, n:] = q[..., 0, :, :] - 1j * q[..., 1, :, :]
    out[..., n:, :n] = q[..., 0, :, :] + 1j * q[..., 1, :, :]
    return out


def _cross(a, X):
    return np.einsum("ce,...emn->...cmn", EPS[:, a, :], X)


def compute_reference_partition(model, sites, mesh, contour, mode="weighted", lloyd=None,
                                timing=None):
    t0 = time.time()
    mesh = tuple(int(n) for n in mesh)
    nk = int(np.prod(mesh))
    kpts = kmesh_frac(mesh)
    N, n = model.nspinor, model.norb
    idx = [model.atom_spinor_index(a) for a in sites]
    sel = np.concatenate(idx)
    offs = np.cumsum([0] + [len(x) for x in idx])
    sl = [slice(offs[i], offs[i + 1]) for i in range(len(idx))]
    nsel, ns = len(sel), len(sites)

    f, w_atoms = partition_weights(model, mode)
    r0 = model.r0_index()
    XL = model.X * f[None, None]                                    # (nR,3,n,n)
    if mode == "onsite":
        mask = np.zeros_like(f)
        for a in range(model.natoms):
            o = model.atom_orbitals(a)
            mask[np.ix_(o, o)] = 1
        XL = np.zeros_like(model.X)
        XL[r0] = model.X[r0] * (0.5 * mask)[None]
    # perturbation generators on selected rows, all R
    Y = np.stack([_spinor(_cross(a, XL))[:, sel, :] for a in range(3)])        # (3,nR,nsel,N)
    pairs6 = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
    Z = np.stack([_spinor(0.5 * (_cross(a, _cross(b, XL)) + _cross(b, _cross(a, XL))))[:, sel, :]
                  for a, b in pairs6])                                           # (6,nR,nsel,N)
    # finite rotations for Lloyd
    ll_ops = {}
    if lloyd:
        for (i, j, nu) in lloyd["pairs"]:
            for t, th in enumerate(lloyd["angles"]):
                for s, sgn in ((i, -0.5), (j, 0.5)):
                    if (s, t, sgn) in ll_ops:
                        continue
                    rot = rodrigues(lloyd["axis"], sgn * th)
                    dX = np.einsum("ab,rbmn->ramn", rot, XL) - XL
                    ll_ops[(s, t, sgn)] = _spinor(dX)[:, model.atom_spinor_index(sites[s]), :]

    HR = model.spinor_HR()
    myk = par.split(nk)
    nl = len(myk)
    E_l = np.zeros((nl, N))
    Ps_l = np.zeros((nl, nsel, N), complex)
    U_l = np.zeros((nl, 3, nsel, N), complex)
    Q_l = np.zeros((nl, 6, nsel, N), complex)
    L_l = {key: np.zeros((nl,) + op.shape[1:2] + (N,), complex) for key, op in ll_ops.items()}
    for c0 in range(0, nl, 32):
        ks = myk[c0:c0 + 32]
        ph = np.exp(2j * np.pi * kpts[ks] @ model.R.T)                 # (nkc,nR)
        Hk = np.tensordot(ph, HR, axes=(1, 0))
        Hk = 0.5 * (Hk + np.conj(np.transpose(Hk, (0, 2, 1))))
        e, v = np.linalg.eigh(Hk)
        sl_k = slice(c0, c0 + len(ks))
        E_l[sl_k] = e
        Ps_l[sl_k] = v[:, sel, :]
        Yk = np.einsum("kr,armn->kamn", ph, Y)
        U_l[sl_k] = np.matmul(Yk, v[:, None])
        Zk = np.einsum("kr,armn->kamn", ph, Z)
        Q_l[sl_k] = np.matmul(Zk, v[:, None])
        for key, op in ll_ops.items():
            Lk = np.einsum("kr,rmn->kmn", ph, op)
            L_l[key][sl_k] = np.matmul(Lk, v)
    E = par.allgather_rows(E_l, myk, nk)
    Ps = par.allgather_rows(Ps_l, myk, nk)
    U = par.allgather_rows(U_l, myk, nk)
    Q = par.allgather_rows(Q_l, myk, nk)
    Lop = {key: par.allgather_rows(v, myk, nk) for key, v in L_l.items()}
    t_eig = time.time() - t0

    mu = contour.efermi
    if contour.kT > 0:
        focc = 1.0 / (np.exp(np.clip((E - mu) / contour.kT, -200, 200)) + 1.0)
    else:
        focc = (E <= mu).astype(float)
    rho = np.einsum("kmb,kb,knb->mn", Ps, focc, Ps.conj()) / nk
    moments = np.zeros((ns, 3))
    charges = np.zeros(ns)
    for i in range(ns):
        r = rho[sl[i], sl[i]]
        no = r.shape[0] // 2
        charges[i] = np.trace(r).real
        for c in range(3):
            moments[i, c] = np.trace(r @ np.kron(PAULI[c], np.eye(no))).real

    K = np.zeros((ns, ns) + mesh + (3, 3))
    T1 = np.zeros((ns, 3, 3))
    PsH = np.conj(np.swapaxes(Ps, -1, -2))
    UH = np.conj(np.swapaxes(U, -1, -2))                     # (nk,3,N,nsel)

    def to_R(Mk):
        return np.fft.fftn(Mk.reshape(mesh + Mk.shape[1:]), axes=(0, 1, 2)) / nk

    def neg(MR):
        return np.roll(np.flip(MR, axis=(0, 1, 2)), 1, axis=(0, 1, 2))

    phases = None
    ll_E = None
    if lloyd:
        angles = np.asarray(lloyd["angles"])
        ll_pairs = lloyd["pairs"]
        ll_E = np.zeros((len(ll_pairs), len(angles), 3))
        phases = {nu: np.exp(-2j * np.pi * kpts @ np.array(nu)) for (_, _, nu) in ll_pairs}

    for iz in par.split(len(contour)):
        z, wz = contour.z[iz], contour.w[iz]
        g = 1.0 / (z - E)                                      # (nk,N)
        Psg = Ps * g[:, None, :]
        Ug = U * g[:, None, None, :]
        G = to_R(np.matmul(Psg, PsH))                          # (mesh,nsel,nsel)
        YG = to_R(np.matmul(Ug, PsH[:, None]))
        GY = to_R(np.matmul(Psg[:, None], UH))
        YGY = to_R(np.matmul(Ug[:, :, None], UH[:, None]))
        Gm, YGm, GYm, YGYm = neg(G), neg(YG), neg(GY), neg(YGY)
        for i in range(ns):
            for j in range(ns):
                si, sj = sl[i], sl[j]
                F = (np.einsum("...amn,...bnm->...ab", YG[..., :, si, sj], YGm[..., :, sj, si])
                     + np.einsum("...abmn,...nm->...ab", YGY[..., :, :, si, sj], Gm[..., sj, si])
                     + np.einsum("...mn,...banm->...ab", G[..., si, sj], YGYm[..., :, :, sj, si])
                     + np.einsum("...bmn,...anm->...ab", GY[..., :, si, sj], GYm[..., :, sj, si]))
                K[i, j] += np.real(wz * F)
        # first-order (rigid) term: Tr_i[(Z G)_ii + (G Z^+)_ii] at R = 0
        ZG0 = np.einsum("kpmx,kx,kxn->pmn", Q, g, PsH) / nk     # (6,nsel,nsel)
        GZ0 = np.einsum("kmx,kx,kpnx->pmn", Ps, g, np.conj(Q)) / nk
        for i in range(ns):
            si = sl[i]
            v6 = np.trace(ZG0[:, si, si], axis1=1, axis2=2) + np.trace(GZ0[:, si, si], axis1=1, axis2=2)
            for p, (a, b) in enumerate(pairs6):
                T1[i, a, b] += np.real(wz * v6[p])
                if a != b:
                    T1[i, b, a] += np.real(wz * v6[p])
        if lloyd:
            for p, (i, j, nu) in enumerate(ll_pairs):
                ph_p = phases[nu]
                ii, jj = sl[i], sl[j]
                Pg = {"i": Ps[:, ii] * g[:, None, :], "j": Ps[:, jj] * g[:, None, :]}
                PH = {"i": np.conj(np.swapaxes(Ps[:, ii], -1, -2)), "j": np.conj(np.swapaxes(Ps[:, jj], -1, -2))}

                def red(M, s, t):
                    if s == t:
                        return M.mean(0)
                    ph = ph_p if s == "i" else np.conj(ph_p)
                    return np.einsum("k,kmn->mn", ph, M) / nk

                PP = {(s, t): red(np.matmul(Pg[s], PH[t]), s, t) for s in "ij" for t in "ij"}
                for t_ in range(len(angles)):
                    Lx = {"i": Lop[(i, t_, -0.5)], "j": Lop[(j, t_, 0.5)]}
                    LH = {s: np.conj(np.swapaxes(Lx[s], -1, -2)) for s in "ij"}
                    Lg = {s: Lx[s] * g[:, None, :] for s in "ij"}
                    AP = {(s, t): red(np.matmul(Lg[s], PH[t]), s, t) for s in "ij" for t in "ij"}
                    AA = {(s, t): red(np.matmul(Lg[s], LH[t]), s, t) for s in "ij" for t in "ij"}
                    PA = {(s, t): red(np.matmul(Pg[s], LH[t]), s, t) for s in "ij" for t in "ij"}

                    def mgl(ss):
                        Mx = np.block([[np.block([[AP[(s, t)], AA[(s, t)]], [PP[(s, t)], PA[(s, t)]]]) for t in ss]
                                       for s in ss])
                        return np.linalg.slogdet(np.eye(Mx.shape[0]) - Mx)[1]

                    ll_E[p, t_] += np.real(wz) * np.array([mgl("ij"), mgl("i"), mgl("j")])
    K = par.allreduce_sum(K)
    T1 = par.allreduce_sum(T1)
    nel = float(np.sum(focc)) / nk
    out = dict(K=K, T1=T1, moments=moments, charges=charges, orbital={}, nel=nel, sites=list(sites),
               mesh=mesh, partition=mode, atom_weights=w_atoms)
    if lloyd:
        ll_E = par.allreduce_sum(ll_E)
        out["lloyd"] = dict(pairs=ll_pairs, angles=np.asarray(lloyd["angles"]),
                            E=-(ll_E[:, :, 0] - ll_E[:, :, 1] - ll_E[:, :, 2]), E_single=-ll_E[:, :, 1:])
    if timing is not None:
        timing["eig"] = timing.get("eig", 0) + t_eig
        timing["z"] = timing.get("z", 0) + time.time() - t0 - t_eig
    return out
