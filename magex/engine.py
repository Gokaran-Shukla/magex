"""Green's-function engine: second-order energy response to local spin rotations.

For a reference model (time-odd spin field fixed along some direction) we compute,
for all spin sites i, j and all lattice vectors R of the k-mesh supercell,

  K_ab(i,j,R) = d^2 Omega / d phi_{i,a} d phi_{j,b}        (bubble part)
              = -(1/pi) Im \\int f(e) Tr[ W_ia G_ij(R) W_jb G_ji(-R) ] de

where phi_{i,a} rotates the exchange field p_i of site i about the Cartesian axis a,
W_ia = (e_a x p_i).sigma.  The on-site Hessian additionally contains the first-order
term  T_ab(i) = Tr[ rho_ii  (1/2){e_a x (e_b x p_i) + (a<->b)}.sigma ].

Omega is the grand potential at fixed chemical potential (magnetic force theorem).
Optionally, the exact finite-angle pair interaction energies are obtained with
Lloyd's formula  dOmega_S = (1/pi) Im \\int f ln det(1 - G_SS V_SS).
"""
from __future__ import annotations

import time

import numpy as np

from . import parallel as par
from .model import PAULI

EPS = np.zeros((3, 3, 3))
EPS[0, 1, 2] = EPS[1, 2, 0] = EPS[2, 0, 1] = 1
EPS[0, 2, 1] = EPS[2, 1, 0] = EPS[1, 0, 2] = -1


def rodrigues(axis, theta):
    u = np.asarray(axis, float) / np.linalg.norm(axis)
    ux = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return np.eye(3) + np.sin(theta) * ux + (1 - np.cos(theta)) * ux @ ux


def kmesh_frac(mesh):
    g = [np.arange(n) / n for n in mesh]
    return np.stack(np.meshgrid(*g, indexing="ij"), -1).reshape(-1, 3)


def vec_sigma(q):
    """sum_c q_c (x) sigma_c  in spin-block ordering; q (3, no, no) -> (2no, 2no)."""
    return sum(np.kron(PAULI[c], q[c]) for c in range(3))


def cross_e(a, p):
    """e_a x p for a vector of matrices p (3,no,no)."""
    return np.einsum("ce,emn->cmn", EPS[:, a, :], p)


class SiteOps:
    """Perturbation operators for one spin site in a given reference model."""

    def __init__(self, model, atom):
        self.atom = atom
        self.idx = model.atom_spinor_index(atom)
        self.p = model.onsite_exchange(atom)                       # (3,no,no)
        self.W = np.array([vec_sigma(cross_e(a, self.p)) for a in range(3)])
        D2 = np.zeros((3, 3) + self.W.shape[1:], complex)
        for a in range(3):
            for b in range(3):
                q = 0.5 * (cross_e(a, cross_e(b, self.p)) + cross_e(b, cross_e(a, self.p)))
                D2[a, b] = vec_sigma(q)
        self.D2 = D2
        self.P = vec_sigma(self.p)

    def V_rot(self, rot):
        """Finite perturbation: exchange field rotated by matrix rot, minus original."""
        pr = np.einsum("ab,bmn->amn", rot, self.p)
        return vec_sigma(pr - self.p)


def compute_reference(model, sites, mesh, contour, lloyd=None, orbital_pairs=None,
                      timing=None):
    """Run the GF machinery for one reference direction.

    sites         list of atom indices (magnetic + ligand spin sites)
    mesh          (N1,N2,N3) k-mesh = real-space supercell for G(R)
    contour       Contour
    lloyd         optional dict(pairs=[(si,sj,nu)], axis=u, angles=[...])
                  si,sj: positions in ``sites``; nu: FFT index tuple of R
    orbital_pairs optional list of (si,sj,nu) for orbital-resolved K
    """
    t0 = time.time()
    mesh = tuple(int(n) for n in mesh)
    nk = int(np.prod(mesh))
    kpts = kmesh_frac(mesh)
    ops = [SiteOps(model, a) for a in sites]
    sel = np.concatenate([o.idx for o in ops])
    offs = np.cumsum([0] + [len(o.idx) for o in ops])
    sl = [slice(offs[i], offs[i + 1]) for i in range(len(ops))]
    nsel = len(sel)
    ns = len(sites)

    # ---------------- eigen-decomposition, distributed over k
    HR = model.spinor_HR()
    myk = par.split(nk)
    E_loc = np.zeros((len(myk), model.nspinor))
    P_loc = np.zeros((len(myk), nsel, model.nspinor), complex)
    chunk = 64
    for c0 in range(0, len(myk), chunk):
        ks = myk[c0:c0 + chunk]
        Hk = model.Hk(kpts[ks], HR)
        Hk = 0.5 * (Hk + np.conj(np.transpose(Hk, (0, 2, 1))))
        e, v = np.linalg.eigh(Hk)
        E_loc[c0:c0 + len(ks)] = e
        P_loc[c0:c0 + len(ks)] = v[:, sel, :]
    E = par.allgather_rows(E_loc, myk, nk)
    Psi = par.allgather_rows(P_loc, myk, nk)
    t_eig = time.time() - t0

    # ---------------- ground-state densities (Fermi function at contour T)
    mu = contour.efermi
    if contour.kT > 0:
        x = np.clip((E - mu) / contour.kT, -200, 200)
        focc = 1.0 / (np.exp(x) + 1.0)
    else:
        focc = (E <= mu).astype(float)
    rho = np.einsum("kmb,kb,knb->mn", Psi, focc, Psi.conj()) / nk
    moments = np.zeros((ns, 3))
    charges = np.zeros(ns)
    for i, o in enumerate(ops):
        r = rho[sl[i], sl[i]]
        no = r.shape[0] // 2
        charges[i] = np.trace(r).real
        for c in range(3):
            moments[i, c] = np.trace(r @ np.kron(PAULI[c], np.eye(no))).real
    nel_total = float(np.sum(focc)) / nk

    # ---------------- energy integration, distributed over z
    K = np.zeros((ns, ns) + mesh + (3, 3))
    T1 = np.zeros((ns, 3, 3))
    orb = {}
    if orbital_pairs:
        for (i, j, nu) in orbital_pairs:
            orb[(i, j, nu)] = np.zeros((3, 3, len(ops[i].idx) // 2, len(ops[j].idx) // 2))
    ll_E = None
    if lloyd:
        angles = np.asarray(lloyd["angles"])
        ll_pairs = lloyd["pairs"]
        ll_E = np.zeros((len(ll_pairs), len(angles), 3))     # F_ij, F_i, F_j integrated
        Vs = {}
        for (i, j, nu) in ll_pairs:
            for t, th in enumerate(angles):
                Vs[(i, t)] = ops[i].V_rot(rodrigues(lloyd["axis"], -th / 2))
                Vs[(j, t, "j")] = ops[j].V_rot(rodrigues(lloyd["axis"], th / 2))

    myz = par.split(len(contour))
    PsiH = np.conj(np.transpose(Psi, (0, 2, 1)))
    for iz in myz:
        z, w = contour.z[iz], contour.w[iz]
        g = 1.0 / (z - E)
        Gk = np.matmul(Psi * g[:, None, :], PsiH)
        GR = np.fft.fftn(Gk.reshape(mesh + (nsel, nsel)), axes=(0, 1, 2)) / nk
        GRm = np.roll(np.flip(GR, axis=(0, 1, 2)), 1, axis=(0, 1, 2))
        for i in range(ns):
            Gii0 = GR[0, 0, 0][sl[i], sl[i]]
            T1[i] += np.real(w * np.einsum("abmn,nm->ab", ops[i].D2, Gii0))
            for j in range(ns):
                Gij = GR[..., sl[i], sl[j]]
                Gji = GRm[..., sl[j], sl[i]]
                A = np.einsum("amn,...nk->a...mk", ops[i].W, Gij)
                B = np.einsum("bmn,...nk->b...mk", ops[j].W, Gji)
                F = np.einsum("a...mk,b...km->...ab", A, B)
                K[i, j] += np.real(w * F)
        for (i, j, nu) in orb:
            A = np.einsum("amn,nk->amk", ops[i].W, GR[nu][sl[i], sl[j]])
            B = np.einsum("bmn,nk->bmk", ops[j].W, GRm[nu][sl[j], sl[i]])
            F = np.einsum("amk,bkm->abmk", A, B)        # (3,3,2ni,2nj)
            ni, nj = F.shape[2] // 2, F.shape[3] // 2
            F = F[:, :, :ni] + F[:, :, ni:]
            F = F[:, :, :, :nj] + F[:, :, :, nj:]
            orb[(i, j, nu)] += np.real(w * F)
        if lloyd:
            for p, (i, j, nu) in enumerate(ll_pairs):
                Gii, Gjj = GR[0, 0, 0][sl[i], sl[i]], GR[0, 0, 0][sl[j], sl[j]]
                Gij, Gji = GR[nu][sl[i], sl[j]], GRm[nu][sl[j], sl[i]]
                GSS = np.block([[Gii, Gij], [Gji, Gjj]])
                ni, nj = Gii.shape[0], Gjj.shape[0]
                for t in range(len(angles)):
                    Vi, Vj = Vs[(i, t)], Vs[(j, t, "j")]
                    VSS = np.zeros_like(GSS)
                    VSS[:ni, :ni], VSS[ni:, ni:] = Vi, Vj
                    fij = np.linalg.slogdet(np.eye(ni + nj) - GSS @ VSS)[1]
                    fi = np.linalg.slogdet(np.eye(ni) - Gii @ Vi)[1]
                    fj = np.linalg.slogdet(np.eye(nj) - Gjj @ Vj)[1]
                    ll_E[p, t] += np.real(w) * np.array([fij, fi, fj])
    K = par.allreduce_sum(K)
    T1 = par.allreduce_sum(T1)
    for key in orb:
        orb[key] = par.allreduce_sum(orb[key])
    out = dict(K=K, T1=T1, moments=moments, charges=charges, orbital=orb,
               nel=nel_total, sites=list(sites), mesh=mesh,
               p_norm=[float(np.linalg.norm(o.p)) for o in ops])
    if lloyd:
        ll_E = par.allreduce_sum(ll_E)
        # interaction energy  dOmega_ij - dOmega_i - dOmega_j ;  dOmega = (1/pi) Im int f lndet
        out["lloyd"] = dict(pairs=ll_pairs, angles=np.asarray(lloyd["angles"]),
                            E=-(ll_E[:, :, 0] - ll_E[:, :, 1] - ll_E[:, :, 2]),
                            E_single=-ll_E[:, :, 1:])
    if timing is not None:
        timing["eig"] = timing.get("eig", 0) + t_eig
        timing["z"] = timing.get("z", 0) + time.time() - t0 - t_eig
    return out
