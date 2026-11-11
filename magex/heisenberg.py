"""Map Green's-function energy Hessians onto the extended Heisenberg model

    E = sum_i e_i.A_i.e_i  -  sum_{i != j} e_i . J_ij . e_j          (TB2J convention)

unit spin vectors, every ordered pair counted (J_ji = J_ij^T), J > 0 ferromagnetic.
J_ij = J_iso 1 + J_ani(sym, traceless) + J_A(antisym);  DMI  D = (J_A[y,z], J_A[z,x], J_A[x,y])
so that   -e_i.J_A.e_j = -D.(e_i x e_j).

Equations used (rotation of exchange fields about Cartesian axes a,b, reference n_i):
    K_ab(i,j,R)        = -2 (e_a x n_i) . J_ij(R) . (e_b x n_j)                  (i,R) != (j,0)
    Hess_ab(i)         =  2 (e_a x n).A.(e_b x n) + 2 n.A.S_ab + c.S_ab           on site
    S_ab = 1/2 [e_a x (e_b x n) + e_b x (e_a x n)],   c = -2 sum_{(j,R)!=(i,0)} J_ij(R) n_j
plus an isotropic transverse offset delta_i (t_a.t_b) that absorbs the violation of
rotational invariance caused by off-site (non-rotated) spin-dependent hoppings.
All reference directions are fitted together by least squares; the residual measures
how much the bilinear parameters depend on the reference state.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

E3 = np.eye(3)


def min_image(nu, mesh, fi, fj, cell):
    """Lattice vector R == nu (mod mesh) minimising |tau_j + R - tau_i|; returns (R, dist, degeneracy)."""
    nu = np.array(nu)
    mesh = np.array(mesh)
    best, cand = None, []
    for m in itertools.product((-1, 0, 1), repeat=3):
        R = nu + mesh * np.array(m)
        d = np.linalg.norm((fj + R - fi) @ cell)
        cand.append((d, tuple(R)))
    cand.sort()
    dmin = cand[0][0]
    deg = sum(1 for d, _ in cand if abs(d - dmin) < 1e-6)
    return np.array(cand[0][1]), dmin, deg


def all_images(nu, mesh, fi, fj, cell, tol=1e-6):
    """All lattice vectors R == nu (mod mesh) at the minimal distance (supercell-boundary ties)."""
    nu, mesh = np.array(nu), np.array(mesh)
    cand = []
    for m in itertools.product((-2, -1, 0, 1, 2), repeat=3):
        R = nu + mesh * np.array(m)
        cand.append((np.linalg.norm((fj + R - fi) @ cell), tuple(int(x) for x in R)))
    dmin = min(c[0] for c in cand)
    return [R for d, R in cand if d - dmin < tol]


def decompose(J):
    iso = np.trace(J) / 3.0
    S = 0.5 * (J + J.T)
    A = 0.5 * (J - J.T)
    ani = S - iso * E3
    D = np.array([A[1, 2], A[2, 0], A[0, 1]])
    return iso, ani, D


@dataclass
class Pair:
    i: int            # site index (position in Results.sites)
    j: int
    R: tuple
    nu: tuple         # FFT index
    dist: float
    vec: np.ndarray
    J: np.ndarray     # full 3x3 tensor (meV)
    degeneracy: int = 1
    resid: float = 0.0

    @property
    def iso(self):
        return decompose(self.J)[0]

    @property
    def ani(self):
        return decompose(self.J)[1]

    @property
    def D(self):
        return decompose(self.J)[2]


@dataclass
class Results:
    sites: list
    symbols: list
    cell: np.ndarray
    positions: np.ndarray
    mesh: tuple
    moments: np.ndarray                  # (ns,3) from the z (or first) reference
    charges: np.ndarray
    refs: list                           # reference directions used
    pairs: list = field(default_factory=list)
    A: np.ndarray | None = None          # (ns,3,3) single-ion anisotropy (meV)
    full_tensor: bool = False
    extra: dict = field(default_factory=dict)

    def pair_lookup(self):
        return {(p.i, p.j, p.R): p for p in self.pairs}


def _signs(moments, d):
    s = np.sign(moments @ d)
    s[s == 0] = 1
    return s


def build_results(model, sites, refdata, refs, mag_mask, has_soc, units=1000.0):
    """refdata: list of compute_reference outputs (one per direction in refs).

    units: factor eV -> meV.
    """
    ns = len(sites)
    mesh = refdata[0]["mesh"]
    frac = model.frac_positions
    cell = model.cell
    signs = [_signs(rd["moments"], np.asarray(d, float)) for rd, d in zip(refdata, refs)]
    full = has_soc and len(refs) >= 3
    res = Results(sites=list(sites), symbols=[model.symbols[a] for a in sites], cell=cell,
                  positions=model.positions[list(sites)], mesh=mesh,
                  moments=refdata[-1]["moments"], charges=refdata[-1]["charges"],
                  refs=[tuple(map(float, d)) for d in refs], full_tensor=full)
    res.extra["moments_by_ref"] = [rd["moments"] for rd in refdata]

    # ---------------- pair tensors
    nus = list(itertools.product(*[range(n) for n in mesh]))
    for i in range(ns):
        for j in range(ns):
            rows, rhs_list = [], []
            for r, (rd, d) in enumerate(zip(refdata, refs)):
                d = np.asarray(d, float)
                ni = signs[r][i] * d
                nj = signs[r][j] * d
                for a in range(3):
                    ta = np.cross(E3[a], ni)
                    for b in range(3):
                        tb = np.cross(E3[b], nj)
                        coef = -2.0 * np.outer(ta, tb)
                        if np.allclose(coef, 0):
                            continue
                        rows.append(coef.ravel())
                        rhs_list.append(rd["K"][i, j][..., a, b].reshape(-1))
            Mrow = np.array(rows)
            rhs = np.array(rhs_list) * units           # (neq, nR)
            if full:
                sol, *_ = np.linalg.lstsq(Mrow, rhs, rcond=None)
                Jall = sol.T.reshape(-1, 3, 3)
                resid = np.linalg.norm(Mrow @ sol - rhs, axis=0) / np.sqrt(len(rows))
            else:
                ciso = Mrow @ E3.ravel()
                use = np.abs(ciso) > 1e-12
                jiso = (ciso[use] @ rhs[use]) / (ciso[use] @ ciso[use])
                Jall = jiso[:, None, None] * E3
                resid = np.linalg.norm(Mrow @ Jall.reshape(-1, 9).T - rhs, axis=0) / np.sqrt(len(rows))
            for k, nu in enumerate(nus):
                if i == j and nu == (0, 0, 0):
                    continue
                R, dist, deg = min_image(nu, mesh, frac[sites[i]], frac[sites[j]], cell)
                vec = (frac[sites[j]] + R - frac[sites[i]]) @ cell
                res.pairs.append(Pair(i, j, tuple(int(x) for x in R), nu, dist, vec,
                                      Jall[k], deg, float(resid[k])))

    # ---------------- single-ion anisotropy (needs SOC); fitted for every site
    basis = []
    for (p, q) in [(0, 0), (1, 1), (0, 1), (0, 2), (1, 2)]:
        B = np.zeros((3, 3))
        if p == q:
            B[p, p], B[2, 2] = 1.0, -1.0
        else:
            B[p, q] = B[q, p] = 1.0
        basis.append(B)
    A = np.zeros((ns, 3, 3))
    res.extra["sia_resid"] = np.zeros(ns)
    by_site = {}
    for p in res.pairs:
        by_site.setdefault(p.i, []).append(p)
    for i in range(ns):
        rows, rhs = [], []
        for r, (rd, d) in enumerate(zip(refdata, refs)):
            d = np.asarray(d, float)
            n = signs[r][i] * d
            c = np.zeros(3)
            for p in by_site.get(i, []):
                c += -2.0 * p.J @ (signs[r][p.j] * d)
            Hs = 0.5 * (rd["K"][i, i, 0, 0, 0] + rd["K"][i, i, 0, 0, 0].T) + rd["T1"][i]
            Hs = Hs * units
            for a in range(3):
                for b in range(a, 3):
                    ta, tb = np.cross(E3[a], n), np.cross(E3[b], n)
                    Sab = 0.5 * (np.cross(E3[a], np.cross(E3[b], n)) + np.cross(E3[b], np.cross(E3[a], n)))
                    coef = [2 * ta @ B @ tb + 2 * n @ B @ Sab for B in basis]
                    coef.append(ta @ tb)          # isotropic on-site offset (sum-rule violation)
                    if np.allclose(coef, 0) and abs(Hs[a, b] - c @ Sab) < 1e-12:
                        continue
                    rows.append(coef)
                    rhs.append(Hs[a, b] - c @ Sab)
        if rows:
            sol, *_ = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)
            A[i] = sum(s * B for s, B in zip(sol[:5], basis))
            res.extra.setdefault("onsite_offset", np.zeros(ns))[i] = sol[5]
            res.extra["sia_resid"][i] = float(np.linalg.norm(np.array(rows) @ sol - rhs) / np.sqrt(len(rows)))
    res.A = A if has_soc else np.zeros_like(A)
    res.extra["A_raw"] = A
    res.pairs.sort(key=lambda p: (p.dist, p.i, p.j))
    return res
