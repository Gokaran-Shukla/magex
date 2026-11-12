"""Classical atomistic spin model built from a magex ``results.json``.

Energy (magex/TB2J convention, meV; unit spin vectors e_i; every ordered pair counted):

    E = sum_i e_i.A_i.e_i  -  sum_{i != j} e_i.J_ij.e_j  -  mu_B sum_i mu_i e_i.B

with J_ij the full 3x3 tensor (isotropic + symmetric anisotropic + DMI), A_i the single-ion
anisotropy, mu_i the moment in Bohr magnetons and B the external field in tesla.  Because both
(i,j) and (j,i) appear, the local exchange field of site i is h_i = sum_j J_ij e_j and

    dE/de_i = 2 A_i e_i - 2 h_i - mu_B mu_i B.

The supercell couplings are stored as one sparse (3N x 3N) matrix, so energies and fields of
any spin configuration are a single sparse mat-vec.  If the supercell is so small that a
coupling connects a site with its own periodic image, that term is single-site and is folded
into the anisotropy (A_i <- A_i - sym(J)), which is exact for the periodic supercell.
"""
from __future__ import annotations

import itertools
import json
import os
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

MU_B = 5.7883818060e-2      # meV / T
K_B = 8.617333262e-2        # meV / K
GAMMA = 0.1760859630        # rad / (ps T)   (electron gyromagnetic ratio, g = 2.0023)


# ----------------------------------------------------------------------------- loading
@dataclass
class UnitCellModel:
    """Magnetic sites of one unit cell and their couplings (meV)."""
    cell: np.ndarray                 # (3,3) rows = lattice vectors (A)
    positions: np.ndarray            # (ns,3) cartesian (A)
    symbols: list
    moments: np.ndarray              # (ns,) |M| in mu_B
    ref_dirs: np.ndarray             # (ns,3) reference (ground-state) directions
    signs: np.ndarray                # (ns,) +-1 relative to the reference axis
    couplings: list                  # [(i, j, R(3,), J(3,3))] every ordered pair, i != j or R != 0
    A: np.ndarray                    # (ns,3,3)
    info: dict = field(default_factory=dict)

    @property
    def nsites(self):
        return len(self.moments)


def _all_images(nu, mesh, fi, fj, cell, tol=1e-6):
    nu, mesh = np.array(nu), np.array(mesh)
    cand = []
    for m in itertools.product((-2, -1, 0, 1, 2), repeat=3):
        R = nu + mesh * np.array(m)
        cand.append((np.linalg.norm((fj + R - fi) @ cell), tuple(int(x) for x in R)))
    dmin = min(c[0] for c in cand)
    return [R for d, R in cand if d - dmin < tol], dmin


def load_results(path, J="mft", anisotropic=True, sia=True, rcut=None, jtol=1e-4,
                 moments=None, axis=None):
    """Read a magex results.json and return a UnitCellModel of the magnetic sites.

    J            isotropic part from "mft", "bruno" or "downfold" (anisotropic part and DMI always
                 from the tensor fit)
    anisotropic  keep J_ani and DMI (False: isotropic Heisenberg)
    sia          keep single-ion anisotropy
    rcut         keep pairs with distance <= rcut (A); None -> 2 x nearest-neighbour distance
    jtol         drop couplings with max |J_ab| < jtol (meV)
    moments      override |M| per magnetic site (mu_B)
    axis         reference axis for the ordered state (default: last magex reference)
    """
    if str(path).endswith(".gz"):
        import gzip
        with gzip.open(path, "rt") as f:
            d = json.load(f)
    else:
        with open(path) as f:
            d = json.load(f)
    mp = list(d.get("mag_pos") or range(len(d["sites"])))
    cell = np.array(d["cell"], float)
    pos_all = np.array(d["positions"], float)
    frac_all = pos_all @ np.linalg.inv(cell)
    M = np.array(d["moments"], float)
    ax = np.array(axis if axis is not None else d["refs"][-1], float)
    ax /= np.linalg.norm(ax)
    s = np.sign(M[mp] @ ax)
    s[s == 0] = 1
    mom = np.linalg.norm(M[mp], axis=1) if moments is None else np.asarray(moments, float)
    corr = None
    if J != "mft":
        if J not in (d.get("corrections") or {}):
            raise ValueError(f"results.json has no '{J}' couplings; available: {list(d['corrections'])}")
        corr = np.array(d["corrections"][J])
    raw = []
    for p in d["pairs"]:
        if p["i"] not in mp or p["j"] not in mp:
            continue
        Jt = np.array(p["J"], float)
        if corr is not None:
            iso = np.trace(Jt) / 3
            Jt = Jt - iso * np.eye(3) + corr[mp.index(p["i"]), mp.index(p["j"])][tuple(p["nu"])] * np.eye(3)
        if not anisotropic:
            Jt = np.trace(Jt) / 3 * np.eye(3)
        imgs, dist = _all_images(p["nu"], d["mesh"], frac_all[p["i"]], frac_all[p["j"]], cell)
        raw.append((mp.index(p["i"]), mp.index(p["j"]), imgs, dist, Jt))
    dmin = min(r[3] for r in raw)
    if rcut is None:
        rcut = 2.0 * dmin
    couplings, J0_all, J0_kept, shells = [], 0.0, 0.0, set()
    for i, j, imgs, dist, Jt in raw:
        w = Jt / len(imgs)
        J0_all += s[i] * s[j] * np.trace(Jt) / 3
        if dist > rcut + 1e-6 or np.max(np.abs(Jt)) < jtol:
            continue
        J0_kept += s[i] * s[j] * np.trace(Jt) / 3
        shells.add(round(dist, 3))
        for R in imgs:
            couplings.append((i, j, np.array(R, int), w))
    A = np.array(d["A"], float)[mp] if sia else np.zeros((len(mp), 3, 3))
    info = dict(source=os.path.abspath(path), J=J, rcut=float(rcut), nn_distance=float(dmin),
                shells=[float(x) for x in sorted(shells)], ncouplings_per_cell=len(couplings),
                J0_fraction=float(J0_kept / J0_all) if J0_all else 1.0, axis=ax.tolist())
    return UnitCellModel(cell, pos_all[mp], [d["symbols"][k] for k in mp], mom,
                         s[:, None] * ax[None, :], s, couplings, A, info)


def mean_field_tc(uc: UnitCellModel):
    """Mean-field ordering temperature (K) for the collinear reference state.

    k_B T_MF = (2/3) lambda_max( s_a s_b sum_R J^iso_ab(R) ), the largest eigenvalue of the
    sign-gauged J(q=0) matrix over the sublattices (classical spins, this convention).
    """
    ns = uc.nsites
    J0 = np.zeros((ns, ns))
    for i, j, R, Jt in uc.couplings:
        J0[i, j] += uc.signs[i] * uc.signs[j] * np.trace(Jt) / 3
    J0 = 0.5 * (J0 + J0.T)
    return float(2.0 / 3.0 * np.max(np.linalg.eigvalsh(J0)) / K_B)


def jq_stability(uc: UnitCellModel, nq=16):
    """Is the collinear reference the classical ground state of the (truncated) isotropic model?

    Returns (lambda(q=0), max_q lambda(q), q_max) where lambda is the largest eigenvalue of the
    sign-gauged J(q) matrix (meV).  max > lambda(0) means a spiral (or other order) has lower
    energy than the reference, e.g. because the coupling cutoff truncated long-range exchange.
    """
    ns = uc.nsites
    g = np.arange(nq) / nq
    Q = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    Jq = np.zeros((len(Q), ns, ns), complex)
    for i, j, R, Jt in uc.couplings:
        Jq[:, i, j] += uc.signs[i] * uc.signs[j] * np.trace(Jt) / 3 * np.exp(2j * np.pi * Q @ R)
    Jq = 0.5 * (Jq + np.conj(np.transpose(Jq, (0, 2, 1))))
    lam = np.linalg.eigvalsh(Jq)[:, -1]
    k = int(np.argmax(lam))
    return float(lam[0]), float(lam[k]), Q[k]


# ----------------------------------------------------------------------------- supercell
class SpinSystem:
    """Supercell of the magnetic unit cell with sparse couplings."""

    def __init__(self, uc: UnitCellModel, supercell=(10, 10, 10), pbc=(True, True, True),
                 field=(0.0, 0.0, 0.0)):
        self.uc = uc
        self.L = tuple(int(x) for x in supercell)
        self.pbc = tuple(bool(x) for x in pbc)
        ns = uc.nsites
        cells = np.array(list(itertools.product(*[range(n) for n in self.L])), int)
        self.ncell = len(cells)
        self.N = self.ncell * ns
        Lv = np.array(self.L)
        cidx = lambda c: (c[:, 0] * Lv[1] + c[:, 1]) * Lv[2] + c[:, 2]
        # site index = cell_index * ns + sublattice
        self.sublattice = np.tile(np.arange(ns), self.ncell)
        self.cell_index = np.repeat(np.arange(self.ncell), ns)
        self.cells = cells
        self.positions = (cells[self.cell_index] @ uc.cell) + uc.positions[self.sublattice]
        self.mu = uc.moments[self.sublattice].astype(float)
        self.signs = uc.signs[self.sublattice].astype(float)
        self.ref = uc.ref_dirs[self.sublattice].copy()
        A = uc.A[self.sublattice].copy()
        rows, cols, vals = [], [], []
        self.self_folded = 0
        self.bonds = []          # (src, dst) for graph colouring / visualisation
        for (i, j, R, Jt) in uc.couplings:
            tgt = cells + R[None, :]
            ok = np.ones(self.ncell, bool)
            for a in range(3):
                if self.pbc[a]:
                    tgt[:, a] %= Lv[a]
                else:
                    ok &= (tgt[:, a] >= 0) & (tgt[:, a] < Lv[a])
            src = cidx(cells[ok]) * ns + i
            dst = cidx(tgt[ok]) * ns + j
            same = src == dst
            if np.any(same):
                # coupling to own periodic image: single-site term -e.J.e
                A[src[same]] -= 0.5 * (Jt + Jt.T)
                self.self_folded += int(np.sum(same))
                src, dst = src[~same], dst[~same]
            for a in range(3):
                for b in range(3):
                    if Jt[a, b] != 0.0:
                        rows.append(3 * src + a)
                        cols.append(3 * dst + b)
                        vals.append(np.full(len(src), Jt[a, b]))
            self.bonds.append(np.stack([src, dst], 1))
        rows = np.concatenate(rows) if rows else np.zeros(0, int)
        cols = np.concatenate(cols) if cols else np.zeros(0, int)
        vals = np.concatenate(vals) if vals else np.zeros(0)
        self.Jsp = sp.csr_matrix((vals, (rows, cols)), shape=(3 * self.N, 3 * self.N))
        self.Jsp.sum_duplicates()
        self.A = A
        self.bonds = np.concatenate(self.bonds) if self.bonds else np.zeros((0, 2), int)
        self.has_A = bool(np.max(np.abs(A)) > 0)
        self.set_field(field)

    # -------------------------------------------------------------- field / energy
    def set_field(self, B):
        self.B = np.asarray(B, float).reshape(3)

    def exchange_field(self, e):
        """h_i = sum_j J_ij e_j  (meV), shape (N,3)."""
        return (self.Jsp @ e.reshape(-1)).reshape(-1, 3)

    def gradient(self, e):
        """dE/de_i (meV), shape (N,3)."""
        g = -2.0 * self.exchange_field(e) - MU_B * self.mu[:, None] * self.B[None, :]
        if self.has_A:
            g += 2.0 * np.einsum("iab,ib->ia", self.A, e)
        return g

    def effective_field(self, e):
        """B_eff,i = -(1/(mu_B mu_i)) dE/de_i  (tesla)."""
        return -self.gradient(e) / (MU_B * self.mu[:, None])

    def energy(self, e, per_spin=False):
        """Total energy (meV) of configuration e (N,3)."""
        E = -np.sum(e * self.exchange_field(e)) - MU_B * np.sum(self.mu * (e @ self.B))
        if self.has_A:
            E += np.einsum("ia,iab,ib->", e, self.A, e)
        return E / self.N if per_spin else E

    # -------------------------------------------------------------- configurations
    def init_spins(self, kind="reference", axis=(0, 0, 1), seed=None, cone=0.0, path=None):
        rng = np.random.default_rng(seed)
        if kind == "random":
            e = rng.normal(size=(self.N, 3))
        elif kind == "ferro":
            e = np.tile(np.asarray(axis, float), (self.N, 1))
        elif kind == "reference":
            e = self.ref.copy()
        elif kind == "reference_axis":
            ax = np.asarray(axis, float) / np.linalg.norm(axis)
            e = self.signs[:, None] * ax[None, :]
        elif kind == "file":
            e = np.load(path)["spins"] if str(path).endswith(".npz") else np.loadtxt(path, delimiter=",")[:, -3:]
            if e.shape != (self.N, 3):
                raise ValueError(f"spin file has shape {e.shape}, expected {(self.N, 3)}")
        else:
            raise ValueError(f"unknown initial state '{kind}'")
        if cone > 0:
            e = e + cone * rng.normal(size=e.shape)
        return normalize(e)

    def order_parameter(self, e):
        """|sum_i s_i e_i| / N (ferromagnet: magnetisation; AFM: Neel vector)."""
        return float(np.linalg.norm((self.signs[:, None] * e).sum(0)) / self.N)

    def magnetization(self, e, weighted=False):
        if weighted:
            return (self.mu[:, None] * e).sum(0) / self.mu.sum()
        return e.mean(0)

    def kind(self):
        """'ferromagnet', 'antiferromagnet' or 'ferrimagnet' from the reference state."""
        s, mu = self.uc.signs, self.uc.moments
        if np.all(s == s[0]):
            return "ferromagnet"
        return "antiferromagnet" if abs(np.sum(s * mu)) < 1e-3 * np.sum(mu) else "ferrimagnet"

    def summary(self):
        i = self.uc.info
        lines = [f"spin model: {self.uc.nsites} magnetic sites/cell ({', '.join(self.uc.symbols)}), "
                 f"moments {np.round(self.uc.moments, 3).tolist()} mu_B, reference = {self.kind()}",
                 f"  couplings: J = {i['J']}, rcut = {i['rcut']:.3f} A (NN {i['nn_distance']:.3f} A), "
                 f"{len(i['shells'])} shells, {i['ncouplings_per_cell']} couplings/cell, "
                 f"{100 * i['J0_fraction']:.1f}% of J(q=0) retained",
                 f"  supercell {self.L}, pbc {self.pbc}: N = {self.N} spins, nnz = {self.Jsp.nnz}",
                 f"  single-ion anisotropy: {'on' if self.has_A else 'off'};  field B = {self.B.tolist()} T"]
        l0, lmax, qmax = jq_stability(self.uc)
        if lmax > l0 * (1 + 1e-3) + 1e-6:
            lines.append(f"  WARNING: the reference state is not the ground state of this coupling set: "
                         f"J(q={np.round(qmax, 3).tolist()}) = {lmax:.2f} > J(0) = {l0:.2f} meV. "
                         "Increase rcut (long-range exchange truncated?) or expect a spiral.")
        if self.self_folded:
            lines.append(f"  WARNING: {self.self_folded} couplings reach their own periodic image "
                         "(folded into on-site terms); enlarge the supercell")
        return "\n".join(lines)


def normalize(e):
    return e / np.linalg.norm(e, axis=1, keepdims=True)


def build_from_config(cfg, base="."):
    """SpinSystem from a [spinmodel] table."""
    sm = dict(cfg.get("spinmodel", {}))
    if "model" in sm:                      # inline model Hamiltonian: model = {lattice=..., J=[...], D=...}
        import tempfile
        from .models import write_model
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        write_model(path, **sm["model"])
    else:
        path = sm.get("results", "magex_results/results.json")
        if not os.path.isabs(path):
            path = os.path.join(base, path)
    uc = load_results(path, J=sm.get("J", "mft"), anisotropic=sm.get("anisotropic", True),
                      sia=sm.get("sia", True), rcut=sm.get("rcut"), jtol=sm.get("jtol", 1e-4),
                      moments=sm.get("moments"), axis=sm.get("axis"))
    return uc, sm
