"""Tight-binding model in a spinor basis with explicit spin decomposition.

Every model is stored as

    H(R) = H0(R) (x) 1  +  sum_a [ X_a(R) + S_a(R) ] (x) sigma_a

on an orbital basis of ``norb`` real-orbital Wannier functions (spinor size 2*norb,
spin-block ordering: [all up | all down]).

  X_a  time-odd spin part (exchange field, incl. off-site spin-dependent hopping)
  S_a  time-even spin part (spin-orbit coupling)

For a time-reversal-symmetric (real) orbital gauge, X = Re(h), S = i Im(h) where h is
the full spin-vector part.  The *reference direction* of the magnetisation is changed by
rotating only X (spin space, relative to the lattice) while keeping H0 and S fixed.
This is what allows the full exchange tensor, DMI and MAE to be obtained from a single
DFT calculation (collinear + atomic SOC, or one spinor run).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import io_w90
from .orbitals import ls_matrix, XI_DEFAULT

PAULI = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]], complex)


def rotation_to(n):
    """Proper rotation matrix taking z-hat to unit vector n."""
    n = np.asarray(n, float) / np.linalg.norm(n)
    z = np.array([0, 0, 1.0])
    v = np.cross(z, n)
    c = float(np.dot(z, n))
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1 / (1 + c))


@dataclass
class TBModel:
    cell: np.ndarray                 # (3,3) rows = lattice vectors (Angstrom)
    positions: np.ndarray            # (nat,3) cartesian Angstrom
    symbols: list
    R: np.ndarray                    # (nR,3) int
    H0: np.ndarray                   # (nR,n,n)
    X: np.ndarray                    # (nR,3,n,n) time-odd spin part
    S: np.ndarray                    # (nR,3,n,n) time-even spin part (SOC)
    orb_atom: np.ndarray             # (n,)
    orb_l: np.ndarray
    orb_m: np.ndarray
    efermi: float = 0.0
    info: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ basics
    @property
    def norb(self):
        return self.H0.shape[1]

    @property
    def nspinor(self):
        return 2 * self.norb

    @property
    def natoms(self):
        return len(self.symbols)

    @property
    def frac_positions(self):
        return self.positions @ np.linalg.inv(self.cell)

    def atom_orbitals(self, a):
        return np.where(self.orb_atom == a)[0]

    def atom_spinor_index(self, a):
        o = self.atom_orbitals(a)
        return np.concatenate([o, o + self.norb])

    def r0_index(self):
        return int(np.where(np.all(self.R == 0, axis=1))[0][0])

    def has_soc(self):
        return bool(np.max(np.abs(self.S)) > 1e-8)

    # ------------------------------------------------------------- rotations
    def rotated(self, rot):
        """Model whose time-odd spin field is rotated by the 3x3 matrix ``rot`` (spin space)."""
        Xr = np.einsum("ab,rbmn->ramn", rot, self.X)
        return TBModel(self.cell, self.positions, self.symbols, self.R, self.H0, Xr, self.S,
                       self.orb_atom, self.orb_l, self.orb_m, self.efermi, dict(self.info))

    def with_direction(self, n):
        """Rotate a model whose magnetisation is along z into direction n."""
        return self.rotated(rotation_to(n))

    # ---------------------------------------------------------- Hamiltonian
    def spinor_HR(self):
        h = self.X + self.S
        n = self.norb
        out = np.zeros((len(self.R), 2 * n, 2 * n), complex)
        out[:, :n, :n] = self.H0 + h[:, 2]
        out[:, n:, n:] = self.H0 - h[:, 2]
        out[:, :n, n:] = h[:, 0] - 1j * h[:, 1]
        out[:, n:, :n] = h[:, 0] + 1j * h[:, 1]
        return out

    def Hk(self, kfrac, HR=None):
        """H(k) = sum_R H(R) exp(2 pi i k.R); kfrac (nk,3) -> (nk,N,N)."""
        if HR is None:
            HR = self.spinor_HR()
        ph = np.exp(2j * np.pi * (np.atleast_2d(kfrac) @ self.R.T))
        return np.tensordot(ph, HR, axes=(1, 0))

    def onsite_exchange(self, a):
        """Time-odd on-site spin field p_a (3, no, no) of atom a."""
        o = self.atom_orbitals(a)
        r0 = self.r0_index()
        return self.X[r0][:, o][:, :, o]

    def exchange_operator(self, a):
        """Spinor operator p_a . sigma on the atom's spinor block (2no,2no)."""
        p = self.onsite_exchange(a)
        return sum(np.kron(PAULI[c], p[c]) for c in range(3))

    # ------------------------------------------------------------- SOC
    def add_atomic_soc(self, xi=None, scale=1.0):
        """Add sum_atoms xi_l L.S on site; xi: dict {(symbol,l): eV}; defaults from table."""
        xi = dict(XI_DEFAULT if xi is None else xi)
        r0 = self.r0_index()
        S = self.S.copy()
        used = {}
        for a in range(self.natoms):
            o = self.atom_orbitals(a)
            ls_here = sorted(set(self.orb_l[o]))
            for l in ls_here:
                if l == 0:
                    continue
                key = (self.symbols[a], int(l))
                if key not in xi:
                    continue
                sel = o[self.orb_l[o] == l]
                # assumes shells appear once per atom in m-order
                if len(sel) != 2 * l + 1:
                    raise ValueError(f"atom {a}: incomplete l={l} shell")
                LS = ls_matrix(l)                       # spin-block ordered (2n,2n)
                nn = 2 * l + 1
                ss = np.array([LS[:nn, :nn], LS[:nn, nn:], LS[nn:, :nn], LS[nn:, nn:]])
                # LS = sum_c (L_c/2) (x) sigma_c  ->  S_c = xi * L_c / 2
                hz = 0.5 * (ss[0] - ss[3])
                hx = 0.5 * (ss[1] + ss[2])
                hy = 0.5j * (ss[1] - ss[2])
                for c, hc in enumerate((hx, hy, hz)):
                    S[r0, c][np.ix_(sel, sel)] += scale * xi[key] * hc
                used[key] = xi[key] * scale
        m = TBModel(self.cell, self.positions, self.symbols, self.R, self.H0, self.X, S,
                    self.orb_atom, self.orb_l, self.orb_m, self.efermi, dict(self.info))
        m.info["soc_xi"] = used
        return m

    # ------------------------------------------------------------ builders
    @staticmethod
    def from_spinor_HR(cell, positions, symbols, R, HR, orb_atom, orb_l, orb_m,
                       order="interleaved", efermi=0.0, tr_split=True):
        nw = HR.shape[1]
        n = nw // 2
        if order == "interleaved":
            perm = np.concatenate([np.arange(0, nw, 2), np.arange(1, nw, 2)])
        elif order == "blocked":
            perm = np.arange(nw)
        else:
            raise ValueError(order)
        H = HR[:, perm][:, :, perm]
        uu, ud, du, dd = H[:, :n, :n], H[:, :n, n:], H[:, n:, :n], H[:, n:, n:]
        H0 = 0.5 * (uu + dd)
        h = np.stack([0.5 * (ud + du), 0.5j * (ud - du), 0.5 * (uu - dd)], axis=1)
        if tr_split:
            X = h.real.astype(complex)
            S = 1j * h.imag
        else:
            X, S = h, np.zeros_like(h)
        return TBModel(cell, positions, symbols, R, H0, X, S, orb_atom, orb_l, orb_m, efermi)

    @staticmethod
    def from_collinear_HR(cell, positions, symbols, R, Hup, Hdn, orb_atom, orb_l, orb_m, efermi=0.0):
        H0 = 0.5 * (Hup + Hdn)
        X = np.zeros((len(R), 3) + Hup.shape[1:], complex)
        X[:, 2] = 0.5 * (Hup - Hdn)
        S = np.zeros_like(X)
        return TBModel(cell, positions, symbols, R, H0, X, S, orb_atom, orb_l, orb_m, efermi)


# ---------------------------------------------------------------------------
#                         Wannier90 front end
# ---------------------------------------------------------------------------
def _merge_R(R1, H1, R2, H2):
    """Put two R-indexed arrays on a common R list (union)."""
    keys = {tuple(r): i for i, r in enumerate(R1)}
    allR = list(map(tuple, R1)) + [tuple(r) for r in R2 if tuple(r) not in keys]
    idx = {r: i for i, r in enumerate(allR)}
    out1 = np.zeros((len(allR),) + H1.shape[1:], complex)
    out2 = np.zeros((len(allR),) + H2.shape[1:], complex)
    out1[[idx[tuple(r)] for r in R1]] = H1
    out2[[idx[tuple(r)] for r in R2]] = H2
    return np.array(allR, int), out1, out2


def _orbital_table(win, natoms_labels, nwann_orb):
    """Expand the projections block to per-orbital (atom, l, m) in W90 order."""
    labels, symbols = natoms_labels
    atom, ls, ms = [], [], []
    for site, lst in win.projections:
        sites = [i for i, lab in enumerate(labels) if lab == site]
        if not sites:
            sites = [i for i, s in enumerate(symbols) if s == site]
        if not sites:
            raise ValueError(f"projection site '{site}' matches no atom")
        for a in sites:
            for l in lst:
                for m in range(2 * l + 1):
                    atom.append(a)
                    ls.append(l)
                    ms.append(m)
    if len(atom) != nwann_orb:
        raise ValueError(f"projections expand to {len(atom)} orbitals but hr.dat has {nwann_orb}")
    return np.array(atom), np.array(ls), np.array(ms)


def _home_shift(R, Hlist, cell, positions, orb_atom, centres, tol):
    """Translate every WF into the home cell of its atom.  Returns new R and H's."""
    if centres is None:
        return R, Hlist, np.zeros((len(orb_atom), 3), int)
    inv = np.linalg.inv(cell)
    d = (centres - positions[orb_atom]) @ inv
    T = np.rint(d).astype(int)
    resid = np.linalg.norm((d - T) @ cell, axis=1)
    bad = np.where(resid > tol)[0]
    if len(bad):
        msg = ", ".join(f"WF{i + 1}: {resid[i]:.2f} A" for i in bad[:8])
        raise ValueError(
            "Wannier centres are far from their projection atoms (" + msg + ").\n"
            "Use projected WFs (num_iter=0) or a bigger tolerance (wf_centre_tol).")
    if not np.any(T):
        return R, Hlist, T
    # H'_{mn}(R') with R' = R + T_m - T_n
    n = len(orb_atom)
    newR = {}
    out = [dict() for _ in Hlist]
    dT = T[:, None, :] - T[None, :, :]
    for ir, r in enumerate(R):
        Rp = r[None, None, :] + dT  # (n,n,3)
        for key in np.unique(Rp.reshape(-1, 3), axis=0):
            mask = np.all(Rp == key, axis=2)
            k = tuple(key)
            if k not in newR:
                newR[k] = len(newR)
                for o in out:
                    o[k] = np.zeros((n, n), complex)
            for o, H in zip(out, Hlist):
                o[k][mask] += H[ir][mask]
    Rn = np.array(list(newR.keys()), int)
    return Rn, [np.array([o[tuple(r)] for r in Rn]) for o in out], T


def load_wannier90(cfg) -> TBModel:
    """cfg: dict with keys
        win          path to seedname.win (cell, atoms, projections)
        hr_up/hr_dn  collinear hr files       OR   hr  (spinor hr file)
        centres      optional seedname_centres.xyz (write_xyz = true)
        spinor_order 'interleaved' (QE, VASP+W90 default) | 'blocked'
        efermi       Fermi energy (eV)
    """
    win = io_w90.read_win(cfg["win"])
    if cfg.get("poscar"):
        win.cell, win.symbols, win.positions = io_w90.read_poscar(cfg["poscar"])
    if win.cell is None or win.positions is None:
        raise ValueError("no structure: give unit_cell_cart/atoms in the .win or 'poscar' in [input]")
    labels = []
    text = open(cfg["win"]).read()
    for name in ("atoms_frac", "atoms_cart"):
        blk = io_w90._block(text, name)
        if blk:
            labels = [ln.split()[0] for ln in blk if not ln.lower().startswith(("bohr", "ang"))]
    if not labels:
        labels = list(win.symbols)
    centres = None
    if cfg.get("centres"):
        centres, _ = io_w90.read_centres(cfg["centres"])
    tol = float(cfg.get("wf_centre_tol", 0.6))
    if cfg.get("hr_up"):
        R1, Hu = io_w90.read_hr(cfg["hr_up"])
        R2, Hd = io_w90.read_hr(cfg["hr_dn"])
        R, Hu, Hd = _merge_R(R1, Hu, R2, Hd)
        oa, ol, om = _orbital_table(win, (labels, win.symbols), Hu.shape[1])
        R, (Hu, Hd), T = _home_shift(R, [Hu, Hd], win.cell, win.positions, oa, centres, tol)
        m = TBModel.from_collinear_HR(win.cell, win.positions, win.symbols, R, Hu, Hd,
                                      oa, ol, om, float(cfg.get("efermi", 0.0)))
        m.info["source"] = "wannier90-collinear"
    else:
        R, H = io_w90.read_hr(cfg["hr"])
        nw = H.shape[1]
        oa, ol, om = _orbital_table(win, (labels, win.symbols), nw // 2)
        order = cfg.get("spinor_order", "interleaved")
        if centres is not None:
            if order == "interleaved":
                centres = centres[0::2]
            else:
                centres = centres[: nw // 2]
        # shift in orbital space acts identically on both spin components
        n = nw // 2
        perm = (np.concatenate([np.arange(0, nw, 2), np.arange(1, nw, 2)])
                if order == "interleaved" else np.arange(nw))
        Hb = H[:, perm][:, :, perm]
        blocks = [Hb[:, :n, :n], Hb[:, :n, n:], Hb[:, n:, :n], Hb[:, n:, n:]]
        R, blocks, T = _home_shift(R, blocks, win.cell, win.positions, oa, centres, tol)
        Hb = np.zeros((len(R), nw, nw), complex)
        Hb[:, :n, :n], Hb[:, :n, n:], Hb[:, n:, :n], Hb[:, n:, n:] = blocks
        m = TBModel.from_spinor_HR(win.cell, win.positions, win.symbols, R, Hb, oa, ol, om,
                                   order="blocked", efermi=float(cfg.get("efermi", 0.0)),
                                   tr_split=bool(cfg.get("tr_split", True)))
        m.info["source"] = "wannier90-spinor"
        # diagnostic for the time-reversal split (gauge must be ~real)
        m.info["tr_gauge_imag_H0"] = float(np.max(np.abs(m.H0.imag)))
    m.info["win"] = win
    return m
