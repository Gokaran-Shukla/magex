"""Readers for Wannier90 files: seedname_hr.dat, _centres.xyz, .win, .eig."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from .orbitals import L_OF

BOHR = 0.52917721067


def read_hr(fname):
    """Return (Rvecs (nR,3) int, H (nR, nw, nw) complex) with degeneracy divided out."""
    with open(fname) as f:
        f.readline()
        nw = int(f.readline())
        nR = int(f.readline())
        deg = []
        while len(deg) < nR:
            deg += [int(x) for x in f.readline().split()]
        data = np.loadtxt(f)
    data = data.reshape(nR, nw * nw, 7)
    R = data[:, 0, :3].astype(int)
    m = data[0, :, 3].astype(int) - 1
    n = data[0, :, 4].astype(int) - 1
    H = np.zeros((nR, nw, nw), complex)
    H[:, m, n] = data[:, :, 5] + 1j * data[:, :, 6]
    H /= np.array(deg)[:, None, None]
    return R, H


def read_centres(fname):
    """Return (wf_centres (nw,3), atoms list[(symbol, pos)]) in Angstrom."""
    centres, atoms = [], []
    with open(fname) as f:
        n = int(f.readline())
        f.readline()
        for _ in range(n):
            p = f.readline().split()
            if p[0] == "X":
                centres.append([float(x) for x in p[1:4]])
            else:
                atoms.append((p[0], np.array([float(x) for x in p[1:4]])))
    return np.array(centres), atoms


@dataclass
class WinInfo:
    cell: np.ndarray | None = None        # rows = lattice vectors, Angstrom
    symbols: list = field(default_factory=list)
    positions: np.ndarray | None = None   # cartesian Angstrom
    projections: list = field(default_factory=list)  # [(site_spec, [l,...]), ...]
    spinors: bool = False
    mp_grid: tuple | None = None
    kpoints: np.ndarray | None = None
    dis_froz_max: float | None = None
    num_wann: int | None = None
    exclude_bands: list = field(default_factory=list)


def _block(text, name):
    m = re.search(rf"begin\s+{name}(.*?)end\s+{name}", text, re.I | re.S)
    if not m:
        return None
    lines = [ln.split("!")[0].split("#")[0].strip() for ln in m.group(1).splitlines()]
    return [ln for ln in lines if ln]


def _keyword(text, key):
    m = re.search(rf"^\s*{key}\s*[=:\s]\s*(.+)$", text, re.I | re.M)
    return m.group(1).split("!")[0].strip() if m else None


def _parse_range(s):
    out = []
    for tok in s.replace(",", " ").split():
        if "-" in tok:
            a, b = tok.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(tok))
    return out


def _ff(x):
    """Fortran-aware float ('1.0d0')."""
    return float(x.lower().replace("d", "e"))


def read_win(fname) -> WinInfo:
    text = open(fname).read()
    w = WinInfo()
    blk = _block(text, "unit_cell_cart")
    if blk:
        unit = 1.0
        if blk[0].lower().startswith("bohr"):
            unit, blk = BOHR, blk[1:]
        elif blk[0].lower().startswith("ang"):
            blk = blk[1:]
        w.cell = np.array([[_ff(x) for x in ln.split()[:3]] for ln in blk[:3]]) * unit
    for name, frac in (("atoms_frac", True), ("atoms_cart", False)):
        blk = _block(text, name)
        if not blk:
            continue
        unit = 1.0
        if not frac and blk[0].lower().startswith("bohr"):
            unit, blk = BOHR, blk[1:]
        elif not frac and blk[0].lower().startswith("ang"):
            blk = blk[1:]
        sy, pos = [], []
        for ln in blk:
            p = ln.split()
            sy.append(re.sub(r"\d+$", "", p[0]))
            pos.append([_ff(x) for x in p[1:4]])
        pos = np.array(pos)
        w.symbols = sy
        w.positions = pos @ w.cell if frac else pos * unit
    blk = _block(text, "projections")
    if blk:
        for ln in blk:
            if ln.lower().startswith("random"):
                raise ValueError("'random' projections cannot be mapped to orbitals")
            site, orbs = ln.split(":", 1)
            orbs = orbs.split(":")[0]  # drop local-axis / radial specifiers
            if re.search(r":\s*[zx]\s*=", ln):
                raise ValueError(f"local axes in projection '{ln}' are not supported yet")
            ls = []
            for o in orbs.replace(";", ",").split(","):
                o = o.strip().lower()
                if o.startswith("l="):
                    ls.append(int(o[2:]))
                elif o in L_OF:
                    ls.append(L_OF[o])
                else:
                    raise ValueError(f"projection '{o}' must be a full shell (s,p,d,f or l=)")
            w.projections.append((site.strip(), ls))
    sp = _keyword(text, "spinors")
    w.spinors = bool(sp and sp.lower().startswith((".t", "t")))
    mp = _keyword(text, "mp_grid")
    if mp:
        w.mp_grid = tuple(int(x) for x in mp.split()[:3])
    blk = _block(text, "kpoints")
    if blk:
        w.kpoints = np.array([[_ff(x) for x in ln.split()[:3]] for ln in blk])
    dfm = _keyword(text, "dis_froz_max")
    if dfm:
        w.dis_froz_max = _ff(dfm.split()[0])
    nw = _keyword(text, "num_wann")
    if nw:
        w.num_wann = int(nw.split()[0])
    eb = _keyword(text, "exclude_bands")
    if eb:
        w.exclude_bands = _parse_range(eb)
    return w


def read_eig(fname, nk):
    d = np.loadtxt(fname)
    nb = int(d[:, 0].max())
    return d[:, 2].reshape(nk, nb)


def read_poscar(fname):
    """Minimal VASP POSCAR reader -> (cell rows Angstrom, symbols list, cartesian positions)."""
    with open(fname) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    scale = float(lines[1].split()[0])
    cell = np.array([[float(x) for x in lines[i].split()[:3]] for i in (2, 3, 4)]) * scale
    i = 5
    if lines[i].split()[0].isdigit():
        species = lines[0].split()
    else:
        species = lines[i].split()
        i += 1
    counts = [int(x) for x in lines[i].split()]
    i += 1
    if lines[i][0] in "sS":
        i += 1
    cart = lines[i][0] in "cCkK"
    i += 1
    n = sum(counts)
    pos = np.array([[float(x) for x in lines[i + k].split()[:3]] for k in range(n)])
    symbols = [sp for sp, c in zip(species, counts) for _ in range(c)]
    pos = pos * scale if cart else pos @ cell
    return cell, symbols, pos
