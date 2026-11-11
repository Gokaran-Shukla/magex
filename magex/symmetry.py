"""Space-group symmetrisation of exchange tensors and single-ion anisotropy.

Each pair tensor is replaced by the group average  J(p) = <Rc^T J(g.p) Rc>_g  over the
crystal space group (spglib), with atoms distinguished by species and |moment|.
Exchange couples two axial vectors, so improper operations and time reversal enter
only through Rc J Rc^T.  The size of the change is reported as a Wannier-noise estimate.
"""
from __future__ import annotations

import numpy as np


def space_group_ops(model, sites, moments, symprec=1e-3, momprec=0.05):
    import spglib
    types = []
    mag = {a: float(np.linalg.norm(m)) for a, m in zip(sites, moments)}
    keys = {}
    for a, s in enumerate(model.symbols):
        key = (s, round(mag.get(a, 0.0) / momprec))
        keys.setdefault(key, len(keys) + 1)
        types.append(keys[key])
    cell = (model.cell, model.frac_positions, types)
    sym = spglib.get_symmetry(cell, symprec=symprec)
    ds = spglib.get_symmetry_dataset(cell, symprec=symprec)
    name = getattr(ds, "international", None) if ds is not None else None
    return sym["rotations"], sym["translations"], name


def _atom_map(W, t, frac, tol=1e-3):
    nat = len(frac)
    new = frac @ W.T + t
    amap = np.zeros(nat, int)
    shift = np.zeros((nat, 3), int)
    for a in range(nat):
        d = new[a][None, :] - frac
        L = np.rint(d)
        r = np.linalg.norm(d - L, axis=1)
        b = int(np.argmin(r))
        if r[b] > tol * 10:
            raise RuntimeError("symmetry operation does not map the structure onto itself")
        amap[a] = b
        shift[a] = L[b].astype(int)
    return amap, shift


def symmetrize(res, model, symprec=1e-3):
    rots, trans, name = space_group_ops(model, res.sites, res.moments, symprec)
    frac = model.frac_positions
    C = model.cell
    mesh = np.array(res.mesh)
    site_pos = {a: k for k, a in enumerate(res.sites)}
    lookup = {(p.i, p.j, p.nu): p for p in res.pairs}
    acc = {k: np.zeros((3, 3)) for k in lookup}
    cnt = {k: 0 for k in lookup}
    accA = np.zeros_like(res.A)
    for W, t in zip(rots, trans):
        Rc = C.T @ W @ np.linalg.inv(C.T)
        amap, shift = _atom_map(W, t, frac, symprec)
        for (i, j, nu), p in lookup.items():
            gi, gj = amap[res.sites[i]], amap[res.sites[j]]
            if gi not in site_pos or gj not in site_pos:
                continue
            Rn = shift[res.sites[j]] + W @ np.array(p.R) - shift[res.sites[i]]
            key = (site_pos[gi], site_pos[gj], tuple(int(x) for x in np.mod(Rn, mesh)))
            q = lookup.get(key)
            if q is None or abs(q.dist - p.dist) > 1e-4:
                continue           # aliased partner at the supercell boundary
            acc[(i, j, nu)] += Rc.T @ q.J @ Rc
            cnt[(i, j, nu)] += 1
        for i, a in enumerate(res.sites):
            gi = amap[a]
            if gi in site_pos:
                accA[i] += Rc.T @ res.A[site_pos[gi]] @ Rc
    change = 0.0
    for k, p in lookup.items():
        if cnt[k]:
            Js = acc[k] / cnt[k]
            change = max(change, float(np.max(np.abs(Js - p.J))))
            p.J = Js
    Anew = accA / len(rots)
    changeA = float(np.max(np.abs(Anew - res.A))) if res.A is not None else 0.0
    res.A = Anew
    res.extra["symmetry"] = dict(spacegroup=name, nops=len(rots), max_change_J=change,
                                 max_change_A=changeA)
    return res
