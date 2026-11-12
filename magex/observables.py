"""Observables and output formats for atomistic spin configurations.

  resolved_magnetisation   M per sublattice, material, layer (height) or any integer label
  chirality                scalar chirality e_i.(e_j x e_k) on triangles and vector chirality per bond
  static_structure_factor  S(q) = |sum_i e_i exp(i q.r_i)|^2 / N  on a q grid or q list
  correlation_function     C(r) = <e_i . e_j>_{|r_ij| = r}  (shell-averaged)
  autocorrelation          A(t) = <e_i(t0) . e_i(t0 + t)>  from a trajectory
  write_vtk / write_xyz / write_ovf_atomistic   ParaView (VTK legacy polydata), extended XYZ (OVITO), OVF
"""
from __future__ import annotations

import numpy as np


def resolved_magnetisation(system, e, by="sublattice", weighted=False):
    """dict label -> mean moment direction (3,) for by in sublattice | material | layer | an (N,) int array."""
    if isinstance(by, str):
        lab = {"sublattice": system.sublattice, "material": system.material,
               "layer": system.cells[system.cell_index][:, 2]}[by]
        names = system.material_names if by == "material" else None
    else:
        lab, names = np.asarray(by), None
    out = {}
    w = system.mu if weighted else np.ones(system.N)
    for k in np.unique(lab):
        sel = lab == k
        key = names[k] if names is not None and k < len(names) else int(k)
        out[key] = (w[sel, None] * e[sel]).sum(0) / w[sel].sum()
    return out


def triangles(system):
    """Unique nearest-neighbour triangles (i, j, k) oriented counter-clockwise about +z."""
    P = system.positions
    b = system.bonds
    d = np.linalg.norm(system._minimg(P[b[:, 1]] - P[b[:, 0]]), axis=1)
    nn = b[np.abs(d - d.min()) < 1e-3]
    nbr = [set() for _ in range(system.N)]
    for a, c in nn:
        nbr[a].add(c)
        nbr[c].add(a)
    out = set()
    for i in range(system.N):
        for j in nbr[i]:
            for k in nbr[j] & nbr[i]:
                t = sorted((i, j, k))
                out.add(tuple(t))
    T = np.array(sorted(out), int).reshape(-1, 3)
    if len(T):
        r1 = system._minimg(P[T[:, 1]] - P[T[:, 0]])
        r2 = system._minimg(P[T[:, 2]] - P[T[:, 0]])
        flip = np.cross(r1, r2)[:, 2] < 0
        T[flip] = T[flip][:, [0, 2, 1]]
    return T


def chirality(system, e, T=None):
    """(scalar chirality per triangle (M,), mean vector chirality <e_i x e_j> over nn bonds (3,))."""
    T = triangles(system) if T is None else T
    chi = np.einsum("ia,ia->i", e[T[:, 0]], np.cross(e[T[:, 1]], e[T[:, 2]])) if len(T) else np.zeros(0)
    b = system.bonds
    vec = np.cross(e[b[:, 0]], e[b[:, 1]]).mean(0) if len(b) else np.zeros(3)
    return chi, vec


def static_structure_factor(system, e, q):
    """S(q) for q (M,3) in 1/A (Cartesian): |sum_i e_i exp(i q.r_i)|^2 / N, summed over components."""
    q = np.atleast_2d(q)
    ph = np.exp(1j * system.positions @ q.T)              # (N, M)
    F = np.einsum("ia,im->ma", e, ph)
    return np.sum(np.abs(F) ** 2, axis=1) / system.N


def sq_grid(system, e):
    """S(q) on the commensurate grid of the supercell (one sublattice per FFT) -> (q_frac grid, S)."""
    L = system.L
    full = np.zeros((int(np.prod(L)) * system.uc.nsites, 3))
    full[system.grid_index] = e
    M = full.reshape(tuple(L) + (system.uc.nsites, 3))
    qf = np.stack(np.meshgrid(*[np.fft.fftfreq(n) for n in L], indexing="ij"), -1)
    qc = 2 * np.pi * qf @ np.linalg.inv(system.uc.cell).T
    F = 0
    for s in range(system.uc.nsites):
        F = F + np.fft.fftn(M[..., s, :], axes=(0, 1, 2)) * np.exp(-1j * qc @ system.uc.positions[s])[..., None]
    S = np.sum(np.abs(F) ** 2, axis=-1) / system.N
    return qf, S


def correlation_function(system, e, rmax=None, nbins=None):
    """Shell-averaged C(r) = <e_i . e_j> (pairs within rmax, minimum image).  Returns (r, C, counts)."""
    P = system.positions
    N = system.N
    rmax = rmax or 0.5 * float(np.min(np.array(system.L) * np.linalg.norm(system.uc.cell, axis=1)))
    rs, cs = [], []
    for i0 in range(0, N, 512):
        d = P[None, :, :] - P[i0:i0 + 512, None, :]
        dd = system._minimg(d.reshape(-1, 3)).reshape(d.shape)
        r = np.linalg.norm(dd, axis=-1)
        c = e[i0:i0 + 512] @ e.T
        m = (r > 1e-6) & (r <= rmax)
        rs.append(r[m])
        cs.append(c[m])
    r = np.concatenate(rs)
    c = np.concatenate(cs)
    rr = np.round(r, 3)
    shells, inv = np.unique(rr, return_inverse=True)
    C = np.bincount(inv, c) / np.bincount(inv)
    return shells, C, np.bincount(inv)


def autocorrelation(traj):
    """A(t) = <e_i(t0) . e_i(t0 + t)>_{i, t0} for a trajectory (T, N, 3)."""
    traj = np.asarray(traj)
    nT = len(traj)
    out = np.zeros(nT)
    for lag in range(nT):
        out[lag] = np.mean(np.einsum("tia,tia->ti", traj[: nT - lag], traj[lag:]))
    return out


# ----------------------------------------------------------------------------- output
def write_xyz(path, system, e, comment="", append=False):
    """Extended XYZ (OVITO / ASE): species, position, spin vector, moment."""
    names = system.uc.symbols
    with open(path, "a" if append else "w") as f:
        f.write(f"{system.N}\n")
        f.write(f'Properties=species:S:1:pos:R:3:spin:R:3:mu:R:1 {comment}\n')
        for k in range(system.N):
            p, s = system.positions[k], e[k]
            f.write(f"{names[system.sublattice[k]]} {p[0]:.5f} {p[1]:.5f} {p[2]:.5f} "
                    f"{s[0]:.6f} {s[1]:.6f} {s[2]:.6f} {system.mu[k]:.4f}\n")
    return path


def write_vtk(path, system, e, extra=None):
    """VTK legacy POLYDATA with spin vectors, mz, sublattice and material (ParaView: Glyph filter)."""
    N = system.N
    with open(path, "w") as f:
        f.write("# vtk DataFile Version 3.0\nmagex spins\nASCII\nDATASET POLYDATA\n")
        f.write(f"POINTS {N} double\n")
        np.savetxt(f, system.positions, fmt="%.6f")
        f.write(f"VERTICES {N} {2 * N}\n")
        np.savetxt(f, np.column_stack([np.ones(N, int), np.arange(N)]), fmt="%d")
        f.write(f"POINT_DATA {N}\nVECTORS spin double\n")
        np.savetxt(f, e, fmt="%.6f")
        for name, arr in [("mz", e[:, 2]), ("sublattice", system.sublattice), ("material", system.material)] + \
                list((extra or {}).items()):
            f.write(f"SCALARS {name} double 1\nLOOKUP_TABLE default\n")
            np.savetxt(f, np.asarray(arr, float), fmt="%.6g")
    return path


def write_ovf_atomistic(path, system, e):
    """OVF 2.0 irregular-mesh text file (positions + spins), readable by Spirit/OOMMF-style tools."""
    with open(path, "w") as f:
        f.write("# OOMMF OVF 2.0\n# Segment count: 1\n# Begin: Segment\n# Begin: Header\n# Title: magex\n")
        f.write("# meshtype: irregular\n# meshunit: m\n# valuedim: 3\n# valueunits: 1 1 1\n")
        f.write(f"# pointcount: {system.N}\n# End: Header\n# Begin: Data Text\n")
        np.savetxt(f, np.column_stack([system.positions * 1e-10, e]), fmt="%.8e")
        f.write("# End: Data Text\n# End: Segment\n")
    return path
