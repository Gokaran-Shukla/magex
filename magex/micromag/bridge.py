"""Atomistic -> micromagnetic parameters from a magex (or model) results.json.

Gradient expansion of  E = -sum_{i!=j} e_i.J_ij.e_j + sum_i e_i.A_i.e_i  for a slowly varying
ferromagnetic texture (e_j ~ e_i + (r_ij.grad) e):

  exchange stiffness tensor  A_ab = (1/2V_c) sum_{i in cell} sum_j J^iso_ij r_a r_b
                             (isotropic part A = tr/3 in 3D, (A_xx + A_yy)/2 for films)
  interfacial DMI            D_int  = -(1/2V_c) sum (z x r_hat).D_ij |r|  over in-plane bonds
  bulk DMI                   D_bulk =  (1/3V_c) sum  r_hat.D_ij |r|
  uniaxial anisotropy        K_u = [ sum_i (A_i,xx - A_i,zz) + sum_ij (J^S_zz - J^S_xx) ] / V_c
  saturation magnetisation   Ms = mu_B sum_i |mu_i| / V_c

with D_ij the DMI vector of the antisymmetric part of J_ij.  For 2D layers (vacuum in the
cell) V_c = in-plane cell area x thickness (default: the interlayer spacing given by the
user, or the nearest-neighbour distance).  Only meaningful for ferromagnets; for
antiferromagnets the sublattice-resolved couplings are reported instead.
"""
from __future__ import annotations

import numpy as np

from .constants import MEV, MU_B

ANG = 1e-10


def micromagnetic_parameters(path, J="mft", rcut=None, thickness_A=None, film_normal=2, log=None):
    from ..spinmodel import load_results
    uc = load_results(path, J=J, rcut=rcut)
    cell = uc.cell
    n = film_normal
    lengths = np.linalg.norm(cell, axis=1)
    pos = uc.positions
    zspread = np.ptp(pos[:, n]) if len(pos) > 1 else 0.0
    is_film = abs(cell[n, n]) - zspread > 8.0          # > 8 A vacuum along the normal
    if is_film:
        inplane = [k for k in range(3) if k != n]
        area = np.linalg.norm(np.cross(cell[inplane[0]], cell[inplane[1]]))
        t = thickness_A or (zspread + (uc.info.get("nn_distance") or 3.0))
        Vc = area * t
    else:
        Vc = abs(np.linalg.det(cell))
        t = None
    Aab = np.zeros((3, 3))
    Dint = 0.0
    Dbulk = 0.0
    Ktwo = 0.0
    zhat = np.zeros(3)
    zhat[n] = 1.0
    for (i, j, R, Jt) in uc.couplings:
        sij = uc.signs[i] * uc.signs[j]
        r = pos[j] + np.asarray(R) @ cell - pos[i]
        d = np.linalg.norm(r)
        Jt = np.asarray(Jt)
        iso = np.trace(Jt) / 3
        Aab += 0.5 * sij * iso * np.outer(r, r)
        JA = 0.5 * (Jt - Jt.T)
        Dv = np.array([JA[1, 2], JA[2, 0], JA[0, 1]])
        rh = r / d
        if abs(r[n]) < 1e-3:
            Dint += -0.5 * np.dot(np.cross(zhat, rh), Dv) * d
        Dbulk += np.dot(rh, Dv) * d / 3
        JS = 0.5 * (Jt + Jt.T)
        Ktwo += sij * (JS[n, n] - JS[(n + 1) % 3, (n + 1) % 3])
    Ksi = sum(Ai[(n + 1) % 3, (n + 1) % 3] - Ai[n, n] for Ai in uc.A)
    conv_A = MEV / ANG                                     # meV/A -> J/m
    Aab_SI = Aab / Vc * conv_A
    A_iso = (np.trace(Aab_SI) - Aab_SI[n, n]) / 2 if is_film else np.trace(Aab_SI) / 3
    out = dict(
        Ms=float(MU_B * np.sum(np.abs(uc.moments)) / (Vc * ANG**3)),
        A=float(A_iso), A_tensor=Aab_SI.tolist(),
        D_int=float(Dint / Vc * MEV / ANG**2), D_bulk=float(Dbulk / Vc * MEV / ANG**2),
        Ku=float((Ksi + Ktwo) / Vc * MEV / ANG**3), Ku_single_ion=float(Ksi / Vc * MEV / ANG**3),
        Ku_two_site=float(Ktwo / Vc * MEV / ANG**3),
        V_cell_A3=float(Vc), film=bool(is_film), thickness_A=t, ferromagnet=bool(np.all(uc.signs == uc.signs[0])),
        rcut_A=uc.info.get("rcut"),
    )
    if out["Ms"] > 0 and out["A"] > 0:
        D = out["D_int"] if abs(out["D_int"]) >= abs(out["D_bulk"]) else out["D_bulk"]
        if D:
            out["spiral_period_nm"] = float(4 * np.pi * out["A"] / abs(D) * 1e9)
    if log:
        log("micromagnetic parameters from " + str(path))
        log(f"  Ms = {out['Ms']:.4g} A/m   A = {out['A']:.4g} J/m   D_int = {out['D_int'] * 1e3:.4g} mJ/m^2   "
            f"D_bulk = {out['D_bulk'] * 1e3:.4g} mJ/m^2   Ku = {out['Ku']:.4g} J/m^3 "
            f"(single-ion {out['Ku_single_ion']:.3g}, two-site {out['Ku_two_site']:.3g})")
        if not out["ferromagnet"]:
            log("  NOTE: antiferromagnetic reference; A uses sign-weighted couplings (staggered-order stiffness)")
    return out
