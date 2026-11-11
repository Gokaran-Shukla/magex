"""Finite-angle (Lloyd) analysis and magnetocrystalline anisotropy."""
from __future__ import annotations

import numpy as np

from . import parallel as par
from .engine import kmesh_frac


def lloyd_fit(ll, signs, J_small, dref, axis, units=1000.0):
    """Fit Lloyd pair-interaction energies (i rotated by -t/2, j by +t/2 about ``axis``;
    E_int = dOmega_ij - dOmega_i - dOmega_j) to isotropic bilinear + biquadratic terms.

    Isotropic model  E = -sum_{i!=j} [J' e_i.e_j + B (e_i.e_j)^2]  gives
        E_int(t) = 4 s J' c (1 - c)  -  2 B cos t (cos t - 1),   c = cos(t/2),  s = s_i s_j
    with small-angle limit  J_small = J' + 2 s B.  The bilinear reference curve uses the full
    force-theorem tensor:  E_bil(t) = -2 (e_i - n_i).J_ij.(e_j - n_j).
    """
    from .engine import rodrigues
    th = ll["angles"]
    c = np.cos(th / 2)
    dref = np.asarray(dref, float)
    out = []
    for p, (i, j, nu) in enumerate(ll["pairs"]):
        E = ll["E"][p] * units
        ss = signs[i] * signs[j]
        Jt = np.asarray(J_small[p])
        ni, nj = signs[i] * dref, signs[j] * dref
        Ebil = np.array([-2 * (rodrigues(axis, -t / 2) @ ni - ni) @ Jt @ (rodrigues(axis, t / 2) @ nj - nj)
                         for t in th])
        # subtract the anisotropic part of the bilinear curve before the isotropic fit
        Eiso_bil = 4 * ss * (np.trace(Jt) / 3) * c * (1 - c)
        Efit = E - (Ebil - Eiso_bil)
        M = np.stack([4 * ss * c * (1 - c), -2 * np.cos(th) * (np.cos(th) - 1)], axis=1)
        (Jp, B), *_ = np.linalg.lstsq(M, Efit, rcond=None)
        scale = max(np.max(np.abs(E)), np.max(np.abs(Ebil)), 1e-12)
        out.append(dict(pair=(i, j, nu), angles=th, E=E, E_bilinear=Ebil, J_bilinear=float(Jp), B=float(B),
                        J_small_angle=float(np.trace(Jt) / 3), J_eff_from_fit=float(Jp + 2 * ss * B),
                        heisenberg_error=float(np.max(np.abs(E - Ebil)) / scale),
                        fit_resid=float(np.linalg.norm(M @ [Jp, B] - Efit) / np.sqrt(len(E)) / scale)))
    return out


def grand_potential_directions(model, directions, mesh, efermi, kT):
    """Band grand potential (frozen potential) per cell for global magnetisation directions.

    The model's magnetisation must point along z.  Returns eV per cell.
    """
    kpts = kmesh_frac(mesh)
    myk = par.split(len(kpts))
    out = np.zeros(len(directions))
    for idir, d in enumerate(directions):
        m = model.with_direction(d)
        HR = m.spinor_HR()
        acc = 0.0
        for c0 in range(0, len(myk), 64):
            ks = myk[c0:c0 + 64]
            e = np.linalg.eigvalsh(m.Hk(kpts[ks], HR))
            x = (e - efermi) / kT
            acc += np.sum(-kT * np.logaddexp(0.0, -x))
        out[idir] = acc
    return par.allreduce_sum(out) / len(kpts)


def default_directions():
    s2, s3 = 1 / np.sqrt(2), 1 / np.sqrt(3)
    return np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [s2, s2, 0], [s2, 0, s2], [0, s2, s2],
                     [s2, -s2, 0], [s3, s3, s3]])


def fit_quadratic_anisotropy(directions, E):
    """E(n) = const + n.A.n with A symmetric traceless; returns (A, const, rms)."""
    rows = []
    for n in directions:
        n = n / np.linalg.norm(n)
        rows.append([1.0, n[0]**2 - n[2]**2, n[1]**2 - n[2]**2, 2 * n[0] * n[1], 2 * n[0] * n[2], 2 * n[1] * n[2]])
    M = np.array(rows)
    sol, *_ = np.linalg.lstsq(M, E, rcond=None)
    c, axx, ayy, axy, axz, ayz = sol
    A = np.array([[axx, axy, axz], [axy, ayy, ayz], [axz, ayz, -axx - ayy]])
    rms = float(np.linalg.norm(M @ sol - E) / np.sqrt(len(E)))
    return A, c, rms


def heisenberg_anisotropy(res, directions, mag_only=True):
    """Global-rotation energy predicted by the fitted Heisenberg model (meV per cell)."""
    s = np.sign(res.moments @ np.array(res.refs[-1]))
    s[s == 0] = 1
    E = []
    for n in directions:
        n = n / np.linalg.norm(n)
        e = sum(n @ res.A[i] @ n for i in range(len(res.sites)))
        for p in res.pairs:
            e -= s[p.i] * s[p.j] * (n @ p.J @ n)
        E.append(e)
    return np.array(E)
