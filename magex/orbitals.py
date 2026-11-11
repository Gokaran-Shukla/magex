"""Real-harmonic orbital operators in the Wannier90 convention.

Orbital order (Wannier90 user guide, Table 3.1):
    s : s
    p : pz, px, py
    d : dz2, dxz, dyz, dx2-y2, dxy
    f : fz3, fxz2, fyz2, fz(x2-y2), fxyz, fx(x2-3y2), fy(3x2-y2)

L matrices are built numerically from the Cartesian polynomials themselves,
L = -i r x grad, projected back on the shell.  This fixes every phase to the
Wannier90 definitions without relying on tabulated transformation matrices.
"""
from __future__ import annotations

import numpy as np

# Cartesian polynomial (unnormalised) for every real harmonic, W90 order.
_POLY = {
    0: [lambda x, y, z: np.ones_like(x)],
    1: [lambda x, y, z: z, lambda x, y, z: x, lambda x, y, z: y],
    2: [
        lambda x, y, z: 3 * z**2 - (x**2 + y**2 + z**2),
        lambda x, y, z: x * z,
        lambda x, y, z: y * z,
        lambda x, y, z: x**2 - y**2,
        lambda x, y, z: x * y,
    ],
    3: [
        lambda x, y, z: z * (5 * z**2 - 3 * (x**2 + y**2 + z**2)),
        lambda x, y, z: x * (5 * z**2 - (x**2 + y**2 + z**2)),
        lambda x, y, z: y * (5 * z**2 - (x**2 + y**2 + z**2)),
        lambda x, y, z: z * (x**2 - y**2),
        lambda x, y, z: x * y * z,
        lambda x, y, z: x * (x**2 - 3 * y**2),
        lambda x, y, z: y * (3 * x**2 - y**2),
    ],
}

L_OF = {"s": 0, "p": 1, "d": 2, "f": 3}
NAMES = {
    0: ["s"],
    1: ["pz", "px", "py"],
    2: ["dz2", "dxz", "dyz", "dx2-y2", "dxy"],
    3: ["fz3", "fxz2", "fyz2", "fz(x2-y2)", "fxyz", "fx(x2-3y2)", "fy(3x2-y2)"],
}

_CACHE: dict[int, np.ndarray] = {}


def l_matrices(l: int) -> np.ndarray:
    """Return array (3, 2l+1, 2l+1) with Lx, Ly, Lz in the W90 real basis (hbar=1)."""
    if l in _CACHE:
        return _CACHE[l]
    n = 2 * l + 1
    if l == 0:
        out = np.zeros((3, 1, 1), complex)
        _CACHE[0] = out
        return out
    # product Gauss-Legendre x uniform-phi grid: exact for polynomials of degree <= 2*12
    ct, wt = np.polynomial.legendre.leggauss(12)
    ph = np.arange(25) * 2 * np.pi / 25
    CT, PH = np.meshgrid(ct, ph, indexing="ij")
    W = (wt[:, None] * np.ones_like(PH)).ravel()
    st = np.sqrt(1 - CT**2)
    x, y, z = (st * np.cos(PH)).ravel(), (st * np.sin(PH)).ravel(), CT.ravel()
    funcs = _POLY[l]
    F = np.stack([f(x, y, z) for f in funcs], axis=1)  # (np, n)
    # normalise on the sphere (Monte Carlo norm only fixes scale; we orthonormalise exactly below)
    h = 1e-5

    def grad(f):
        gx = (f(x + h, y, z) - f(x - h, y, z)) / (2 * h)
        gy = (f(x, y + h, z) - f(x, y - h, z)) / (2 * h)
        gz = (f(x, y, z + h) - f(x, y, z - h)) / (2 * h)
        return gx, gy, gz

    LF = np.zeros((3, len(x), n), complex)
    for m, f in enumerate(funcs):
        gx, gy, gz = grad(f)
        LF[0, :, m] = -1j * (y * gz - z * gy)
        LF[1, :, m] = -1j * (z * gx - x * gz)
        LF[2, :, m] = -1j * (x * gy - y * gx)
    # Express L f_m = sum_n f_n C_nm by least squares (exact: L keeps the shell)
    C = np.stack([np.linalg.lstsq(F, LF[a], rcond=None)[0] for a in range(3)])
    # Normalisation: metric of the polynomial basis S_mn = <f_m|f_n> (MC; ratios exact in limit)
    # Use exact metric by symmetry: basis functions are mutually orthogonal; norms differ.
    norms = np.sqrt(np.sum(W[:, None] * F**2, axis=0))
    D = np.diag(norms)
    Dinv = np.diag(1.0 / norms)
    Lmat = np.stack([D @ C[a] @ Dinv for a in range(3)])
    # enforce exact hermiticity / purely imaginary structure (real basis)
    Lmat = 0.5 * (Lmat + np.conj(np.transpose(Lmat, (0, 2, 1))))
    Lmat = 1j * np.round(Lmat.imag, 6)
    _CACHE[l] = Lmat
    return Lmat


def ls_matrix(l: int) -> np.ndarray:
    """Return a (2n, 2n) matrix ordered as [all orbitals spin-up, all orbitals spin-down]
    (spin-block ordering), S = sigma/2.
    """
    L = l_matrices(l)
    n = 2 * l + 1
    sx = np.array([[0, 1], [1, 0]], complex) / 2
    sy = np.array([[0, -1j], [1j, 0]], complex) / 2
    sz = np.array([[1, 0], [0, -1]], complex) / 2
    out = np.kron(sx, L[0]) + np.kron(sy, L[1]) + np.kron(sz, L[2])
    return out.reshape(2 * n, 2 * n)


# Rough atomic SOC constants xi (eV) for H_soc = xi L.S.  These are *defaults only*;
# for quantitative work fit xi to a fully-relativistic band structure (see fit_soc).
XI_DEFAULT = {
    ("Ti", 2): 0.018, ("V", 2): 0.028, ("Cr", 2): 0.035, ("Mn", 2): 0.045,
    ("Fe", 2): 0.055, ("Co", 2): 0.068, ("Ni", 2): 0.085, ("Cu", 2): 0.10,
    ("Nb", 2): 0.10, ("Mo", 2): 0.11, ("Ru", 2): 0.15, ("Rh", 2): 0.17, ("Pd", 2): 0.20,
    ("Ta", 2): 0.30, ("W", 2): 0.35, ("Re", 2): 0.40, ("Os", 2): 0.45, ("Ir", 2): 0.50,
    ("Pt", 2): 0.60, ("Au", 2): 0.65,
    ("O", 1): 0.02, ("N", 1): 0.01, ("F", 1): 0.03, ("S", 1): 0.05, ("Cl", 1): 0.07,
    ("Se", 1): 0.25, ("Br", 1): 0.30, ("Te", 1): 0.50, ("I", 1): 0.60,
    ("P", 1): 0.03, ("As", 1): 0.20, ("Sb", 1): 0.40, ("Bi", 1): 1.25,
    ("Ga", 1): 0.10, ("Ge", 1): 0.13, ("Sn", 1): 0.30, ("Pb", 1): 0.90, ("In", 1): 0.25,
    ("Sm", 3): 0.15, ("Gd", 3): 0.20, ("Tb", 3): 0.21, ("Dy", 3): 0.23, ("Ho", 3): 0.24,
    ("Er", 3): 0.26, ("Nd", 3): 0.13, ("Eu", 3): 0.17, ("Ce", 3): 0.09,
}
