"""Demagnetising field by FFT convolution with the Newell (1993) cell-averaged tensor.

H_a(r_i) = - sum_j N_ab(r_i - r_j) M_b(r_j)

The tensor is computed with the 27-point second-difference formula of Newell, Williams and
Dunlop (J. Geophys. Res. 98, 9551) for near cells and the point-dipole limit for far cells
(where Newell's formula loses precision to cancellation).  Open boundaries use zero padding
to 2n; periodic axes use an n-point circular convolution with the tensor summed over
``pbc_images`` images (macro-geometry approach of OOMMF/mumax).
"""
from __future__ import annotations

import itertools

import numpy as np
import scipy.fft as sfft


def _asinh_ratio(a, b):
    return np.arcsinh(a / np.maximum(b, 1e-300))


def _f(x, y, z):
    x, y, z = np.abs(x), np.abs(y), np.abs(z)
    x2, y2, z2 = x * x, y * y, z * z
    R = np.sqrt(x2 + y2 + z2)
    with np.errstate(invalid="ignore", divide="ignore"):
        t1 = 0.5 * y * (z2 - x2) * _asinh_ratio(y, np.sqrt(x2 + z2))
        t2 = 0.5 * z * (y2 - x2) * _asinh_ratio(z, np.sqrt(x2 + y2))
        t3 = np.where(x * y * z > 0, x * y * z * np.arctan(y * z / np.maximum(x * R, 1e-300)), 0.0)
        t4 = (2 * x2 - y2 - z2) * R / 6.0
    return t1 + t2 - t3 + t4


def _g(x, y, z):
    sgn = np.sign(x) * np.sign(y)
    x, y, z = np.abs(x), np.abs(y), np.abs(z)
    x2, y2, z2 = x * x, y * y, z * z
    R = np.sqrt(x2 + y2 + z2)
    with np.errstate(invalid="ignore", divide="ignore"):
        t1 = x * y * z * _asinh_ratio(z, np.sqrt(x2 + y2))
        t2 = y / 6.0 * (3 * z2 - y2) * _asinh_ratio(x, np.sqrt(y2 + z2))
        t3 = x / 6.0 * (3 * z2 - x2) * _asinh_ratio(y, np.sqrt(x2 + z2))
        t4 = np.where(z > 0, z2 * z / 6.0 * np.arctan(x * y / np.maximum(z * R, 1e-300)), 0.0)
        t5 = np.where(y > 0, z * y2 / 2.0 * np.arctan(x * z / np.maximum(y * R, 1e-300)), 0.0)
        t6 = np.where(x > 0, z * x2 / 2.0 * np.arctan(y * z / np.maximum(x * R, 1e-300)), 0.0)
        t7 = x * y * R / 3.0
    return sgn * (t1 + t2 + t3 - t4 - t5 - t6 - t7)


_W = {-1: -1.0, 0: 2.0, 1: -1.0}


def _newell(fun, X, Y, Z, dx, dy, dz):
    out = np.zeros_like(X)
    for ex, ey, ez in itertools.product((-1, 0, 1), repeat=3):
        w = _W[ex] * _W[ey] * _W[ez]
        out += w * fun(X + ex * dx, Y + ey * dy, Z + ez * dz)
    return out / (4 * np.pi * dx * dy * dz)


def tensor_at(X, Y, Z, d, far=48.0):
    """Six components (xx, yy, zz, xy, xz, yz) of N at displacements X, Y, Z (m)."""
    dx, dy, dz = d
    R = np.sqrt(X**2 + Y**2 + Z**2)
    near = R <= far * max(d)
    N = np.zeros((6,) + X.shape)
    if np.any(near):
        x, y, z = X[near], Y[near], Z[near]
        N[0][near] = _newell(_f, x, y, z, dx, dy, dz)
        N[1][near] = _newell(_f, y, x, z, dy, dx, dz)
        N[2][near] = _newell(_f, z, y, x, dz, dy, dx)
        N[3][near] = _newell(_g, x, y, z, dx, dy, dz)
        N[4][near] = _newell(_g, x, z, y, dx, dz, dy)
        N[5][near] = _newell(_g, y, z, x, dy, dz, dx)
    fa = ~near
    if np.any(fa):
        x, y, z, r = X[fa], Y[fa], Z[fa], R[fa]
        V = dx * dy * dz
        pre = V / (4 * np.pi * r**5)
        r2 = r * r
        N[0][fa] = pre * (r2 - 3 * x * x)
        N[1][fa] = pre * (r2 - 3 * y * y)
        N[2][fa] = pre * (r2 - 3 * z * z)
        N[3][fa] = -pre * 3 * x * y
        N[4][fa] = -pre * 3 * x * z
        N[5][fa] = -pre * 3 * y * z
    return N


class Demag:
    """Precomputed FFT kernel for a mesh."""

    def __init__(self, mesh, workers=-1, far=48.0, log=None):
        self.mesh = mesh
        self.workers = workers
        n, d = mesh.n, mesh.d
        self.fshape = tuple(n[a] if mesh.pbc[a] else (2 * n[a] if n[a] > 1 else 1) for a in range(3))
        offs = []
        for a in range(3):
            M = self.fshape[a]
            k = np.arange(M)
            o = np.where(k <= M // 2, k, k - M) if mesh.pbc[a] else np.where(k < n[a], k, k - M)
            offs.append(o.astype(float))
        O = np.meshgrid(*offs, indexing="ij")
        imgs = [range(-mesh.pbc_images, mesh.pbc_images + 1) if mesh.pbc[a] else [0] for a in range(3)]
        N = np.zeros((6,) + self.fshape)
        for p in itertools.product(*imgs):
            X = (O[0] + p[0] * n[0]) * d[0]
            Y = (O[1] + p[1] * n[1]) * d[1]
            Z = (O[2] + p[2] * n[2]) * d[2]
            N += tensor_at(X, Y, Z, d, far)
        for a in range(3):          # the (unused) Nyquist offset of open axes carries no coupling
            if not mesh.pbc[a] and self.fshape[a] > 1:
                sl = [slice(None)] * 4
                sl[a + 1] = n[a]
                N[tuple(sl)] = 0.0
        self.N0 = N[:, 0, 0, 0].copy()        # self-demag factors of one cell
        self.K = sfft.rfftn(N, axes=(1, 2, 3), workers=workers)
        if log:
            log(f"  demag kernel {self.fshape}, cell factors Nxx,Nyy,Nzz = {np.round(self.N0[:3], 4).tolist()}")

    def field(self, M):
        """H (A/m), shape (nx,ny,nz,3), for magnetisation M = Ms m (A/m)."""
        n = self.mesh.n
        kw = {} if self.workers is None else dict(workers=self.workers)
        Mk = [sfft.rfftn(M[..., c], s=self.fshape, axes=(0, 1, 2), **kw) for c in range(3)]
        xx, yy, zz, xy, xz, yz = self.K
        Hk = (xx * Mk[0] + xy * Mk[1] + xz * Mk[2],
              xy * Mk[0] + yy * Mk[1] + yz * Mk[2],
              xz * Mk[0] + yz * Mk[1] + zz * Mk[2])
        kw = {} if self.workers is None else dict(workers=self.workers)
        comps = [-sfft.irfftn(Hk[c], s=self.fshape, axes=(0, 1, 2), **kw)[:n[0], :n[1], :n[2]] for c in range(3)]
        return np.stack(comps, axis=-1)
