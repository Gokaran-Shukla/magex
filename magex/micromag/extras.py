"""Micromagnetic extras: Oersted fields, grains/disorder, exchange bias, parameter maps.

Oersted field of a current density J(r) (A/m^2) on the mesh (cells of any region, including
non-magnetic ones such as a heavy-metal underlayer), by FFT convolution of the Biot-Savart kernel
(cell-centre point approximation, self term zero):

    B(r) = (mu0 / 4 pi) sum_j V J_j x (r - r_j) / |r - r_j|^3

Grains: Voronoi tessellation in the film plane with random easy axes (cone spread), Ku and Ms
dispersion, and reduced exchange across grain boundaries (gb_scale).

Exchange bias: unidirectional pinning E = -J_eb m.u per area of the pinned layer, i.e. a static
field B_eb = J_eb / (Ms t) along u on the cells of the region.
"""
from __future__ import annotations

import numpy as np

from .constants import MU0


def oersted_field(mesh, J):
    """B (T) on all cells from current density J (nx,ny,nz,3) A/m^2 (open boundaries)."""
    n, d = mesh.n, mesh.d
    shape = tuple(2 * k if k > 1 else 1 for k in n)
    offs = []
    for a in range(3):
        M = shape[a]
        k = np.arange(M)
        offs.append(np.where(k < n[a], k, k - M).astype(float) * d[a])
    X, Y, Z = np.meshgrid(*offs, indexing="ij")
    R = np.sqrt(X**2 + Y**2 + Z**2)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv3 = np.where(R > 0, 1 / R**3, 0.0)
    K = np.stack([X * inv3, Y * inv3, Z * inv3])            # (3, ...) vector kernel r/|r|^3
    Kf = [np.fft.rfftn(K[c], s=shape) for c in range(3)]
    Jf = [np.fft.rfftn(J[..., c], s=shape) for c in range(3)]
    pre = MU0 / (4 * np.pi) * mesh.V
    # (J x r)_a = eps_abc J_b r_c  ->  B = pre * (J * K) cross-convolution
    B = np.zeros(n + (3,))
    for a, (b, c) in enumerate(((1, 2), (2, 0), (0, 1))):
        conv = Jf[b] * Kf[c] - Jf[c] * Kf[b]
        B[..., a] = pre * np.fft.irfftn(conv, s=shape, axes=(0, 1, 2))[:n[0], :n[1], :n[2]]
    return B


def voronoi_grains(mesh, size_nm, seed=1):
    """Grain id per cell from a 2D Voronoi tessellation with mean grain size size_nm."""
    rng = np.random.default_rng(seed)
    X, Y, Z = mesh.coords()
    Lx, Ly = mesh.size[0], mesh.size[1]
    ng = max(1, int(round(Lx * Ly / (size_nm * 1e-9) ** 2)))
    seeds = np.column_stack([rng.uniform(-Lx / 2, Lx / 2, ng), rng.uniform(-Ly / 2, Ly / 2, ng)])
    from scipy.spatial import cKDTree
    pts = np.column_stack([X[..., 0].ravel(), Y[..., 0].ravel()])
    if mesh.pbc[0] or mesh.pbc[1]:
        tiles = [(i, j) for i in ((-1, 0, 1) if mesh.pbc[0] else (0,)) for j in ((-1, 0, 1) if mesh.pbc[1] else (0,))]
        allseeds = np.concatenate([seeds + [i * Lx, j * Ly] for i, j in tiles])
        ids = np.tile(np.arange(ng), len(tiles))
        _, k = cKDTree(allseeds).query(pts)
        g = ids[k]
    else:
        _, g = cKDTree(seeds).query(pts)
    g = g.reshape(mesh.n[:2])
    return np.repeat(g[:, :, None], mesh.n[2], axis=2), ng


def apply_grains(system, size_nm, Ku_spread=0.0, Ms_spread=0.0, axis_spread_deg=0.0, gb_scale=1.0, seed=1):
    rng = np.random.default_rng(seed + 1)
    g, ng = voronoi_grains(system.mesh, size_nm, seed)
    system.grain = g
    system.gb_scale = float(gb_scale)
    fK = 1 + Ku_spread * rng.normal(size=ng)
    fM = 1 + Ms_spread * rng.normal(size=ng)
    mag = system.magnetic
    system.Ku1[mag] *= np.clip(fK[g], 0, None)[mag]
    system.Ms[mag] *= np.clip(fM[g], 0.05, None)[mag]
    if axis_spread_deg:
        th = np.deg2rad(axis_spread_deg) * np.abs(rng.normal(size=ng))
        ph = rng.uniform(0, 2 * np.pi, ng)
        base = system.anisU[mag].mean(0)
        base = base / np.linalg.norm(base)
        a = np.array([1.0, 0, 0]) if abs(base[0]) < 0.9 else np.array([0, 1.0, 0])
        e1 = np.cross(base, a)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(base, e1)
        axes = (np.cos(th)[:, None] * base + np.sin(th)[:, None] * (np.cos(ph)[:, None] * e1 + np.sin(ph)[:, None] * e2))
        U = axes[g]
        system.anisU[mag] = U[mag]
    system.invalidate()
    return ng


def exchange_bias_field(system, mask, J_eb, direction, thickness=None):
    u = np.asarray(direction, float)
    u = u / np.linalg.norm(u)
    t = thickness or system.mesh.n[2] * system.mesh.d[2]
    Ms = system.Ms
    inv = np.divide(1.0, Ms, out=np.zeros_like(Ms), where=Ms > 0)
    B = np.zeros(system.mesh.n + (3,))
    B[mask] = (J_eb / t * inv[mask])[:, None] * u[None, :]
    return B


def load_map(path, shape):
    """Scalar map (nx,ny,nz) from .npy, a 2D .npy broadcast along z, or an OVF file (first component)."""
    if str(path).endswith(".npy"):
        a = np.load(path)
    else:
        from .io import read_ovf
        a = read_ovf(path)[0][..., 0]
    if a.ndim == 2:
        a = np.repeat(a[:, :, None], shape[2], axis=2)
    if a.shape != tuple(shape):
        raise ValueError(f"map {path} has shape {a.shape}, mesh is {tuple(shape)}")
    return a
