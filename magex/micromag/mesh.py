"""Finite-difference mesh and geometry masks.

Arrays are indexed (ix, iy, iz[, component]); cell centres are measured from the centre of
the mesh so that shapes such as disks are centred by default.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Mesh:
    n: tuple                         # (nx, ny, nz)
    d: tuple                         # cell size (m)
    pbc: tuple = (False, False, False)
    pbc_images: int = 10             # macro-geometry images summed in the demag tensor along periodic axes

    def __post_init__(self):
        self.n = tuple(int(x) for x in self.n)
        self.d = tuple(float(x) for x in self.d)
        self.pbc = tuple(bool(x) for x in self.pbc)

    @property
    def shape(self):
        return self.n

    @property
    def V(self):
        return self.d[0] * self.d[1] * self.d[2]

    @property
    def size(self):
        return tuple(n * d for n, d in zip(self.n, self.d))

    def coords(self):
        """Cell-centre coordinates X, Y, Z (m), each (nx, ny, nz), origin at the mesh centre."""
        ax = [(np.arange(n) + 0.5) * d - 0.5 * n * d for n, d in zip(self.n, self.d)]
        return np.meshgrid(*ax, indexing="ij")

    def __repr__(self):
        nm = [f"{x * 1e9:g}" for x in self.d]
        return (f"Mesh {self.n[0]}x{self.n[1]}x{self.n[2]} cells of {nm[0]}x{nm[1]}x{nm[2]} nm^3 "
                f"({self.size[0] * 1e9:g} x {self.size[1] * 1e9:g} x {self.size[2] * 1e9:g} nm^3), pbc={self.pbc}")


def _inside_polygon(x, y, verts):
    from matplotlib.path import Path
    p = Path(np.asarray(verts))
    return p.contains_points(np.column_stack([x.ravel(), y.ravel()])).reshape(x.shape)


def geometry_mask(mesh: Mesh, shape="rectangle", **p):
    """Boolean mask of the cells inside a shape.  Lengths in nm (keys ending in _nm) or m.

    shapes: rectangle (size_nm=[lx,ly] optional), disk (diameter_nm), ellipse (diameters_nm=[a,b]),
            ring (outer_nm, inner_nm), polygon (vertices_nm=[[x,y],...]), triangle (side_nm),
            png (path, threshold, invert): dark pixels = magnetic, image stretched over the xy extent
    common: center_nm=[x,y], z_layers=[k0,k1] (inclusive cell range), rotate_deg
    """
    if shape in ("union", "intersect", "xor", "difference"):
        subs = [geometry_mask(mesh, **dict(q)) for q in p["shapes"]]
        m = subs[0].copy()
        for q in subs[1:]:
            m = {"union": m | q, "intersect": m & q, "xor": m ^ q, "difference": m & ~q}[shape]
        return m
    if "fraction" in p:
        raise ValueError("use volume_fraction() for edge smoothing")
    X, Y, Z = mesh.coords()
    nm = 1e-9
    c = np.array(p.get("center_nm", [0, 0]), float) * nm
    cx, cy = c[0], c[1]
    cz = c[2] if len(c) > 2 else 0.0
    x, y, z = X - cx, Y - cy, Z - cz
    if p.get("rotate_x_deg") or p.get("rotate_y_deg"):
        ax_ = np.deg2rad(p.get("rotate_x_deg", 0.0))
        ay_ = np.deg2rad(p.get("rotate_y_deg", 0.0))
        y, z = np.cos(ax_) * y + np.sin(ax_) * z, -np.sin(ax_) * y + np.cos(ax_) * z
        x, z = np.cos(ay_) * x - np.sin(ay_) * z, np.sin(ay_) * x + np.cos(ay_) * z
    if p.get("rotate_deg"):
        t = np.deg2rad(p["rotate_deg"])
        x, y = np.cos(t) * x + np.sin(t) * y, -np.sin(t) * x + np.cos(t) * y
    if shape in ("rectangle", "box", "film", "full"):
        if "size_nm" in p:
            lx, ly = np.array(p["size_nm"], float) * nm
            m = (np.abs(x) <= lx / 2 + 1e-15) & (np.abs(y) <= ly / 2 + 1e-15)
        else:
            m = np.ones(mesh.n, bool)
    elif shape == "sphere":
        m = x**2 + y**2 + z**2 <= (p["diameter_nm"] * nm / 2) ** 2 * (1 + 1e-12)
    elif shape == "cylinder":
        m = (x**2 + y**2 <= (p["diameter_nm"] * nm / 2) ** 2 * (1 + 1e-12)) & \
            (np.abs(z) <= p.get("height_nm", 1e9) * nm / 2 + 1e-15)
    elif shape == "cuboid":
        lx, ly, lz = np.array(p["size_nm"], float) * nm
        m = (np.abs(x) <= lx / 2 + 1e-15) & (np.abs(y) <= ly / 2 + 1e-15) & (np.abs(z) <= lz / 2 + 1e-15)
    elif shape == "ellipsoid":
        a, b, cc = np.array(p["diameters_nm"], float) * nm / 2
        m = (x / a) ** 2 + (y / b) ** 2 + (z / cc) ** 2 <= 1 + 1e-12
    elif shape == "cone":
        h = p["height_nm"] * nm
        r = p["diameter_nm"] * nm / 2 * np.clip(0.5 - z / h, 0, 1)     # base at z = -h/2, apex at +h/2
        m = (x**2 + y**2 <= r**2) & (np.abs(z) <= h / 2)
    elif shape == "disk":
        r = p["diameter_nm"] * nm / 2
        m = x**2 + y**2 <= r**2 * (1 + 1e-12)
    elif shape == "ellipse":
        a, b = np.array(p["diameters_nm"], float) * nm / 2
        m = (x / a) ** 2 + (y / b) ** 2 <= 1 + 1e-12
    elif shape == "ring":
        r2 = x**2 + y**2
        m = (r2 <= (p["outer_nm"] * nm / 2) ** 2) & (r2 >= (p["inner_nm"] * nm / 2) ** 2)
    elif shape == "triangle":
        s = p["side_nm"] * nm
        h = s * np.sqrt(3) / 2
        m = _inside_polygon(x, y, [[-s / 2, -h / 3], [s / 2, -h / 3], [0, 2 * h / 3]])
    elif shape == "polygon":
        m = _inside_polygon(x, y, np.array(p["vertices_nm"], float) * nm)
    elif shape == "png":
        from PIL import Image
        img = np.asarray(Image.open(p["path"]).convert("L"), float) / 255.0
        img = img[::-1].T                                   # (x, y) with y up
        ix = np.clip(((X / mesh.size[0] + 0.5) * img.shape[0]).astype(int), 0, img.shape[0] - 1)
        iy = np.clip(((Y / mesh.size[1] + 0.5) * img.shape[1]).astype(int), 0, img.shape[1] - 1)
        m = img[ix, iy] < p.get("threshold", 0.5)
        if p.get("invert"):
            m = ~m
    else:
        raise ValueError(f"unknown shape '{shape}'")
    m = np.broadcast_to(m, mesh.n).copy()
    if "z_layers" in p:
        k0, k1 = p["z_layers"]
        zsel = np.zeros(mesh.n[2], bool)
        zsel[k0:k1 + 1] = True
        m &= zsel[None, None, :]
    if p.get("subtract"):
        for sub in p["subtract"]:
            sub = dict(sub)
            m &= ~geometry_mask(mesh, sub.pop("shape"), **sub)
    return m


def volume_fraction(mesh: Mesh, shape, samples=4, **p):
    """Fraction (0..1) of every cell inside the shape from samples^3 sub-cell points (mumax3 EdgeSmooth)."""
    acc = np.zeros(mesh.n)
    off = (np.arange(samples) + 0.5) / samples - 0.5
    for ox in off:
        for oy in off:
            for oz in (off if mesh.n[2] > 1 else [0.0]):
                q = dict(p)
                c = list(np.array(q.get("center_nm", [0, 0, 0]), float)) + [0.0] * 3
                q["center_nm"] = [c[0] - ox * mesh.d[0] * 1e9, c[1] - oy * mesh.d[1] * 1e9, c[2] - oz * mesh.d[2] * 1e9]
                acc += geometry_mask(mesh, shape, **q)
    return acc / (samples**2 * (samples if mesh.n[2] > 1 else 1))
