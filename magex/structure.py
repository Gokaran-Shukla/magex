"""Atomistic structure builder and initial spin textures (VAMPIRE/UppASD/Spirit-style system set-up).

``apply_structure(system, cfg)`` edits a freshly built ``SpinSystem`` according to a ``[structure]`` table:

  shape        "sphere" | "cylinder" | "cube" | "ellipsoid" | "truncated_octahedron" | "film" | "none"
  size         radius (A) for sphere/cylinder/truncated octahedron, [a, b, c] semi-axes for ellipsoid,
               edge for cube; height = cylinder height (A); axis = cylinder axis (default z)
  center       [x, y, z] (A); default: centre of the supercell
  core_shell   {radius = R_core (A), core = "core", shell = "shell"}       material assignment by radius
  layers       [{material = "Co", thickness = 3}, ...]                     multilayer along z (unit cells)
  alloy        {host = "A", guest = "B", fraction = 0.3, seed = 1}         random substitution
  vacancies    fraction of sites removed at random (or {fraction, seed, material})
  roughness    {amplitude = 1.0 (cells), correlation = 10 (A), seed = 1, surface = "top"}
  pinned       [{region = "bottom" | "top" | "xmin" | "xmax" | "ymin" | "ymax" | "material:<name>",
                 layers = 1, direction = [0, 0, 1]}]
  materials    [{name, moment_scale, anisotropy_scale, Ku = [K (meV), axis], Kc, cubic_axes, K2, K3,
                 K_axis, ks (Neel surface anisotropy, meV), T (site temperature)}]
  exchange_scale  n_mat x n_mat matrix multiplying J_ij between materials (interface exchange)
  neel_surface    ks (meV) for all under-coordinated sites (Neel pair anisotropy, VAMPIRE convention)
  dipolar      true | {images = 4}                                         exact FFT dipole-dipole field
  quadruplets  {K = meV, list = [[i, j, k, l], ...]}                         four-spin terms on explicit sites
  temperature_profile  {axis = "x", T0 = 10, T1 = 300}  or {spot = [x, y, z], sigma = A, T0, T1}

``init_texture(system, kind, ...)`` returns spins for "spiral", "skyrmion(s)", "bubble", "hopfion", "domain".
"""
from __future__ import annotations

import numpy as np

from .spinmodel import normalize

_AX = {"x": 0, "y": 1, "z": 2}


def _center(system, c):
    if c is not None:
        return np.asarray(c, float)
    P = system.positions
    return 0.5 * (P.min(0) + P.max(0))


def shape_mask(system, shape, size=None, center=None, height=None, axis="z"):
    """Boolean mask of sites inside a particle shape (positions in A)."""
    if shape in (None, "none", "film", "bulk"):
        return np.ones(system.N, bool)
    r = system.positions - _center(system, center)
    if shape == "sphere":
        return np.linalg.norm(r, axis=1) <= float(size)
    if shape == "cylinder":
        a = _AX[axis] if isinstance(axis, str) else int(axis)
        perp = np.delete(r, a, axis=1)
        m = np.linalg.norm(perp, axis=1) <= float(size)
        if height is not None:
            m &= np.abs(r[:, a]) <= 0.5 * float(height)
        return m
    if shape == "cube":
        return np.all(np.abs(r) <= 0.5 * float(size), axis=1)
    if shape == "ellipsoid":
        s = np.asarray(size, float) * np.ones(3)
        return np.sum((r / s) ** 2, axis=1) <= 1.0
    if shape == "truncated_octahedron":
        R = float(size)
        return np.all(np.abs(r) <= R, axis=1) & (np.abs(r).sum(1) <= 1.5 * R)
    raise ValueError(f"unknown shape '{shape}'")


def _mat_index(system, name):
    if name not in system.material_names:
        system.material_names.append(name)
    return system.material_names.index(name)


def _region(system, spec):
    reg = spec.get("region", "bottom")
    n = int(spec.get("layers", 1))
    if reg.startswith("material:"):
        return system.material == system.material_names.index(reg.split(":", 1)[1])
    c = system.cells[system.cell_index]
    ax = {"bottom": 2, "top": 2, "xmin": 0, "xmax": 0, "ymin": 1, "ymax": 1}[reg]
    lo, hi = c[:, ax].min(), c[:, ax].max()
    if reg in ("bottom", "xmin", "ymin"):
        return c[:, ax] < lo + n
    return c[:, ax] > hi - n


def apply_structure(system, cfg, log=None):
    """Apply a [structure] table to a SpinSystem (in place); returns the system."""
    if not cfg:
        return system
    log = log or (lambda *a: None)
    from . import terms
    st = dict(cfg)
    rng = np.random.default_rng(int(st.get("seed", 12345)))
    N0 = system.N
    # ------------------------------------------------------------------ materials (before carving)
    mats = st.get("materials", [])
    for m in mats:
        _mat_index(system, m["name"])
    if "layers" in st:
        cz = system.cells[system.cell_index][:, 2]
        z0 = 0
        for lay in st["layers"]:
            t = int(lay["thickness"])
            system.material[(cz >= z0) & (cz < z0 + t)] = _mat_index(system, lay["material"])
            z0 += t
    if "core_shell" in st:
        cs = st["core_shell"]
        r = np.linalg.norm(system.positions - _center(system, st.get("center")), axis=1)
        core = r <= float(cs["radius"])
        system.material[core] = _mat_index(system, cs.get("core", "core"))
        system.material[~core] = _mat_index(system, cs.get("shell", "shell"))
    if "alloy" in st:
        al = st["alloy"]
        host = system.material == _mat_index(system, al.get("host", "bulk"))
        g = _mat_index(system, al["guest"])
        r2 = np.random.default_rng(al.get("seed", 1))
        system.material[host & (r2.random(system.N) < float(al["fraction"]))] = g
    # ------------------------------------------------------------------ carve
    keep = shape_mask(system, st.get("shape"), st.get("size"), st.get("center"), st.get("height"),
                      st.get("axis", "z"))
    if "roughness" in st:
        ro = st["roughness"]
        r2 = np.random.default_rng(ro.get("seed", 1))
        P = system.positions
        xy = P[:, :2]
        nseed = max(8, int(np.prod(np.ptp(xy, 0) + 1) / max(float(ro.get("correlation", 10.0)), 1e-3) ** 2))
        cen = xy.min(0) + r2.random((nseed, 2)) * (np.ptp(xy, 0) + 1e-9)
        h = r2.integers(0, int(np.ceil(ro.get("amplitude", 1.0))) + 1, nseed)
        d = ((xy[:, None, :] - cen[None]) ** 2).sum(-1)
        depth = h[np.argmin(d, 1)]                         # Voronoi islands of random height
        cz = system.cells[system.cell_index][:, 2]
        if ro.get("surface", "top") == "top":
            keep &= cz <= cz.max() - depth
        else:
            keep &= cz >= cz.min() + depth
    if "vacancies" in st:
        v = st["vacancies"]
        v = {"fraction": v} if not isinstance(v, dict) else v
        r2 = np.random.default_rng(v.get("seed", 2))
        cand = np.ones(system.N, bool) if "material" not in v else \
            system.material == system.material_names.index(v["material"])
        keep &= ~(cand & (r2.random(system.N) < float(v["fraction"])))
    system.carve(keep)
    if system.N < N0:
        log(f"  structure: {N0 - system.N} of {N0} sites removed ({st.get('shape', 'film')}"
            f"{', vacancies' if 'vacancies' in st else ''}{', roughness' if 'roughness' in st else ''}) -> N = {system.N}")
    # ------------------------------------------------------------------ per-material properties
    nm = len(system.material_names)
    Kc = np.zeros(system.N)
    K2 = np.zeros(system.N)
    K3 = np.zeros(system.N)
    cubic_axes, K_axis = None, (0, 0, 1)
    for m in mats:
        sel = system.material == system.material_names.index(m["name"])
        if "moment_scale" in m:
            system.mu[sel] *= float(m["moment_scale"])
        if "anisotropy_scale" in m:
            system.A[sel] *= float(m["anisotropy_scale"])
        if "Ku" in m:                                       # extra uniaxial: E = -K (e.u)^2
            K, ax = m["Ku"][0], np.asarray(m["Ku"][1], float)
            ax = ax / np.linalg.norm(ax)
            system.A[sel] -= float(K) * np.outer(ax, ax)[None]
        if "Kc" in m:
            Kc[sel] = float(m["Kc"])
            cubic_axes = m.get("cubic_axes", cubic_axes)
        if "K2" in m or "K3" in m:
            K2[sel] = float(m.get("K2", 0.0))
            K3[sel] = float(m.get("K3", 0.0))
            K_axis = m.get("K_axis", K_axis)
        if "ks" in m:
            _neel_surface(system, float(m["ks"]), sel)
        if "alpha" in m:
            if getattr(system, "alpha_site", None) is None:
                system.alpha_site = np.full(system.N, np.nan)
            system.alpha_site[sel] = float(m["alpha"])
        if "T" in m:
            if system.site_T is None:
                system.site_T = np.full(system.N, np.nan)
            system.site_T[sel] = float(m["T"])
    if "exchange_scale" in st:
        f = np.ones((nm, nm))
        E = np.asarray(st["exchange_scale"], float)
        f[: E.shape[0], : E.shape[1]] = E
        system.scale_exchange(f)
        log(f"  structure: interface exchange scaling {E.tolist()} over materials {system.material_names}")
    if "neel_surface" in st:
        n = _neel_surface(system, float(st["neel_surface"]), np.ones(system.N, bool))
        log(f"  structure: Neel surface anisotropy ks = {st['neel_surface']} meV on {n} surface sites")
    system.has_A = bool(np.max(np.abs(system.A)) > 0)
    if np.any(Kc):
        system.terms.append(terms.CubicAnisotropy(Kc, cubic_axes))
    if np.any(K2) or np.any(K3):
        system.terms.append(terms.UniaxialHigher(K2, K3, K_axis))
    if "quadruplets" in st:
        q = st["quadruplets"]
        system.terms.append(terms.Quadruplets(q["list"], q["K"], system.N))
    # ------------------------------------------------------------------ pinning, temperature
    for p in st.get("pinned", []):
        sel = _region(system, p)
        if system.pinned is None:
            system.pinned = np.zeros(system.N, bool)
            system.pinned_dir = system.ref.copy()
        system.pinned |= sel
        if "direction" in p:
            d = np.asarray(p["direction"], float)
            system.pinned_dir[sel] = d / np.linalg.norm(d)
        log(f"  structure: {int(sel.sum())} spins pinned ({p.get('region', 'bottom')})")
    tp = st.get("temperature_profile")
    if tp:
        T0, T1 = float(tp["T0"]), float(tp["T1"])
        P = system.positions
        if "spot" in tp:
            r2 = np.sum((P - np.asarray(tp["spot"], float)) ** 2, 1)
            prof = T0 + (T1 - T0) * np.exp(-r2 / (2 * float(tp["sigma"]) ** 2))
        else:
            a = _AX[tp.get("axis", "x")]
            u = (P[:, a] - P[:, a].min()) / max(np.ptp(P[:, a]), 1e-9)
            prof = T0 + (T1 - T0) * u
        if system.site_T is None:
            system.site_T = prof
        else:
            system.site_T = np.where(np.isnan(system.site_T), prof, system.site_T)
    if system.site_T is not None and np.any(np.isnan(system.site_T)):
        system.site_T = None if np.all(np.isnan(system.site_T)) else system.site_T
    # ------------------------------------------------------------------ dipolar (last: needs final sites/moments)
    dip = st.get("dipolar")
    if dip:
        dip = {} if dip is True else dict(dip)
        system.terms.append(terms.Dipolar(system, images=int(dip.get("images", 4))))
        log(f"  structure: dipole-dipole interaction on (FFT, {system.L} grid)")
    return system


def site_temperature(system, T):
    """Effective per-site temperature array (or scalar T if uniform)."""
    sT = getattr(system, "site_T", None)
    if sT is None:
        return T
    return np.where(np.isnan(sT), T, sT)


def gradient_operators(system):
    """Sparse least-squares gradient operators [D_x, D_y, D_z] (1/A): (D_a @ e)_i ~ d e / d x_a at site i,
    from the exchange neighbours (works on any lattice, open boundaries and carved shapes)."""
    import scipy.sparse as sp
    b = system.bonds
    b = b[b[:, 0] != b[:, 1]]
    d = system._minimg(system.positions[b[:, 1]] - system.positions[b[:, 0]])
    r = np.linalg.norm(d, axis=1)
    w = 1.0 / np.maximum(r, 1e-9) ** 2
    M = np.zeros((system.N, 3, 3))
    np.add.at(M, b[:, 0], w[:, None, None] * d[:, :, None] * d[:, None, :])
    Mi = np.linalg.pinv(M, rcond=1e-6)
    c = np.einsum("nab,nb->na", Mi[b[:, 0]], d) * w[:, None]          # weight of (e_j - e_i) for component a
    ops = []
    for a in range(3):
        D = sp.csr_matrix((c[:, a], (b[:, 0], b[:, 1])), shape=(system.N, system.N))
        D = D - sp.diags(np.asarray(D.sum(1)).ravel())
        ops.append(D.tocsr())
    return ops


def _neel_surface(system, ks, sel):
    """Neel pair anisotropy E_i = (ks/2) sum_{j nn} (e_i . r_ij)^2 on under-coordinated sites (folded into A)."""
    b = system.bonds
    P = system.positions
    d = system._minimg(P[b[:, 1]] - P[b[:, 0]])
    r = np.linalg.norm(d, axis=1)
    nn = np.abs(r - r.min()) < 1e-3
    cnt = np.zeros(system.N, int)
    np.add.at(cnt, b[nn, 0], 1)
    zmax = cnt.max()
    surf = (cnt < zmax) & sel
    u = d[nn] / r[nn, None]
    T = np.zeros((system.N, 3, 3))
    np.add.at(T, b[nn, 0], 0.5 * ks * u[:, :, None] * u[:, None, :])
    system.A[surf] += T[surf]
    system.surface = surf
    return int(surf.sum())


# ============================================================================ textures
def _frame(axis):
    n = np.asarray(axis, float)
    n = n / np.linalg.norm(n)
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(n, a)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(n, e1), n


def init_texture(system, kind, axis=(0, 0, 1), seed=None, **k):
    """Initial textures (positions in A).

    spiral   q = [qx, qy, qz] (1/A) or q_frac (reciprocal lattice units), rotation plane normal = axis x q
             (cycloid, default) or "helix" = true
    skyrmion centers = [[x, y], ...] (default: centre), radius = A, polarity = +-1 (core opposite to axis),
             helicity = "auto" (lowest energy, default) | 0, pi (Neel) | +-pi/2 (Bloch), vorticity = 1 (-1: antiskyrmion), wall = A
    bubble   like skyrmion with a wide wall
    hopfion  radius, center, height (3D, preimage tori)
    domain   wall normal "x"|"y"|"z", position (A), width (A), type "neel" | "bloch"
    """
    P = system.positions
    e1, e2, n = _frame(axis)
    c0 = _center(system, k.get("center"))
    if kind == "spiral":
        if "q_frac" in k:
            G = 2 * np.pi * np.linalg.inv(system.uc.cell).T
            q = np.asarray(k["q_frac"], float) @ G
        else:
            q = np.asarray(k.get("q", [0.1, 0, 0]), float)
        qh = q / max(np.linalg.norm(q), 1e-12)
        if k.get("helix", False):
            a1, a2, _ = _frame(qh)
        else:
            a1, a2 = qh, n
        ph = P @ q
        return normalize(np.cos(ph)[:, None] * a2 + np.sin(ph)[:, None] * a1)
    if kind in ("skyrmion", "skyrmions", "bubble") and k.get("helicity", "auto") == "auto":
        # pick the helicity (Neel in/out, Bloch left/right) favoured by the system's DMI
        best = None
        for h in (0.0, np.pi, 0.5 * np.pi, -0.5 * np.pi):
            ek = init_texture(system, kind, axis, seed, **{**k, "helicity": h})
            En = system.energy(ek)
            if best is None or En < best[0]:
                best = (En, ek)
        return best[1]
    if kind in ("skyrmion", "skyrmions", "bubble"):
        centers = k.get("centers", [c0[:2].tolist()])
        R = float(k.get("radius", 10.0))
        w = float(k.get("wall", R / 2 if kind != "bubble" else R / 6))
        pol = float(k.get("polarity", 1))
        hel = float(k.get("helicity", 0.0))
        vort = float(k.get("vorticity", 1))
        e = np.tile(pol * n, (system.N, 1))           # background: +axis for polarity +1, core: -axis
        best = np.full(system.N, np.inf)
        for c in centers:
            c = np.asarray(c, float)
            d = P - (np.r_[c, c0[2]] if len(c) == 2 else c)
            dx, dy = d @ e1, d @ e2
            r = np.hypot(dx, dy)
            sel = (r < 3 * R + 4 * w) & (r < best)
            best[sel] = r[sel]
            th = 2 * np.arctan2(np.sinh(R / w), np.sinh(r / w))       # pi at the core, 0 far away
            ph = vort * np.arctan2(dy, dx) + hel
            v = (np.sin(th) * np.cos(ph))[:, None] * e1 + (np.sin(th) * np.sin(ph))[:, None] * e2 \
                + (pol * np.cos(th))[:, None] * n
            e[sel] = v[sel]
        return normalize(e)
    if kind == "hopfion":
        R = float(k.get("radius", 8.0))
        h = float(k.get("height", R))
        d = P - c0
        x, y, z = d @ e1, d @ e2, (d @ n) * (R / h)
        r2 = (x * x + y * y + z * z) / R**2
        # Hopf map of the stereographic projection: w = (2(x+iy)/R, 2z/R + i(r2-1)) / (1+r2)
        den = 1 + r2
        z1 = (2 * (x + 1j * y) / R) / den
        z2 = (2 * z / R + 1j * (r2 - 1)) / den
        m1 = 2 * np.real(z1 * np.conj(z2))
        m2 = 2 * np.imag(z1 * np.conj(z2))
        m3 = np.abs(z1) ** 2 - np.abs(z2) ** 2
        v = m1[:, None] * e1 + m2[:, None] * e2 - m3[:, None] * n
        return normalize(v)
    if kind == "domain":
        a = _AX[k.get("normal", "x")]
        x0 = float(k.get("position", c0[a]))
        w = float(k.get("width", 5.0))
        th = 2 * np.arctan(np.exp((P[:, a] - x0) / w))
        t = np.zeros(3)
        t[a] = 1.0
        if k.get("type", "neel") == "bloch" or abs(t @ n) > 0.9:
            t = np.cross(n, t) if np.linalg.norm(np.cross(n, t)) > 1e-6 else e1
        t = t - (t @ n) * n
        t /= np.linalg.norm(t)
        return normalize(np.cos(th)[:, None] * n + np.sin(th)[:, None] * t)
    raise ValueError(f"unknown texture '{kind}'")
