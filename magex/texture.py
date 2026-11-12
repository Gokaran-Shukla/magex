"""Real-space spin-texture plots and topological charge for atomistic spin systems."""
from __future__ import annotations

import numpy as np

PLANES = {"xy": (0, 1, 2), "xz": (0, 2, 1), "yz": (1, 2, 0)}


def _mpl():
    from .plotstyle import mpl
    return mpl()


def layer_indices(system, plane="xy", layer=0, tol=0.05):
    """Indices of the spins in one atomic layer normal to the plane; layer may be negative."""
    a, b, n = PLANES[plane]
    z = system.positions[:, n]
    levels = np.unique(np.round(z / tol) * tol)
    lev = levels[layer]
    return np.where(np.abs(z - lev) < tol * 1.01)[0], (a, b, n), len(levels)


def _inplane_supercell_vectors(system, axes):
    a, b, n = axes
    V = (np.array(system.L)[:, None] * system.uc.cell)
    order = np.argsort(np.abs(V[:, n]))
    return V[order[:2]][:, [a, b]], [bool(system.pbc[k]) for k in order[:2]]


def topological_charge(system, e, plane="xy", layer=0):
    """Skyrmion number Q = (1/4pi) sum over triangles of the solid angle (Berg-Luscher).

    The layer is triangulated with a periodic Delaunay construction (3x3 tiling, triangles
    whose centroid lies in the central supercell), so Q is an integer for smooth textures.
    """
    from scipy.spatial import Delaunay
    idx, axes, _ = layer_indices(system, plane, layer)
    a, b, n = axes
    P = system.positions[idx][:, [a, b]]
    S = e[idx]
    if len(idx) < 3:
        return 0.0
    V, per = _inplane_supercell_vectors(system, axes)
    shifts = [(i, j) for i in ((-1, 0, 1) if per[0] else (0,)) for j in ((-1, 0, 1) if per[1] else (0,))]
    Pt = np.concatenate([P + i * V[0] + j * V[1] for i, j in shifts])
    St = np.concatenate([S for _ in shifts])
    try:
        tri = Delaunay(Pt).simplices
    except Exception:
        return 0.0
    cen = Pt[tri].mean(1)
    frac = np.linalg.solve(V.T, (cen - 0.0).T).T
    keep = np.all((frac >= -1e-9) & (frac < 1 - 1e-9), axis=1) if len(shifts) > 1 else np.ones(len(tri), bool)
    tri = tri[keep]
    p1, p2, p3 = Pt[tri[:, 0]], Pt[tri[:, 1]], Pt[tri[:, 2]]
    orient = np.sign((p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0]))
    s1, s2, s3 = St[tri[:, 0]], St[tri[:, 1]], St[tri[:, 2]]
    num = np.einsum("ij,ij->i", s1, np.cross(s2, s3))
    den = 1 + np.einsum("ij,ij->i", s1, s2) + np.einsum("ij,ij->i", s2, s3) + np.einsum("ij,ij->i", s3, s1)
    omega = 2 * np.arctan2(num, den) * orient
    # handedness of the in-plane axes relative to the normal (xz plane is left-handed in (x,z,y))
    hand = 1.0 if plane in ("xy", "yz") else -1.0
    return float(hand * omega.sum() / (4 * np.pi))


def _arrow_subset(P, max_arrows):
    """Pick roughly uniformly spaced sites (grid binning) for the arrow overlay."""
    if len(P) <= max_arrows:
        return np.arange(len(P))
    nb = int(np.sqrt(max_arrows))
    lo, hi = P.min(0), P.max(0)
    g = np.floor((P - lo) / np.maximum(hi - lo, 1e-9) * (nb - 1e-9)).astype(int)
    key = g[:, 0] * (nb + 1) + g[:, 1]
    _, first = np.unique(key, return_index=True)
    return np.sort(first)


def _draw(ax, system, e, plane, layer, max_arrows=900, cbar=True, fig=None, style="mz"):
    """Smooth colour map of the out-of-plane component (or HSL wheel) + arrows of the in-plane part."""
    import matplotlib.tri as mtri
    from .plotstyle import hsl_colors
    idx, (a, b, n), nl = layer_indices(system, plane, layer)
    P = system.positions[idx][:, [a, b]]
    S = e[idx]
    names = "xyz"
    sc = None
    try:
        tri = mtri.Triangulation(P[:, 0], P[:, 1])
        # drop long triangles created across open boundaries / holes
        d = np.median(np.linalg.norm(P[tri.triangles[:, 0]] - P[tri.triangles[:, 1]], axis=1))
        edges = [np.linalg.norm(P[tri.triangles[:, k]] - P[tri.triangles[:, (k + 1) % 3]], axis=1) for k in range(3)]
        tri.set_mask(np.max(edges, axis=0) > 2.5 * d)
    except Exception:
        tri = None
    if style == "hsl":
        rgb = hsl_colors(S[:, [a, b, n]])
        ax.scatter(P[:, 0], P[:, 1], c=rgb, s=max(3, 12000 / max(len(idx), 1)), marker="h", linewidths=0)
    elif tri is not None:
        sc = ax.tripcolor(tri, S[:, n], cmap="RdBu_r", vmin=-1, vmax=1, shading="gouraud", rasterized=True)
    else:
        sc = ax.scatter(P[:, 0], P[:, 1], c=S[:, n], cmap="RdBu_r", vmin=-1, vmax=1, s=12)
    sel = _arrow_subset(P, max_arrows)
    span = max(np.ptp(P[:, 0]), np.ptp(P[:, 1]), 1.0)
    alen = span / np.sqrt(len(sel)) * 0.85
    ax.quiver(P[sel, 0], P[sel, 1], S[sel, a], S[sel, b], angles="xy", scale_units="xy", scale=1.0 / alen,
              width=0.0032, headwidth=3.5, headlength=4, headaxislength=3.5, color="k", alpha=0.8, pivot="middle")
    ax.set_aspect("equal")
    ax.set_xlabel(f"{names[a]} (Å)")
    ax.set_ylabel(f"{names[b]} (Å)")
    ax.minorticks_off()
    if cbar and fig is not None and sc is not None:
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label(f"$e_{names[n]}$")
        cb.set_ticks([-1, -0.5, 0, 0.5, 1])
    return sc


def plot_texture(system, e, path, plane="xy", layer=0, title=None, style="mz"):
    from .plotstyle import savefig
    plt = _mpl()
    idx, axes, nl = layer_indices(system, plane, layer)
    P = system.positions[idx]
    a, b, _ = axes
    w = np.ptp(P[:, a]) or 1
    h = np.ptp(P[:, b]) or 1
    fig, ax = plt.subplots(figsize=(min(9, 2.5 + 5 * w / max(w, h)), min(8, 1.2 + 5 * h / max(w, h))))
    _draw(ax, system, e, plane, layer, fig=fig, style=style)
    ax.set_title((title or "spin texture") + (f"   (layer {layer % nl + 1}/{nl})" if nl > 1 else ""))
    savefig(fig, path)
    plt.close(fig)


def plot_compare(system, spins, titles, path, plane="xy", layer=0):
    from .plotstyle import savefig
    plt = _mpl()
    idx, (a, b, _), _ = layer_indices(system, plane, layer)
    P = system.positions[idx]
    asp = (np.ptp(P[:, b]) or 1) / (np.ptp(P[:, a]) or 1)
    fig, axs = plt.subplots(1, len(spins), figsize=(4.6 * len(spins) + 0.8, 4.6 * min(asp, 1.4) + 0.6),
                            constrained_layout=True)
    for k, (ax, e, t) in enumerate(zip(np.atleast_1d(axs), spins, titles)):
        sc = _draw(ax, system, e, plane, layer, cbar=False)
        ax.set_title(f"({chr(97 + k)}) {t}")
        if k:
            ax.set_ylabel("")
    cb = fig.colorbar(sc, ax=list(np.atleast_1d(axs)), fraction=0.025, pad=0.01, shrink=0.85)
    cb.set_label("$e_{%s}$" % "xyz"[PLANES[plane][2]])
    savefig(fig, path)
    plt.close(fig)


def plot_timeseries(table, path, title="LLG"):
    from .plotstyle import savefig, OKABE_ITO
    plt = _mpl()
    t = table[:, 0]
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 2.9), constrained_layout=True)
    axs[0].plot(t, table[:, 1], color=OKABE_ITO[2])
    axs[0].set_ylabel("$E/N$ (meV)")
    for k, l in enumerate("xyz"):
        axs[1].plot(t, table[:, 2 + k], color=OKABE_ITO[k], label=f"$m_{l}$")
    axs[1].plot(t, table[:, 5], color="k", ls="--", lw=1, label="order")
    axs[1].set_ylim(-1.05, 1.05)
    axs[1].legend(ncol=4, loc="lower right")
    axs[2].semilogy(t, np.maximum(table[:, 6], 1e-16), color=OKABE_ITO[4])
    axs[2].set_ylabel(r"max $|\mathbf{e}\times\mathbf{B}|$ (T)")
    for k, ax in enumerate(axs):
        ax.set_xlabel("$t$ (ps)")
        ax.set_title(f"({chr(97 + k)})", loc="left")
    fig.suptitle(title)
    savefig(fig, path)
    plt.close(fig)


def make_gif(system, snaps, path, plane="xy", layer=0, fps=6):
    plt = _mpl()
    from matplotlib.animation import FuncAnimation, PillowWriter
    fig, ax = plt.subplots(figsize=(5.5, 5))

    def frame(k):
        ax.clear()
        t, e = snaps[k]
        _draw(ax, system, e, plane, layer, cbar=False)
        ax.set_title(f"t = {t:.2f} ps")

    anim = FuncAnimation(fig, frame, frames=len(snaps))
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
