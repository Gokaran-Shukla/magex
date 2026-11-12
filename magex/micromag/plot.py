"""Figures for micromagnetic states, time traces and hysteresis loops (PDF + 300 dpi PNG)."""
from __future__ import annotations

import numpy as np

from ..plotstyle import OKABE_ITO, color_wheel, hsl_colors, mpl, savefig


def topological_charge(system, m, layer=0):
    """Q = (1/4 pi) sum of Berg-Luscher solid angles over the two triangles of each plaquette."""
    mm = m[:, :, layer]
    mag = system.magnetic[:, :, layer]
    px, py = system.mesh.pbc[0], system.mesh.pbc[1]
    a = mm
    b = np.roll(mm, -1, 0)
    c = np.roll(np.roll(mm, -1, 0), -1, 1)
    d = np.roll(mm, -1, 1)
    ok = mag & np.roll(mag, -1, 0) & np.roll(np.roll(mag, -1, 0), -1, 1) & np.roll(mag, -1, 1)
    if not px:
        ok[-1, :] = False
    if not py:
        ok[:, -1] = False

    def omega(s1, s2, s3):
        num = np.einsum("...i,...i->...", s1, np.cross(s2, s3))
        den = 1 + (s1 * s2).sum(-1) + (s2 * s3).sum(-1) + (s3 * s1).sum(-1)
        return 2 * np.arctan2(num, den)

    w = omega(a, b, c) + omega(a, c, d)
    return float(np.sum(w[ok]) / (4 * np.pi))


def _extent(mesh):
    return (-mesh.size[0] / 2 * 1e9, mesh.size[0] / 2 * 1e9, -mesh.size[1] / 2 * 1e9, mesh.size[1] / 2 * 1e9)


def _draw_state(ax, system, m, layer=0, style="hsl", arrows=24, fig=None, cbar=True):
    mesh = system.mesh
    mm = m[:, :, layer]
    mag = system.magnetic[:, :, layer]
    ext = _extent(mesh)
    im = None
    if style == "hsl":
        rgb = hsl_colors(mm)
        rgba = np.concatenate([rgb, mag[..., None].astype(float)], -1)
        ax.imshow(np.transpose(rgba, (1, 0, 2)), origin="lower", extent=ext, interpolation="bilinear")
    else:
        comp = {"mx": 0, "my": 1, "mz": 2}[style]
        data = np.where(mag, mm[..., comp], np.nan)
        im = ax.imshow(data.T, origin="lower", extent=ext, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="bilinear")
        if cbar and fig is not None:
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cb.set_label(f"$m_{style[1]}$")
    nx, ny = mesh.n[:2]
    sx = max(1, int(np.ceil(max(nx, ny) / arrows)))
    ix = np.arange(sx // 2, nx, sx)
    iy = np.arange(sx // 2, ny, sx)
    X, Y = np.meshgrid(ix, iy, indexing="ij")
    keep = mag[X, Y]
    xc = (X + 0.5) * mesh.d[0] * 1e9 - mesh.size[0] / 2 * 1e9
    yc = (Y + 0.5) * mesh.d[1] * 1e9 - mesh.size[1] / 2 * 1e9
    U, V = mm[X, Y, 0], mm[X, Y, 1]
    L = sx * min(mesh.d[:2]) * 1e9 * 0.85
    ax.quiver(xc[keep], yc[keep], U[keep], V[keep], angles="xy", scale_units="xy", scale=1 / L, pivot="middle",
              color="k", alpha=0.8, width=0.0035, headwidth=3.5, headlength=4, headaxislength=3.5)
    if np.any(mag) and np.nanmax(np.abs(mm[..., 2][mag])) > 0.5 and np.nanmin(mm[..., 2][mag]) < -0.2:
        try:
            xs = (np.arange(nx) + 0.5) * mesh.d[0] * 1e9 - mesh.size[0] / 2 * 1e9
            ys = (np.arange(ny) + 0.5) * mesh.d[1] * 1e9 - mesh.size[1] / 2 * 1e9
            ax.contour(xs, ys, np.where(mag, mm[..., 2], np.nan).T, levels=[0.0], colors="w", linewidths=0.8)
        except Exception:
            pass
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_aspect("equal")
    ax.set_xlabel("$x$ (nm)")
    ax.set_ylabel("$y$ (nm)")
    ax.minorticks_off()
    return im


def plot_state(system, m, path, layer=0, title=None, style="both"):
    """HSL panel (hue = in-plane angle, lightness = m_z) and m_z panel side by side."""
    plt = mpl()
    mesh = system.mesh
    asp = mesh.size[1] / mesh.size[0]
    styles = ["hsl", "mz"] if style == "both" else [style]
    h = max(2.4, 4.0 * min(asp, 1.5)) + 0.6
    fig = plt.figure(figsize=(4.4 * len(styles) + 1.6, h), constrained_layout=True)
    ratios = []
    for st in styles:
        ratios += [1.0, 0.16] if st == "hsl" else [1.0]
    gs = fig.add_gridspec(1, len(ratios), width_ratios=ratios)
    col = 0
    axs = []
    for k, st in enumerate(styles):
        ax = fig.add_subplot(gs[0, col])
        axs.append(ax)
        _draw_state(ax, system, m, layer, st, fig=fig)
        lab = {"hsl": "in-plane angle (hue), $m_z$ (lightness)", "mz": "$m_z$", "mx": "$m_x$", "my": "$m_y$"}[st]
        ax.set_title(f"({chr(97 + k)}) {lab}", fontsize=9)
        if k:
            ax.set_ylabel("")
        col += 1
        if st == "hsl":
            wax = fig.add_subplot(gs[0, col])
            color_wheel(wax)
            col += 1
    Q = topological_charge(system, m, layer)
    mm = system.average(m)
    sup = (title + "\n" if title else "") + \
        f"$\\langle m\\rangle$ = ({mm[0]:.3f}, {mm[1]:.3f}, {mm[2]:.3f}),  $Q$ = {Q:+.2f}"
    fig.suptitle(sup, fontsize=9.5)
    savefig(fig, path)
    plt.close(fig)
    return Q


def plot_table(table, path, title=None):
    plt = mpl()
    t = table[:, 0] * 1e9
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 2.9), constrained_layout=True)
    for k, l in enumerate("xyz"):
        axs[0].plot(t, table[:, 1 + k], color=OKABE_ITO[k], label=f"$\\langle m_{l}\\rangle$")
    axs[0].set_ylim(-1.05, 1.05)
    axs[0].legend(ncol=3, loc="best")
    labels = ["total", "exchange", "DMI", "anisotropy", "demag", "Zeeman"]
    E = table[:, 4:10] * 1e18
    for k, l in enumerate(labels):
        if np.any(E[:, k]):
            axs[1].plot(t, E[:, k], color=OKABE_ITO[k] if k else "k", lw=2 if k == 0 else 1.2, label=l)
    axs[1].set_ylabel("$E$ (aJ)")
    axs[1].legend(ncol=2, fontsize=7)
    axs[2].semilogy(t, np.maximum(table[:, 10], 1e-12), color=OKABE_ITO[4])
    axs[2].set_ylabel(r"max $|\mathbf{m}\times\mathbf{B}|$ (T)")
    for k, ax in enumerate(axs):
        ax.set_xlabel("$t$ (ns)")
        ax.set_title(f"({chr(97 + k)})", loc="left")
    if title:
        fig.suptitle(title)
    savefig(fig, path)
    plt.close(fig)


def plot_hysteresis(loop, path, label="m.u", title=None, Bunit="mT"):
    plt = mpl()
    s = 1e3 if Bunit == "mT" else 1.0
    fig, ax = plt.subplots(figsize=(4.2, 3.4), constrained_layout=True)
    n = len(loop)
    half = (n + 1) // 2
    ax.plot(loop[:half, 0] * s, loop[:half, 1], "-o", ms=3, color=OKABE_ITO[0], label="descending")
    if n > half:
        ax.plot(loop[half - 1:, 0] * s, loop[half - 1:, 1], "-s", ms=3, color=OKABE_ITO[1], label="ascending")
    ax.axhline(0, color="0.7", lw=0.6)
    ax.axvline(0, color="0.7", lw=0.6)
    ax.set_xlabel(f"$B$ ({Bunit})")
    ax.set_ylabel(r"$\langle\mathbf{m}\rangle\cdot\hat{\mathbf{u}}$")
    ax.set_ylim(-1.08, 1.08)
    ax.legend(loc="lower right")
    if title:
        ax.set_title(title)
    savefig(fig, path)
    plt.close(fig)


def make_movie(system, frames, path, layer=0, fps=8, style="hsl"):
    plt = mpl()
    from matplotlib.animation import FuncAnimation, PillowWriter
    mesh = system.mesh
    asp = mesh.size[1] / mesh.size[0]
    fig, ax = plt.subplots(figsize=(5, max(2.5, 5 * min(asp, 1.4))))

    def draw(k):
        ax.clear()
        t, m = frames[k]
        _draw_state(ax, system, m, layer, style, cbar=False)
        ax.set_title(f"t = {t * 1e9:.3f} ns")

    FuncAnimation(fig, draw, frames=len(frames)).save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
