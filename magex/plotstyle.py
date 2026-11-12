"""Publication-quality figure style shared by all magex plots.

* vector PDF + 300 dpi PNG for every figure (``savefig``)
* colour-blind-safe Okabe-Ito palette for lines; diverging RdBu for signed components;
  HSL colour wheel (hue = in-plane angle, lightness = out-of-plane component) for textures
* inward ticks, minor ticks, consistent font sizes
"""
from __future__ import annotations

import os

import numpy as np

OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]


def mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    apply()
    return plt


def apply():
    import matplotlib as mpl_
    from cycler import cycler
    mpl_.rcParams.update({
        "font.size": 10, "axes.labelsize": 10.5, "axes.titlesize": 10.5, "legend.fontsize": 8.5,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans", "axes.linewidth": 0.9, "lines.linewidth": 1.5,
        "lines.markersize": 4.5, "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True, "xtick.minor.visible": True, "ytick.minor.visible": True,
        "xtick.major.size": 4, "ytick.major.size": 4, "xtick.minor.size": 2, "ytick.minor.size": 2,
        "axes.prop_cycle": cycler(color=OKABE_ITO), "legend.frameon": False,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "figure.dpi": 110,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def savefig(fig, path, dpi=300, pdf=True):
    """Save ``path`` (PNG) and a vector PDF next to it."""
    root, ext = os.path.splitext(path)
    fig.savefig(root + (ext or ".png"), dpi=dpi)
    if pdf:
        fig.savefig(root + ".pdf")


def hsl_colors(m):
    """RGB colours for unit vectors m (...,3): hue = in-plane angle, lightness = (1+m_z)/2 (vectorised HLS)."""
    m = np.asarray(m, float)
    h = (np.arctan2(m[..., 1], m[..., 0]) / (2 * np.pi)) % 1.0
    l = np.clip(0.5 * (1 + m[..., 2]), 0, 1)
    q = np.where(l < 0.5, l * 2, 1.0)
    p = 2 * l - q

    def chan(t):
        t = t % 1.0
        return np.where(t < 1 / 6, p + (q - p) * 6 * t,
                        np.where(t < 0.5, q, np.where(t < 2 / 3, p + (q - p) * (2 / 3 - t) * 6, p)))
    return np.stack([chan(h + 1 / 3), chan(h), chan(h - 1 / 3)], -1)


def color_wheel(ax):
    """Small inset legend for the HSL colouring."""
    n = 101
    x = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(x, x)
    R = np.sqrt(X**2 + Y**2)
    mz = 1 - 2 * np.clip(R, 0, 1)
    m = np.stack([X, Y, mz], -1)
    rgb = hsl_colors(m)
    rgba = np.concatenate([rgb, (R <= 1)[..., None].astype(float)], -1)
    ax.imshow(rgba, origin="lower", extent=(-1, 1, -1, 1))
    ax.set_axis_off()
    ax.set_aspect("equal")
    for ang, lab in ((0, "+x"), (90, "+y"), (180, "-x"), (270, "-y")):
        t = np.deg2rad(ang)
        ax.text(1.28 * np.cos(t), 1.28 * np.sin(t), lab, ha="center", va="center", fontsize=6.5)
    ax.text(0, -1.75, "centre: $m_z=+1$\nrim: $m_z=-1$", ha="center", va="top", fontsize=6)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-2.3, 1.5)
