"""Figures for the magex Beamer deck (data from the validation runs).

Set RUNS to the folder holding your magex runs (fe/, srmno3/, cri3/ with results.json) and
OUT to the output folder; the script expects the magex repository root on sys.path.
"""
import json, sys, itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "tests"))

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.9, "font.family": "DejaVu Sans", "savefig.dpi": 220,
                     "legend.frameon": False})
NAVY, RED, BLUE, GREEN, ORANGE, PURPLE, TEAL, GOLD, GREY = (
    "#14203A", "#C0392B", "#2E6DB4", "#2E8B57", "#E67E22", "#7D3C98", "#138D90", "#B7950B", "#7F8C8D")
OUT = os.path.join(HERE, "figs") + os.sep
R = os.environ.get("RUNS", "runs") + os.sep

def save(fig, name):
    fig.tight_layout(); fig.savefig(OUT + name + ".pdf"); fig.savefig(OUT + name + ".png"); plt.close(fig)

# ---------------------------------------------------------------- 1 contour
from magex.contour import ozaki_poles, fermi_ozaki, KB_EV
fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.4))
kT = KB_EV * 300
zp, Rp = ozaki_poles(60)
ax[0].axhline(0, color="k", lw=0.8)
th = np.linspace(np.pi, 0, 200); emin, ef = -8, 0
ax[0].plot((emin + ef) / 2 + (ef - emin) / 2 * np.cos(th), (ef - emin) / 2 * np.sin(th), color=GREY, lw=1.5,
           ls="--", label="semicircle (T = 0)")
ax[0].scatter(np.zeros(20), zp[:20] * kT, s=18, color=RED, zorder=5, label="Ozaki poles $\\mu+iz_pk_BT$")
ax[0].scatter(np.zeros(20), [ (2*n+1)*np.pi*kT for n in range(20)], s=10, marker="x", color=BLUE, label="Matsubara $(2n{+}1)\\pi k_BT$")
for e in np.linspace(-7, 2, 16):
    ax[0].plot([e], [0], "|", color=GREEN, ms=10)
ax[0].set_xlim(-8.5, 2.5); ax[0].set_ylim(-0.3, 4.6)
ax[0].set_xlabel("Re $z$ (eV)"); ax[0].set_ylabel("Im $z$ (eV)")
ax[0].text(-7.8, 0.15, "band energies $\\varepsilon_{n\\mathbf{k}}$", color=GREEN, fontsize=9)
ax[0].legend(loc="upper left", fontsize=8.5)
ax[0].set_title("integration nodes in the upper half plane", fontsize=11)
xs = np.linspace(-60, 60, 4001); exact = 0.5 * (1 - np.tanh(xs / 2))
for n, c in [(10, ORANGE), (20, PURPLE), (40, TEAL), (60, RED)]:
    ax[1].semilogy(xs, np.abs(fermi_ozaki(xs, n) - exact) + 1e-17, color=c, lw=1.3, label=f"{n} poles")
ax[1].set_xlabel("$x=(\\varepsilon-\\mu)/k_BT$"); ax[1].set_ylabel("$|f_{\\rm Ozaki}-f_{\\rm FD}|$")
ax[1].set_ylim(1e-17, 1); ax[1].legend(fontsize=8.5, ncol=2, loc="lower left")
ax[1].set_title("Fermi function: exponential convergence", fontsize=11)
save(fig, "contour")

# ---------------------------------------------------------------- 2 sum rule
fig, ax = plt.subplots(figsize=(5.4, 3.3))
labels = ["SrMnO$_3$\nMn", "bcc Fe\n(s,p,d WFs)"]
on = [199.9, 222.5]; we = [17.2, 8.9e-13]
x = np.arange(2)
ax.bar(x - 0.18, on, 0.36, color=GREY, label="on-site rotation (LKAG/TB2J)")
ax.bar(x + 0.18, np.maximum(we, 1e-3), 0.36, color=GREEN, label="weighted partition (magex)")
ax.set_yscale("log"); ax.set_ylim(1e-3, 2e3)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("$|\\sum_{jR}K_{ij}(R)+T_i|$ (meV)")
ax.text(1.18, 2.5e-3, "$9\\times10^{-13}$", ha="center", fontsize=9, color=GREEN)
ax.text(0.18, 25, "17\n(unpolarised O)", ha="center", fontsize=8, color=GREEN)
ax.legend(fontsize=8.5, loc="upper left")
save(fig, "sumrule")

# ---------------------------------------------------------------- 3 Fe J(R)
def shells(path, key="mft"):
    d = json.load(open(path))
    out = {}
    corr = None if key == "mft" else np.array(d["corrections"][key])[0, 0]
    for p in d["pairs"]:
        v = np.trace(np.array(p["J"])) / 3 if corr is None else corr[tuple(p["nu"])]
        out.setdefault(round(p["dist"], 3), v)
    ks = sorted(out)
    return np.array(ks), np.array([out[k] for k in ks]), d
a0 = 2.87
dw, Jw, dfe = shells(R + "fe/weighted/results.json")
dwb, Jwb, _ = shells(R + "fe/weighted/results.json", "bruno")
do, Jo, _ = shells(R + "fe/onsite/results.json")
lmto_d = a0 * np.array([np.sqrt(3) / 2, 1, np.sqrt(2), np.sqrt(11) / 2, np.sqrt(3)])
lmto_J = 13.6057 * np.array([1.432, 0.815, -0.016, -0.126, -0.146])
fig, ax = plt.subplots(figsize=(6.6, 3.6))
m = dw < 8
ax.axhline(0, color="k", lw=0.6)
ax.plot(dw[m] / a0, Jw[m], "o-", color=GREEN, label="magex, weighted (MFT)", ms=6)
ax.plot(dwb[m] / a0, Jwb[m], "s--", color=PURPLE, label="magex, Bruno-renormalised", ms=5, mfc="none")
ax.plot(do[m] / a0, Jo[m], "^-", color=GREY, label="on-site rotation (TB2J-like)", ms=6)
ax.plot(lmto_d / a0, lmto_J, "D", color=RED, ms=7, mfc="none", mew=1.6, label="TB-LMTO (Pajda et al. 2001)")
ax.set_xlabel("$r_{ij}/a$"); ax.set_ylabel("$J_{ij}$ (meV)")
ax.legend(fontsize=8.5)
save(fig, "fe_JR")

# ---------------------------------------------------------------- 4 Fe magnons (MFT vs Bruno vs onsite)
from magex.magnon import LSWT, kpath_from_labels
from magex.heisenberg import all_images
def lswt_from(path, key="mft"):
    d = json.load(open(path))
    mp = d["mag_pos"]; M = np.array(d["moments"]); ax_ = np.array(d["refs"][-1], float)
    s = np.sign(M @ ax_); s[s == 0] = 1
    S = [np.linalg.norm(M[i]) / 2 for i in mp]
    cell = np.array(d["cell"]); frac = np.array(d["positions"]) @ np.linalg.inv(cell)
    corr = None if key == "mft" else np.array(d["corrections"][key])
    cpl = []
    for p in d["pairs"]:
        J = np.array(p["J"])
        if corr is not None:
            J = J - np.trace(J) / 3 * np.eye(3) + corr[mp.index(p["i"]), mp.index(p["j"])][tuple(p["nu"])] * np.eye(3)
        imgs = all_images(p["nu"], d["mesh"], frac[p["i"]], frac[p["j"]], cell)
        for Rv in imgs:
            cpl.append((mp.index(p["i"]), mp.index(p["j"]), Rv, J / len(imgs)))
    return LSWT(S, [s[i] * ax_ for i in mp], cpl, np.array(d["A"])[mp])
sp = {"G": [0, 0, 0], "H": [-0.5, 0.5, 0.5], "N": [0, 0, 0.5], "P": [0.25, 0.25, 0.25]}
lab = list("GHNGP")
path, ticks = kpath_from_labels(lab, sp, 50)
fig, ax = plt.subplots(figsize=(6.4, 3.6))
for f, key, c, l, ls in [("fe/weighted", "mft", GREEN, "weighted MFT", "-"), ("fe/weighted", "bruno", PURPLE, "weighted + Bruno", "--"),
                         ("fe/onsite", "mft", GREY, "on-site (TB2J-like)", "-")]:
    W = lswt_from(R + f + "/results.json", key).band(path)
    ax.plot(W, color=c, ls=ls, lw=1.6, label=l)
for t in ticks: ax.axvline(t, color="0.85", lw=0.7)
ax.set_xticks(ticks); ax.set_xticklabels([r"$\Gamma$" if l == "G" else l for l in lab])
ax.set_xlim(0, len(path) - 1); ax.set_ylim(0, None); ax.set_ylabel("$\\hbar\\omega$ (meV)")
ax.legend(fontsize=8.5, loc="upper right")
save(fig, "fe_magnon")

# ---------------------------------------------------------------- 5 Lloyd Fe & SrMnO3
fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.4))
for ax, f, title in [(axs[0], "fe/weighted", "bcc Fe, nearest neighbours"), (axs[1], "srmno3/weighted", "SrMnO$_3$, Mn–Mn nearest neighbours")]:
    d = json.load(open(R + f + "/results.json"))["lloyd"][0]
    th = np.degrees(np.array(d["angles"]))
    ax.plot(th, d["E"], "o-", color=RED, label="Lloyd (exact, frozen potential)")
    ax.plot(th, d["E_bilinear"], "--", color=BLUE, lw=1.8, label="bilinear, $J$ from small angles")
    c = np.cos(np.radians(th) / 2); tt = np.radians(th)
    ss = 1 if "fe" in f else -1
    fit = 4 * ss * d["J_bilinear"] * c * (1 - c) - 2 * d["B"] * np.cos(tt) * (np.cos(tt) - 1)
    ax.plot(th, fit, ":", color=GREEN, lw=2, label=f"fit: $J'$={d['J_bilinear']:.1f}, $B$={d['B']:.1f} meV")
    ax.axhline(0, color="k", lw=0.6); ax.set_xlabel("relative angle $\\theta$ (deg)")
    ax.set_ylabel("$E_{\\rm int}(\\theta)$ (meV)"); ax.set_title(title, fontsize=11); ax.legend(fontsize=8)
    ax.set_xticks([0, 45, 90, 135, 180])
save(fig, "lloyd")

# ---------------------------------------------------------------- 6 SrMnO3 k-convergence vs TB2J
fig, ax = plt.subplots(figsize=(5.0, 3.3))
ax.plot([7, 11], [-8.5569, -8.5363], "o-", color=GREEN, label="magex (T = 0 contour)")
ax.plot([7, 11, 17], [-8.1112, -8.3040, -8.3295], "s-", color=GREY, label="TB2J 0.9.22")
ax.axhline(-8.5396, color=GREEN, ls=":", lw=1, label="magex, Ozaki 300 K, $k=7^3$")
ax.set_xlabel("$k$-mesh $N$ ($N^3$)"); ax.set_ylabel("$J_1$ Mn–Mn (meV)"); ax.legend(fontsize=8.5)
ax.set_xticks([7, 11, 17])
save(fig, "srmno3_kconv")

# ---------------------------------------------------------------- 7 SrMnO3 orbital decomposition heatmap
M = np.array([[0.092, 0, 0, 0, 0], [0, -4.291, 0, 0, 0], [0, 0, -4.291, 0, 0], [0, 0, 0, -0.031, 0], [0, 0, 0, 0, -0.036]])
fig, ax = plt.subplots(figsize=(4.0, 3.4))
im = ax.imshow(M, cmap="RdBu", vmin=-4.5, vmax=4.5)
names = ["$d_{z^2}$", "$d_{xz}$", "$d_{yz}$", "$d_{x^2-y^2}$", "$d_{xy}$"]
ax.set_xticks(range(5)); ax.set_xticklabels(names); ax.set_yticks(range(5)); ax.set_yticklabels(names)
for i in range(5):
    for j in range(5):
        if abs(M[i, j]) > 1e-3: ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=9, color="w" if abs(M[i,j]) > 2 else "k")
ax.set_title("$J_{m m'}$ (meV), Mn–Mn bond along $z$", fontsize=10)
plt.colorbar(im, fraction=0.046)
save(fig, "srmno3_orb")

# ---------------------------------------------------------------- 8 CrI3 consistency (three runs vs one run)
fig, ax = plt.subplots(figsize=(6.4, 3.4))
comp = ["$J_{xx}$", "$J_{yy}$", "$J_{zz}$"]
three = [(0.236, 0.360), (0.450, 0.4265), (0.468, 0.462)]
one = [(0.350, 0.360), (0.435, 0.426), (0.593, 0.587)]
for k in range(3):
    ax.plot([k - 0.15] * 2, three[k], "o", color=GREY, ms=8)
    ax.plot([k - 0.15] * 2, three[k], "-", color=GREY, lw=3, alpha=0.5)
    ax.plot([k + 0.15] * 2, one[k], "o", color=GREEN, ms=8)
    ax.plot([k + 0.15] * 2, one[k], "-", color=GREEN, lw=3, alpha=0.5)
ax.plot([], [], "o-", color=GREY, label="three self-consistent runs (TB2J-style merge)")
ax.plot([], [], "o-", color=GREEN, label="one run, exchange field rotated (magex)")
ax.set_xticks(range(3)); ax.set_xticklabels(comp); ax.set_ylabel("value seen by two references (meV)")
ax.set_ylim(0.15, 0.7); ax.legend(fontsize=8.5, loc="upper left")
ax.set_title("CrI$_3$ NN Cr–Cr: each diagonal component is measured twice", fontsize=10)
save(fig, "cri3_consistency")

# ---------------------------------------------------------------- 9 CrI3 MAE
def mae(path):
    d = json.load(open(path))["mae"]
    return np.array(d["directions"]), np.array(d["E"]), d
fig, ax = plt.subplots(figsize=(6.4, 3.4))
lbl = ["[100]", "[010]", "[001]", "[110]", "[101]", "[011]", "[1$\\bar1$0]", "[111]"]
for path, c, l, off in [(R + "cri3/single_run/results.json", GREY, "spinor SOC (gauge-split)", -0.2),
                        (R + "cri3/replace_soc/results.json", GREEN, "atomic SOC on the spinor exchange", 0.2)]:
    dirs, E, d = mae(path)
    ax.bar(np.arange(8) + off, E, 0.38, color=c, label=l + f"  (TR-odd {d['tr_odd_max']:.2f} meV)")
ax.set_xticks(range(8)); ax.set_xticklabels(lbl, fontsize=9)
ax.set_ylabel("$\\Omega(\\hat n)-\\Omega_{\\min}$ (meV/cell)"); ax.legend(fontsize=8.5)
save(fig, "cri3_mae")

# ---------------------------------------------------------------- 10 GF vs FD parity (cluster)
from toymodels import random_cluster, grand_potential
from magex.contour import Contour
from magex.engine_partition import compute_reference_partition, partition_weights
from magex.engine import rodrigues
from magex.model import TBModel
m = random_cluster(nat=3, no=2, seed=8, soc=0.1)
E = np.linalg.eigvalsh(m.Hk(np.zeros((1, 3))))[0]; mu = 0.5 * (E[5] + E[6])
c = Contour("ozaki", mu, 2000, npole=400)
r = compute_reference_partition(m, [0, 1, 2], (1, 1, 1), c, mode="weighted")
f, _ = partition_weights(m, "weighted")
def rot(mm, rots):
    X = mm.X.copy()
    for a, Rm in rots.items():
        P = np.zeros(mm.norb); P[mm.atom_orbitals(a)] = 1
        XL = mm.X[0] * f[None]
        Xi = P[None, :, None] * XL + np.conj(np.swapaxes(XL, -1, -2)) * P[None, None, :]
        X[0] += np.einsum("ab,bmn->amn", Rm, Xi) - Xi
    return TBModel(mm.cell, mm.positions, mm.symbols, mm.R, mm.H0, X, mm.S, mm.orb_atom, mm.orb_l, mm.orb_m)
gfv, fdv = [], []; h = 1e-3; k0 = np.zeros((1, 3))
for i, j, a, b in itertools.product(range(3), range(3), range(3), range(3)):
    if i == j: continue
    O = lambda si, sj: grand_potential(rot(m, {i: rodrigues(np.eye(3)[a], si*h), j: rodrigues(np.eye(3)[b], sj*h)}), k0, mu, c.kT)
    fdv.append((O(1, 1) - O(1, -1) - O(-1, 1) + O(-1, -1)) / (4 * h * h)); gfv.append(r["K"][i, j, 0, 0, 0, a, b])
gfv, fdv = np.array(gfv) * 1e3, np.array(fdv) * 1e3
fig, ax = plt.subplots(figsize=(4.0, 3.5))
lim = 1.1 * np.max(np.abs(fdv))
ax.plot([-lim, lim], [-lim, lim], color=GREY, lw=1)
ax.scatter(fdv, gfv, s=22, color=RED, zorder=5)
ax.set_xlabel("exact finite difference (meV)"); ax.set_ylabel("Green's function (meV)")
ax.text(-lim * 0.95, lim * 0.75, f"max error\n{np.max(np.abs(gfv - fdv)):.1e} meV", fontsize=9)
ax.set_title("$K_{ab}(i,j)$, cluster with SOC", fontsize=10)
save(fig, "parity")

# ---------------------------------------------------------------- 11 |X(R)| for Fe vs distance
from magex.cli import load_cfg
from magex.model import load_wannier90
mfe = load_wannier90(load_cfg(R + "fe/in.toml")["input"])
dist, xn = [], []
for ir, Rv in enumerate(mfe.R):
    d = np.linalg.norm(Rv @ mfe.cell)
    dist.append(d); xn.append(np.linalg.norm(mfe.X[ir, 2]))
dist, xn = np.array(dist), np.array(xn)
fig, ax = plt.subplots(figsize=(5.2, 3.3))
ax.semilogy(dist, xn * 1e3, "o", color=ORANGE, ms=4, alpha=0.8)
ax.semilogy([0], [xn[mfe.r0_index()] * 1e3], "o", color=RED, ms=8, label="on-site block (rotated by LKAG)")
ax.set_xlabel("$|R|$ (Å)"); ax.set_ylabel("$||X(R)||_F$ (meV)")
ax.set_xlim(-0.3, 9); ax.legend(fontsize=8.5)
ax.set_title("bcc Fe: exchange field in the Wannier basis", fontsize=10)
save(fig, "fe_X_decay")

# ---------------------------------------------------------------- 12 SrMnO3 magnons
d = json.load(open(R + "srmno3/weighted/results.json"))
lsw = lswt_from(R + "srmno3/weighted/results.json")
spS = {"G": [0, 0, 0], "X": [0.5, 0, 0], "M": [0.5, 0.5, 0], "R": [0.5, 0.5, 0.5]}
labS = list("GXMGR"); pathS, tS = kpath_from_labels(labS, spS, 50)
W = lsw.band(pathS)
lswb = lswt_from(R + "srmno3/weighted/results.json", "bruno"); Wb = lswb.band(pathS)
fig, ax = plt.subplots(figsize=(5.6, 3.4))
ax.plot(W, color=GREEN, lw=1.6); ax.plot(Wb, color=PURPLE, lw=1.4, ls="--")
ax.plot([], [], color=GREEN, label="MFT"); ax.plot([], [], color=PURPLE, ls="--", label="Bruno")
for t in tS: ax.axvline(t, color="0.85", lw=0.7)
ax.set_xticks(tS); ax.set_xticklabels([r"$\Gamma$" if l == "G" else l for l in labS])
ax.set_xlim(0, len(pathS) - 1); ax.set_ylim(0, None); ax.set_ylabel("$\\hbar\\omega$ (meV)"); ax.legend(fontsize=8.5)
ax.set_title("G-AFM SrMnO$_3$ (rhombohedral cell)", fontsize=10)
save(fig, "srmno3_magnon")
print("figures done")
