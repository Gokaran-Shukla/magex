"""Figures for Parts II-V of the complete magex deck (spin dynamics, Monte Carlo, landscapes,
micromagnetics, performance).  Every curve is computed here with magex (short runs), except the
VAMPIRE/UppASD/Spirit numbers, which are the measured values of docs/benchmarks.md.

    python make_figs_engine.py          (from docs/slides_src; about 5 minutes on 2 cores)
"""
import os
import sys
import tempfile

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "figs") + os.sep
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.9, "font.family": "DejaVu Sans", "savefig.dpi": 220,
                     "legend.frameon": False})
NAVY, RED, BLUE, GREEN, ORANGE, PURPLE, TEAL, GOLD, GREY = (
    "#14203A", "#C0392B", "#2E6DB4", "#2E8B57", "#E67E22", "#7D3C98", "#138D90", "#B7950B", "#7F8C8D")

from magex.models import write_model
from magex.spinmodel import GAMMA, K_B, MU_B, SpinSystem, load_results, normalize

TMP = tempfile.mkdtemp()


def model(name, **kw):
    p = os.path.join(TMP, name)
    write_model(p, **kw)
    return load_results(p)


def save(fig, name):
    from matplotlib.ticker import NullFormatter
    for a in fig.axes:                       # no crowded minor labels on log axes
        a.xaxis.set_minor_formatter(NullFormatter())
    fig.tight_layout()
    fig.savefig(OUT + name + ".pdf")
    fig.savefig(OUT + name + ".png")
    plt.close(fig)
    print("  wrote", name)


only = sys.argv[1:]


def want(n):
    return not only or n in only


# ------------------------------------------------------------------ 1 performance and cross-validation
if want("bench"):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    codes = ["magex\n2 thr.", "magex\n1 thr.", "VAMPIRE\nserial", "UppASD\n2 thr.", "Spirit\nCPU"]
    llg = [29, 60, 74, 99, 204]
    cols = [GREEN, GREEN, BLUE, ORANGE, PURPLE]
    b = ax[0].bar(codes, llg, color=cols, alpha=0.85)
    ax[0].bar_label(b, fmt="%d", fontsize=9)
    ax[0].set_ylabel("ns per spin per LLG step")
    ax[0].set_title("stochastic LLG, 32$^3$ spins (same CPU)")
    T = [20, 28, 32, 36, 40]
    v = [0.751, 0.578, 0.419, 0.134, 0.048]
    u = [0.751, 0.569, 0.399, 0.125, 0.069]
    m = [0.750, 0.578, 0.411, 0.124, 0.068]
    ax[1].plot(T, m, "o-", color=GREEN, label="magex")
    ax[1].plot(T, v, "s--", color=BLUE, label="VAMPIRE")
    ax[1].plot(T, u, "^:", color=ORANGE, label="UppASD")
    ax[1].axvline(33.49, color=GREY, lw=0.8, ls="--")
    ax[1].text(33.8, 0.6, "$T_C^{\\infty}$ = 33.5 K", color=GREY, fontsize=9)
    ax[1].set_xlabel("T (K)")
    ax[1].set_ylabel("$\\langle|m|\\rangle$")
    ax[1].set_title("same exported model, three codes (13$^3$)")
    ax[1].legend()
    save(fig, "eng_bench")

# ------------------------------------------------------------------ 2 integrators: Larmor error vs dt
if want("integ"):
    from magex.llg import LLGSolver
    s = SpinSystem(model("one.json", lattice="sc", J=[0.0]), (1, 1, 1), (False, False, False), field=(0, 0, 5.0))
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    dts = np.array([0.02, 0.01, 0.005, 0.0025, 0.00125])
    for it, c, mk in [("euler", GREY, "x"), ("heun", GREEN, "o"), ("depondt", BLUE, "s"), ("sib", ORANGE, "^"),
                      ("rk4", PURPLE, "d")]:
        err = []
        for dt in dts:
            e = normalize(np.array([[1.0, 0, 0.3]]))
            sv = LLGSolver(s, 0.0, dt, integrator=it, backend="numpy")
            n = int(round(2.0 / dt))
            for _ in range(n):
                e = sv.step(e)
            ex = np.array([np.cos(GAMMA * 5 * 2.0), np.sin(GAMMA * 5 * 2.0), 0.3]) / np.sqrt(1.09) * np.array([1, 1, 1])
            ex[:2] *= 1.0
            ref = normalize(np.array([[np.cos(GAMMA * 5 * 2.0), np.sin(GAMMA * 5 * 2.0), 0.3]]))[0]
            err.append(max(np.linalg.norm(e[0] - ref), 1e-16))
        ax[0].loglog(dts, err, mk + "-", color=c, label=it)
    ax[0].loglog(dts, 2 * dts**2, ":", color="k", lw=0.8)
    ax[0].text(0.004, 2e-5, "$\\propto\\Delta t^2$", fontsize=9)
    ax[0].set_xlabel("$\\Delta t$ (ps)")
    ax[0].set_ylabel("$|\\mathbf{e}(t) - \\mathbf{e}_{\\rm exact}(t)|$")
    ax[0].set_title("Larmor precession, B = 5 T, t = 2 ps")
    ax[0].legend(fontsize=8, ncol=2)
    from magex import fast
    g = np.empty((200000, 3))
    fast._gauss3(g, fast.seed_streams(3), np.ones(len(g)), np.zeros(len(g), bool))
    x = np.linspace(-5, 5, 400)
    ax[1].hist(g.ravel(), bins=120, density=True, color=TEAL, alpha=0.6, label="ziggurat (magex)")
    ax[1].semilogy(x, np.exp(-x**2 / 2) / np.sqrt(2 * np.pi), color=RED, lw=1.2, label="N(0, 1)")
    ax[1].set_ylim(1e-6, 1)
    ax[1].set_xlabel("x")
    ax[1].set_title("thermal-field generator, 6$\\times$10$^5$ samples")
    ax[1].legend(fontsize=8)
    save(fig, "eng_integrators")

# ------------------------------------------------------------------ 3 Monte Carlo: moves, CMC, LSF
if want("mc"):
    from magex.cmc import anisotropy_vs_T
    from magex.montecarlo import LSFMonteCarlo, MetropolisMC
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    uc = model("free.json", lattice="sc", J=[0.0], K=1.0)
    s = SpinSystem(uc, (12, 12, 12), (False, False, False), field=(0, 0, 2.0))
    Ts = [2, 5, 10, 20, 40]
    th = np.linspace(0, np.pi, 801)
    exact = []
    for T in Ts:
        En = s.A[0][2, 2] * np.cos(th) ** 2 + s.A[0][0, 0] * np.sin(th) ** 2 - 2 * MU_B * 2.0 * np.cos(th)
        w = np.exp(-(En - En.min()) / (K_B * T)) * np.sin(th)
        exact.append(np.trapezoid(w * np.cos(th), th) / np.trapezoid(w, th))
    ax[0].plot(Ts, exact, "-", color="k", lw=1, label="exact integral")
    for mv, c, mk in [("gaussian", GREEN, "o"), ("heatbath", BLUE, "s"), ("uniform", ORANGE, "^")]:
        r = []
        for T in Ts:
            mc = MetropolisMC(s, T, 1, 0.6, 0, mv)
            e = s.init_spins("ferro")
            ms = []
            for n in range(300):
                e = mc.sweep(e)
                if n > 50:
                    ms.append(e[:, 2].mean())
            r.append(np.mean(ms))
        ax[0].plot(Ts, r, mk, color=c, label=mv, ms=6, mfc="none")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("T (K)")
    ax[0].set_ylabel("$\\langle e_z\\rangle$")
    ax[0].set_title("independent spins: moves vs exact")
    ax[0].legend(fontsize=8)
    s2 = SpinSystem(model("k.json", lattice="sc", J=[1.0], K=0.1), (6, 6, 6))
    R = anisotropy_vs_T(s2, [1, 4, 8, 12, 16, 20, 24], therm=300, meas=800)
    mm = np.linspace(0.6, 1, 50)
    ax[1].plot(mm, mm**3, "-", color=GREY, label="$m^3$ (Callen-Callen)")
    ax[1].errorbar(R[:, 1], R[:, 2] / 0.1, R[:, 3] / 0.1, fmt="o", color=RED, label="constrained MC")
    ax[1].set_xlabel("m(T)")
    ax[1].set_ylabel("K(T) / K(0)")
    ax[1].set_title("anisotropy from the torque")
    ax[1].legend(fontsize=8)
    s3 = SpinSystem(model("sc.json", lattice="sc", J=[1.0]), (6, 6, 6))
    Ts3 = [2, 10, 20, 30, 40, 55, 70]
    sm, mo = [], []
    for T in Ts3:
        mc = LSFMonteCarlo(s3, T, 1, 0.15, b=2.0)
        M = s3.ref.copy()
        a1, a2 = [], []
        for n in range(1500):
            M = mc.sweep(M)
            if n > 400:
                a1.append(np.linalg.norm(M, axis=1).mean())
                a2.append(np.linalg.norm(M.mean(0)))
        sm.append(np.mean(a1))
        mo.append(np.mean(a2))
    ax[2].plot(Ts3, sm, "o-", color=PURPLE, label="$\\langle s\\rangle$ (moment length)")
    ax[2].plot(Ts3, mo, "s-", color=GREEN, label="$|\\langle \\mathbf{M}\\rangle|$ (order)")
    ax[2].set_xlabel("T (K)")
    ax[2].set_title("longitudinal fluctuations")
    ax[2].legend(fontsize=8)
    save(fig, "eng_mc")

# ------------------------------------------------------------------ 4 dipolar
if want("dip"):
    from magex.terms import DIP, Dipolar
    uc = model("sc.json", lattice="sc", J=[1.0])
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    ims = [1, 2, 3, 4, 6, 8]
    s = SpinSystem(uc, (12, 12, 1), (True, True, False))
    wt, nt = [], []
    for im in ims:
        for tail, lst in ((True, wt), (False, nt)):
            d = Dipolar(s, images=im, tail=tail)
            lst.append((d.energy(np.tile([0, 0, 1.0], (s.N, 1))) - d.energy(np.tile([1.0, 0, 0], (s.N, 1)))) / s.N)
    dref = Dipolar(s, images=16)
    ref = (dref.energy(np.tile([0, 0, 1.0], (s.N, 1))) - dref.energy(np.tile([1.0, 0, 0], (s.N, 1)))) / s.N
    ax[0].semilogy(ims, np.abs(np.array(nt) - ref) / ref, "o-", color=GREY, label="spherical cut-off only")
    ax[0].semilogy(ims, np.abs(np.array(wt) - ref) / ref, "s-", color=RED, label="+ analytic sheet tail")
    print("  tail errors", np.abs(np.array(wt) - ref) / ref)
    ax[0].set_xlabel("image shells")
    ax[0].set_ylabel("rel. error of $E_\\perp - E_\\parallel$ (ref.: 16 shells)")
    ax[0].set_title("monolayer dipolar sum")
    ax[0].legend(fontsize=8)
    ts, Es = [], []
    for nz in (1, 2, 4, 6, 8):
        f = SpinSystem(uc, (16, 16, nz), (True, True, False))
        d = Dipolar(f, images=6)
        Es.append((d.energy(np.tile([0, 0, 1.0], (f.N, 1))) - d.energy(np.tile([1.0, 0, 0], (f.N, 1)))) / f.N)
        ts.append(nz)
    cont = 2 * np.pi * DIP * 4 * MU_B / 27.0
    ax[1].plot(ts, np.array(Es) / cont, "o-", color=BLUE)
    ax[1].axhline(1, color=GREY, ls="--", lw=0.8)
    ax[1].set_xlabel("film thickness (layers)")
    ax[1].set_ylabel("$(E_\\perp - E_\\parallel)\\,/\\,\\frac{\\mu_0}{2}M_s^2V$")
    ax[1].set_title("shape anisotropy: lattice sum vs continuum")
    save(fig, "eng_dipolar")

# ------------------------------------------------------------------ 5 torques: Zhang-Li wall, FMR
if want("torque"):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    ba = np.array([0, 0.5, 1.0, 2.0])
    v = np.array([0.10, 19.70, 40.2, 78.53])        # magex chain runs (u = 40 m/s, alpha = 0.1)
    ax[0].plot(ba, ba * 40, "-", color=GREY, label="$v = (\\beta/\\alpha)\\,u$")
    ax[0].plot(ba, v, "o", color=RED, ms=7, label="atomistic LLG + STT")
    ax[0].set_xlabel("$\\beta/\\alpha$")
    ax[0].set_ylabel("wall velocity (m/s)")
    ax[0].set_title("current-driven wall, u = 40 m/s")
    ax[0].legend(fontsize=8)
    from magex.programs import fmr
    s = SpinSystem(model("k.json", lattice="sc", J=[1.0], K=0.1), (4, 4, 4))
    fs, fk = [], []
    for B in (0.0, 0.5, 1.0, 2.0, 3.0):
        s.set_field([0, 0, B])
        (f, F), fp, _ = fmr(s, s.init_spins("ferro"), steps=20000)
        fs.append(fp)
        fk.append(GAMMA * (B + 2 * 0.1 / (2 * MU_B)) / (2 * np.pi) * 1e3)
    Bs = [0, 0.5, 1, 2, 3]
    ax[1].plot(Bs, fk, "-", color=GREY, label="Kittel $\\gamma(B + B_K)/2\\pi$")
    ax[1].plot(Bs, fs, "o", color=BLUE, ms=7, label="FMR program")
    ax[1].set_xlabel("B (T)")
    ax[1].set_ylabel("f (GHz)")
    ax[1].set_title("ferromagnetic resonance")
    ax[1].legend(fontsize=8)
    save(fig, "eng_torques")

# ------------------------------------------------------------------ 6 SLD and LSD
if want("sld"):
    from magex.sld import SpinLattice
    s = SpinSystem(model("k.json", lattice="sc", J=[1.0], K=0.1), (4, 4, 4))
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    dts = [4e-3, 2e-3, 1e-3, 5e-4]
    for integ, c, mk in [("heun", ORANGE, "o"), ("symplectic", GREEN, "s")]:
        err = []
        for dt in dts:
            rng = np.random.default_rng(0)
            sl = SpinLattice(s, lam=0.5, k_spring=500.0, dt=dt, integrator=integ, shells="all")
            sl.v = rng.normal(size=sl.v.shape) * 2.0
            e = normalize(s.ref + 0.4 * rng.normal(size=s.ref.shape))
            e, tab = sl.run(e, int(1.0 / dt), every=max(1, int(0.02 / dt)))
            err.append(np.ptp(tab[:, 1]))
        ax[0].loglog(dts, err, mk + "-", color=c, label=f"spin sub-step: {integ}")
    ax[0].loglog(dts, 0.3 * (np.array(dts) / 2e-3) ** 2, ":", color="k", lw=0.8)
    ax[0].set_xlabel("$\\Delta t$ (ps)")
    ax[0].set_ylabel("max |$\\Delta E_{\\rm tot}$| (meV)")
    ax[0].set_title("spin-lattice dynamics, 1 ps")
    ax[0].legend(fontsize=8)
    rng = np.random.default_rng(0)
    for integ, c in [("heun", ORANGE), ("symplectic", GREEN)]:
        sl = SpinLattice(s, lam=1e9, k_spring=500.0, dt=2e-3, integrator=integ)
        e = normalize(s.ref + 0.4 * np.random.default_rng(1).normal(size=s.ref.shape))
        e, tab = sl.run(e, 2000, every=20)
        ax[1].plot(tab[:, 0], tab[:, 1] - tab[0, 1], color=c, label=integ)
    ax[1].set_xlabel("t (ps)")
    ax[1].set_ylabel("$E(t) - E(0)$ (meV)")
    ax[1].set_title("frozen lattice: spin energy conservation")
    ax[1].legend(fontsize=8)
    save(fig, "eng_sld")

# ------------------------------------------------------------------ 7 landscapes: GNEB + spectrum
if want("land"):
    from magex.gneb import AtomisticAdapter, GNEB, minimize as bbmin, slerp_path
    from magex.htst import htst_refined
    uc = model("sk.json", lattice="triangular", a=3.0, J=[2.0], D=1.0, K=0.2, moment=3.0)
    s = SpinSystem(uc, (20, 20, 1), (True, True, False), field=(0, 0, 6.0))
    ad = AtomisticAdapter(s)
    a, _, _ = bbmin(ad, s.init_spins("skyrmion", radius=7.0), 1e-6)
    b, _, _ = bbmin(ad, s.init_spins("ferro"), 1e-6)
    neb = GNEB(ad, slerp_path(a, b, 14), 1.0, True)
    rng = np.random.default_rng(1)
    for k in range(1, len(neb.x) - 1):
        neb.x[k] = normalize(neb.x[k] + 0.02 * rng.normal(size=neb.x[k].shape))
    neb.run(0.0, 4000, 1e-4)
    res = neb.result()
    E = np.array(res["E_rel"])
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    ax[0].plot(res["s"], E, "o-", color=RED)
    ax[0].set_xlabel("reaction coordinate (geodesic, rad)")
    ax[0].set_ylabel("$E - E_{\\rm Sk}$ (meV)")
    ax[0].set_title(f"skyrmion collapse: barrier {res['barrier_forward']:.1f} meV")
    hr = htst_refined(s, neb.x[0], neb.x[res["saddle_image"]], 20.0)
    ax[1].plot(np.arange(12), hr["eps_min"][:12], "o", color=GREEN, label="skyrmion (minimum)")
    ax[1].plot(np.arange(12), hr["eps_sp"][:12], "s", color=RED, label="saddle point")
    ax[1].axhline(0, color=GREY, lw=0.7)
    ax[1].set_xlabel("mode index")
    ax[1].set_ylabel("Hessian eigenvalue (meV)")
    ax[1].set_title(f"HTST: zero modes {hr['n_zero']}, prefactor {hr['prefactor']:.2g} Hz")
    ax[1].legend(fontsize=8)
    save(fig, "eng_landscape")

# ------------------------------------------------------------------ 8 micromagnetics: moving frame + kernels
if want("mm"):
    from magex.micromag.constants import GAMMA_LL
    from magex.micromag.custom import MovingFrame, custom_energy
    from magex.micromag.mesh import Mesh
    from magex.micromag.solver import RK45, relax
    from magex.micromag.system import Micromagnet
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    for Bz, c in [(0.01, BLUE), (0.02, GREEN), (0.04, RED)]:
        mesh = Mesh((80, 1, 1), (2e-9,) * 3)
        s = Micromagnet(mesh, demag=False)
        s.add_region(np.ones(mesh.n, bool), "w", Ms=8e5, A=1.3e-11, Ku1=5e5, alpha=0.5)
        s.custom_energies.append(custom_energy(dict(density="Kh*my**2", constants=dict(Kh=1e5)), mesh))
        X, _, _ = mesh.coords()
        th = 2 * np.arctan(np.exp(-X / 5e-9))
        m = relax(s, np.stack([np.sin(th), 0 * th, np.cos(th)], -1), 1e-4)
        s.set_field([0, 0, Bz])
        fr = MovingFrame(s, "z", "x")
        m, _ = RK45(s, tol=1e-6).run(m, 1.5e-9, 10e-12, fr)
        P = np.array(fr.positions)
        ax[0].plot(P[:, 0] * 1e9, -(P[:, 1] - P[0, 1]) * 1e9, color=c, label=f"B = {Bz * 1e3:.0f} mT")
        ax[0].plot(P[:, 0] * 1e9, GAMMA_LL * np.sqrt(1.3e-11 / 5e5) * Bz / 0.5 * P[:, 0] * 1e9, ":", color=c, lw=0.8)
    ax[0].set_xlabel("t (ns)")
    ax[0].set_ylabel("wall displacement (nm)")
    ax[0].set_title("moving frame vs 1D model $\\gamma\\Delta B/\\alpha$")
    ax[0].legend(fontsize=8)
    lbl = ["local fields\nNumPy", "local fields\nfused kernel", "demag FFT\ndouble", "demag FFT\nsingle"]
    val = [31.0, 2.1, 33.7, 18.1]
    bb = ax[1].bar(lbl, val, color=[GREY, GREEN, BLUE, TEAL], alpha=0.85)
    ax[1].bar_label(bb, fmt="%.1f", fontsize=9)
    ax[1].set_ylabel("ms per evaluation")
    ax[1].set_title("256 x 256 x 1 cells, 2 cores")
    save(fig, "eng_micromag")

print("done")
