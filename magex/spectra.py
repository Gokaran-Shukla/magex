"""Magnon spectra from spin dynamics.

Atomistic:  dynamical structure factor from stochastic LLG trajectories,

    S^{ab}(q, w) = < m^a_q(w) m^b_q(w)* >,   m_q(t) = (1/sqrt N) sum_i e_i(t) exp(-i q.r_i)

(Hann-windowed Welch average over time segments).  At low temperature its maxima follow the
linear spin-wave (LSWT) bands; at higher temperature it contains the magnon softening and
lifetimes that LSWT cannot give.  Transverse (S_perp, relative to the mean magnetisation or Neel
vector) and longitudinal parts are returned.

Micromagnetic:  spin-wave dispersion from a broadband excitation (sinc pulse in time, localised
in space) and a 2D FFT of m(x, t) along a line (stage type "dispersion" in magex micromag).
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

from .quantum import HBAR_MEV_PS


def commensurate_path(labels, special, L, npts=40):
    """q-path (fractional) restricted to points commensurate with an L supercell; returns q, ticks."""
    from .magnon import kpath_from_labels
    path, ticks = kpath_from_labels(labels, special, npts * 4)
    Lv = np.array(L, float)
    q = np.round(path * Lv) / Lv
    keep = [0]
    for k in range(1, len(q)):
        if np.linalg.norm(q[k] - q[keep[-1]]) > 1e-9:
            keep.append(k)
    keep = np.array(keep)
    newticks = [int(np.argmin(np.abs(keep - t))) for t in ticks]
    return q[keep], newticks


def _frame(n):
    n = np.asarray(n, float) / np.linalg.norm(n)
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(n, a)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(n, e1), n


def structure_factor(system, solver, e, qfrac, nsteps, sample_every=10, nseg=4, log=None):
    """Run ``nsteps`` LLG steps sampling every ``sample_every``; returns (E_meV, S_perp, S_par, e).

    qfrac  (nq, 3) fractional coordinates of the magnetic unit cell's reciprocal lattice
    """
    B = 2 * np.pi * np.linalg.inv(system.uc.cell).T            # reciprocal vectors (rows)
    qc = np.asarray(qfrac) @ B
    ph = np.exp(-1j * system.positions @ qc.T) / np.sqrt(system.N)    # (N, nq)
    nsamp = nsteps // sample_every
    mq = np.zeros((nsamp, len(qc), 3), complex)
    ref = np.zeros(3)
    t0 = time.time()
    for s in range(nsamp):
        for _ in range(sample_every):
            e = solver.step(e)
        ref += (system.signs[:, None] * e).sum(0)
        mq[s] = ph.T @ e
    e1, e2, n = _frame(ref)
    dt = solver.dt * sample_every
    seg = nsamp // nseg
    win = np.hanning(seg)
    Sperp = 0.0
    Spar = 0.0
    for k in range(nseg):
        x = mq[k * seg:(k + 1) * seg]
        x = x - x.mean(0, keepdims=True)
        F = np.fft.fft(x * win[:, None, None], axis=0)
        Sperp = Sperp + np.abs(F @ e1) ** 2 + np.abs(F @ e2) ** 2
        Spar = Spar + np.abs(F @ n) ** 2
    f = np.fft.fftfreq(seg, dt)                                 # 1/ps
    E = 2 * np.pi * f * HBAR_MEV_PS
    order = np.argsort(E)
    if log:
        log(f"  S(q,w): {nsamp} samples x {len(qc)} q-points, {nseg} segments, "
            f"dE = {E[1] - E[0]:.3f} meV, E_max = {E.max():.1f} meV, {time.time() - t0:.1f} s")
    return E[order], (Sperp / nseg)[order].T, (Spar / nseg)[order].T, e


def lswt_bands(uc, qfrac):
    """LSWT energies (meV) of a UnitCellModel along qfrac."""
    from .magnon import LSWT
    S = np.abs(np.asarray(uc.moments, float)).reshape(uc.nsites, -1).max(1) / 2
    dirs = np.asarray(uc.ref_dirs, float)
    lsw = LSWT(S, dirs, [(i, j, np.asarray(R), np.asarray(J)) for (i, j, R, J) in uc.couplings], uc.A)
    return lsw.band(qfrac)


def plot_sqw(E, S, qfrac, ticks, labels, path, bands=None, emax=None, title=None):
    from .plotstyle import mpl, savefig
    plt = mpl()
    from matplotlib.colors import LogNorm
    emax = emax or (1.15 * np.nanmax(bands) if bands is not None else E.max())
    sel = (E >= 0) & (E <= emax)
    Z = S[:, sel].T
    Z = Z / Z.max()
    fig, ax = plt.subplots(figsize=(5.2, 3.8), constrained_layout=True)
    x = np.arange(len(qfrac))
    m = ax.pcolormesh(x, E[sel], Z, shading="nearest", cmap="magma", norm=LogNorm(vmin=1e-4, vmax=1), rasterized=True)
    cb = fig.colorbar(m, ax=ax, pad=0.01)
    cb.set_label(r"$S_\perp(\mathbf{q},\omega)$ (norm.)")
    if bands is not None:
        for b in np.atleast_2d(bands.T):
            ax.plot(x, b, color="#56B4E9", lw=1.0, ls="--", label=None)
        ax.plot([], [], color="#56B4E9", ls="--", label="LSWT")
        ax.legend(loc="upper right", labelcolor="white", frameon=False)
    ax.set_xticks([x[t] for t in ticks])
    ax.set_xticklabels([r"$\Gamma$" if l in ("G", "Γ") else l for l in labels])
    for t in ticks:
        ax.axvline(x[t], color="w", lw=0.4, alpha=0.6)
    ax.set_xlim(0, len(qfrac) - 1)
    ax.set_ylim(0, emax)
    ax.set_ylabel(r"$\hbar\omega$ (meV)")
    ax.minorticks_off()
    if title:
        ax.set_title(title)
    savefig(fig, path)
    plt.close(fig)


def run_sqw_config(cfg, base=".", log=print):
    """magex sqw input.toml: [spinmodel] + [sqw]."""
    from .llg import LLGSolver
    from .spinmodel import SpinSystem, build_from_config
    uc, sm = build_from_config(cfg, base)
    c = dict(dict(temperature=10.0, alpha=0.01, dt=1e-3, therm_steps=4000, steps=32768, sample_every="auto",
                  segments=4, field=[0, 0, 0], path="GXMG", special_points=None, npts=40, emax=None,
                  thermostat="classical", seed=1, outdir="sqw_out", lswt=True), **cfg.get("sqw", {}))
    outdir = c["outdir"] if os.path.isabs(c["outdir"]) else os.path.join(base, c["outdir"])
    os.makedirs(outdir, exist_ok=True)
    L = sm.get("supercell", [24, 24, 1])
    system = SpinSystem(uc, L, sm.get("pbc", [True, True, True]), c["field"])
    log(system.summary())
    if cfg.get("structure"):
        from .structure import apply_structure
        apply_structure(system, cfg["structure"], log)
    labels = list(c["path"]) if isinstance(c["path"], str) else list(c["path"])
    special = c["special_points"] or dict(G=[0, 0, 0], X=[0.5, 0, 0], M=[0.5, 0.5, 0], K=[1 / 3, 1 / 3, 0],
                                         R=[0.5, 0.5, 0.5], Y=[0, 0.5, 0], Z=[0, 0, 0.5])
    q, ticks = commensurate_path(labels, special, L, int(c["npts"]))
    bands = lswt_bands(uc, q) if c["lswt"] else None
    if c["sample_every"] == "auto":
        emax_est = 1.6 * (np.nanmax(bands) if bands is not None else 100.0)
        c["sample_every"] = max(1, int(np.pi * HBAR_MEV_PS / (c["dt"] * emax_est)))
        log(f"sample_every = {c['sample_every']} (Nyquist {np.pi * HBAR_MEV_PS / (c['dt'] * c['sample_every']):.1f} meV)")
    from .damping import resolve_alpha
    solver = LLGSolver(system, resolve_alpha(c["alpha"], base), c["dt"], c["temperature"], c["seed"],
                       thermostat=c["thermostat"])
    e = system.init_spins("reference")
    log(f"thermalising {c['therm_steps']} steps at T = {c['temperature']} K (alpha = {solver.alpha})")
    for _ in range(int(c["therm_steps"])):
        e = solver.step(e)
    E, Sp, Sl, e = structure_factor(system, solver, e, q, int(c["steps"]), int(c["sample_every"]),
                                    int(c["segments"]), log)
    np.savez_compressed(os.path.join(outdir, "sqw.npz"), E_meV=E, S_perp=Sp, S_par=Sl, q=q, ticks=ticks,
                        labels=labels, lswt=bands if bands is not None else np.zeros(0))
    plot_sqw(E, Sp, q, ticks, labels, os.path.join(outdir, "sqw.png"), bands, c["emax"],
             title=f"$S_\\perp(\\mathbf{{q}},\\omega)$, T = {c['temperature']} K ({c['thermostat']})")
    # peak positions vs LSWT (lowest branch) for a quick check
    if bands is not None:
        pos = E >= 0
        peak = E[pos][np.argmax(Sp[:, pos], axis=1)]
        dev = np.median(np.abs(peak - bands.min(1)) / np.maximum(bands.max(), 1e-9))
        log(f"  median |peak - LSWT| / bandwidth = {dev:.3f}")
        with open(os.path.join(outdir, "sqw_summary.json"), "w") as f:
            json.dump(dict(peak_meV=peak.tolist(), lswt_meV=bands.tolist(), q=q.tolist(), T=c["temperature"]), f)
    log(f"S(q,w) written to {outdir}/")
    return E, Sp, q, bands


# ----------------------------------------------------------------------------- micromagnetic
def micromag_dispersion(system, m, time_ns=2.0, fmax_GHz=60.0, B_amp=1e-3, axis=0, source_nm=None,
                        width_cells=1, dt_fs=None, layer=0, line=None, log=None):
    """Spin-wave dispersion m(k, f) by a sinc pulse localised on a line of cells and a 2D FFT.

    axis     propagation axis (0 = x, 1 = y);  the source is a strip of ``width_cells`` across it
    returns  k (1/nm), f (GHz), power (nk, nf), component index used
    """
    from .micromag.constants import GAMMA_LL
    from .micromag.solver import Heun
    mesh = system.mesh
    if dt_fs is None:                      # keep gamma B_max dt <= 0.05 rad for the fixed-step integrator
        mag = system.magnetic
        Ms = system.Ms[mag].min()
        Bmax = (8 * system.A[mag].max() / (Ms * min(d for d, nn in zip(mesh.d, mesh.n) if nn > 1) ** 2)
                + 2 * abs(system.Ku1[mag]).max() / Ms + 4e-7 * np.pi * system.Ms[mag].max() + np.linalg.norm(system.B_ext))
        dt_fs = 0.05 / (GAMMA_LL * Bmax) * 1e15
        if log:
            log(f"    dispersion: fixed time step {dt_fs:.2f} fs (B_max ~ {Bmax:.0f} T)")
    n = mesh.n
    X, Y, Z = mesh.coords()
    pos = (X, Y)[axis]
    src = 0.0 if source_nm is None else source_nm * 1e-9
    mask = np.abs(pos - src) < (width_cells * mesh.d[axis] / 2 + 1e-15)
    mavg = system.average(m)
    u = mavg / np.linalg.norm(mavg)
    exc = np.cross(u, [0, 0, 1.0]) if abs(u[2]) < 0.9 else np.array([1.0, 0, 0])
    exc = exc / np.linalg.norm(exc)
    fc = fmax_GHz * 1e9
    t0 = 5.0 / fc
    system.B_local.append((mask, lambda t: B_amp * exc * np.sinc(2 * fc * (t - t0))))
    rec_dt = 1.0 / (2.5 * fc)
    other = 1 - axis
    line = n[other] // 2 if line is None else line
    sl = [slice(None)] * 3
    sl[other] = line
    sl[2] = layer
    frames = []

    def cb(t, mm):
        frames.append(mm[tuple(sl)].copy())
    t_start = system.t
    solver = Heun(system, dt=dt_fs * 1e-15)
    m, _ = solver.run(m, time_ns * 1e-9, rec_dt, cb, log=log)
    system.B_local.pop()
    system.t = t_start
    data = np.array(frames)                                   # (nt, nline, 3)
    data = data - data.mean(0, keepdims=True)
    comp = int(np.argmax([np.var(data[..., c]) for c in range(3)]))
    sig = data[..., comp] * np.hanning(len(data))[:, None]
    F = np.fft.fftshift(np.fft.fft2(sig))
    P = np.abs(F) ** 2
    f = np.fft.fftshift(np.fft.fftfreq(len(data), rec_dt)) / 1e9
    k = np.fft.fftshift(np.fft.fftfreq(n[axis], mesh.d[axis])) * 2 * np.pi * 1e-9      # rad/nm
    keep = f >= 0
    return k, f[keep], P[keep].T, comp


def plot_dispersion(k, f, P, path, analytic=None, title=None, kmax=None, fmax=None):
    from matplotlib.colors import LogNorm
    from .plotstyle import mpl, savefig
    plt = mpl()
    fig, ax = plt.subplots(figsize=(5.0, 3.8), constrained_layout=True)
    Z = P.T / P.max()
    mpc = ax.pcolormesh(k, f, Z, shading="nearest", cmap="magma", norm=LogNorm(vmin=1e-5, vmax=1), rasterized=True)
    cb = fig.colorbar(mpc, ax=ax, pad=0.01)
    cb.set_label("spin-wave power (norm.)")
    if analytic is not None:
        ax.plot(k, analytic, ls="--", color="#56B4E9", lw=1.1, label="analytic (thin film)")
        ax.legend(loc="upper center", labelcolor="white")
    ax.set_xlim(-(kmax or np.abs(k).max()), kmax or np.abs(k).max())
    ax.set_ylim(0, fmax or f.max())
    ax.set_xlabel(r"$k$ (rad nm$^{-1}$)")
    ax.set_ylabel(r"$f$ (GHz)")
    ax.minorticks_off()
    if title:
        ax.set_title(title)
    savefig(fig, path)
    plt.close(fig)


def analytic_film_dispersion(system, k_nm, axis=0):
    """Thin-film spin-wave dispersion (GHz) of a uniformly magnetised film along the propagation axis.

    out-of-plane m:  f = gamma/2pi [B0 + 2K/Ms - mu0 Ms + B_ex(k)]
    in-plane m:      f = gamma/2pi sqrt[(B0 + B_ex)(B0 + B_ex + mu0 Ms - 2K/Ms)]
                         + gamma/2pi (2 D/Ms) sin(k d)/d ((m x z).k_hat)        (interfacial DMI)
    with B_ex = (2A/Ms)(2 - 2 cos k d)/d^2 (discrete exchange), K along z; mu0 Ms only with demag on.
    The dynamic dipolar k-dependence of thin films (kt << 1) is neglected.
    """
    from .micromag.constants import GAMMA_LL, MU0
    mag = system.magnetic
    Ms = system.Ms[mag].mean()
    A = system.A[mag].mean()
    K = system.Ku1[mag].mean() * abs(system.anisU[mag].mean(0)[2])
    D = system.D_int[mag].mean()
    d = system.mesh.d[axis]
    k = np.asarray(k_nm) * 1e9
    Bex = 2 * A / Ms * (2 - 2 * np.cos(k * d)) / d**2
    B0 = float(np.linalg.norm(system.B_ext))
    mu = MU0 * Ms if system.use_demag else 0.0
    m = system.average(system.init("uniform", direction=system.B_ext if B0 > 0 else (0, 0, 1)))
    g = GAMMA_LL / (2 * np.pi)
    if abs(m[2]) > 0.9:
        f = g * (B0 + 2 * K / Ms - mu + Bex)
    else:
        kh = np.zeros(3)
        kh[axis] = 1.0
        sgn = float(np.dot(np.cross(m, [0, 0, 1.0]), kh))
        f = g * np.sqrt(np.clip((B0 + Bex) * (B0 + Bex + mu - 2 * K / Ms), 0, None)) \
            + g * 2 * D / Ms * np.sin(k * d) / d * sgn
    return f / 1e9
