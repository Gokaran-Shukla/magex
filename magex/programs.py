"""Simulation programs (VAMPIRE "sim:program" / UppASD / Spirit method equivalents).

magex program input.toml   with  [spinmodel], optional [structure], and [program]:

  type = "hysteresis"      field loop B(t) along `direction` from +Bmax to -Bmax and back (npts points);
                           each point relaxed by a minimiser (T = 0) or by MC/LLG at temperature T
  type = "field_cooling"   MC cooling from T_start to T_end in a field (m(T), E(T))
  type = "anneal"          simulated annealing (MC, geometric schedule) followed by a minimiser
  type = "fmr"             ferromagnetic resonance: small field pulse, LLG ring-down, FFT of m(t)
  type = "cmc"             constrained Monte Carlo: torque curve and K(T)
  type = "minimize"        lbfgs | ncg | vp | llg relaxation of an initial state (texture)
  type = "eigenmodes"      Hessian / LLG eigenfrequencies of a relaxed state (small systems)
  type = "lsf"             Monte Carlo with longitudinal spin fluctuations, m(T) and <s>(T)
Outputs: CSV tables, PNG figures, final spins as NPZ/CSV, VTK and extended XYZ.
"""
from __future__ import annotations

import os

import numpy as np

from .spinmodel import MU_B, normalize


def _setup(cfg, base, log, field=(0, 0, 0)):
    from .spinmodel import SpinSystem, build_from_config
    from .structure import apply_structure
    uc, sm = build_from_config(cfg, base)
    system = SpinSystem(uc, sm.get("supercell", [16, 16, 1]), sm.get("pbc", [True, True, False]), field)
    log(system.summary())
    apply_structure(system, cfg.get("structure"), log)
    return system


def _out(base, c, default):
    d = c.get("outdir", default)
    d = d if os.path.isabs(d) else os.path.join(base, d)
    os.makedirs(d, exist_ok=True)
    return d


def _init(system, c):
    tex = dict(c.get("texture", {}))
    return system.init_spins(c.get("init", "ferro"), c.get("init_axis", [0, 0, 1]), c.get("seed", 1), **tex)


def _write_spins(outdir, tag, system, e):
    from .llg import save_spins
    from .observables import write_vtk, write_xyz
    save_spins(os.path.join(outdir, f"spins_{tag}.npz"), system, e)
    write_vtk(os.path.join(outdir, f"spins_{tag}.vtk"), system, e)
    write_xyz(os.path.join(outdir, f"spins_{tag}.xyz"), system, e)


def hysteresis(system, e, Bmax, npts=41, direction=(0, 0, 1), T=0.0, method="lbfgs", mc_sweeps=500,
               seed=1, log=None, tilt=1e-3):
    from .minimize import minimize
    u = np.asarray(direction, float)
    u /= np.linalg.norm(u)
    # small tilt breaks the symmetry of a perfectly aligned loop (coherent reversal needs a seed)
    a = np.array([1.0, 0, 0]) if abs(u[0]) < 0.9 else np.array([0, 1.0, 0])
    ut = normalize((u + tilt * np.cross(u, a))[None])[0]
    Bs = np.r_[np.linspace(Bmax, -Bmax, npts), np.linspace(-Bmax, Bmax, npts)[1:]]
    rows = []
    mc = None
    if T > 0:
        from .montecarlo import MetropolisMC
        mc = MetropolisMC(system, T, seed, move="gaussian")
    for k, B in enumerate(Bs):
        system.set_field(B * ut)
        if mc is None:
            e, info = minimize(system, e, method, tol=1e-6)
        else:
            m_acc = []
            for n in range(mc_sweeps):
                e = mc.sweep(e)
                if n >= mc_sweeps // 2:
                    m_acc.append(e.mean(0) @ u)
        m = float(e.mean(0) @ u) if mc is None else float(np.mean(m_acc))
        rows.append([B, m, system.energy(e, per_spin=True)])
        if log and k % max(1, len(Bs) // 10) == 0:
            log(f"  B = {B:+8.3f} T   m.u = {m:+.4f}")
    return np.array(rows), e


def field_cooling(system, e, T_start, T_end, nT=30, sweeps=1000, B=None, seed=1, log=None):
    from .montecarlo import MetropolisMC
    if B is not None:
        system.set_field(B)
    mc = MetropolisMC(system, T_start, seed)
    rows = []
    for T in np.linspace(T_start, T_end, nT):
        mc.T = float(T)
        ms, Es = [], []
        for n in range(sweeps):
            e = mc.sweep(e)
            if n >= sweeps // 2 and n % 2 == 0:
                ms.append(e.mean(0))
                Es.append(system.energy(e, per_spin=True))
        m = np.mean(ms, 0)
        rows.append([T, *m, np.linalg.norm(m), system.order_parameter(e), np.mean(Es)])
        if log:
            log(f"  T = {T:8.2f} K  m = {np.round(m, 4).tolist()}  E/N = {np.mean(Es):.5f} meV")
    return np.array(rows), e


def anneal(system, e, T_start, T_end=0.1, nT=40, sweeps=200, method="lbfgs", seed=1, log=None):
    from .minimize import minimize
    from .montecarlo import MetropolisMC
    mc = MetropolisMC(system, T_start, seed)
    for T in np.geomspace(T_start, max(T_end, 1e-3), nT):
        mc.T = float(T)
        for _ in range(sweeps):
            e = mc.sweep(e)
    e, info = minimize(system, e, method, tol=1e-6)
    if log:
        log(f"  annealed {T_start} -> {T_end} K, relaxed: E/N = {system.energy(e, True):.6f} meV")
    return e


def fmr(system, e, B_pulse=0.01, direction=(1, 0, 0), alpha=0.001, dt=2e-3, steps=20000, every=5, log=None,
        integrator="heun"):
    """Ring-down after a short in-plane field step; returns (freq GHz, |FFT m|), f_peak, table."""
    from .llg import LLGSolver
    from .minimize import minimize
    e, _ = minimize(system, e, "lbfgs", tol=1e-7)
    p = np.asarray(direction, float)
    p /= np.linalg.norm(p)
    sv = LLGSolver(system, alpha, dt, 0.0, integrator=integrator)
    t0 = 5 * dt
    sv.field_func = lambda t: (B_pulse * p if t < t0 else np.zeros(3))
    ts, ms = [], []
    for n in range(steps):
        e = sv.step(e)
        if n % every == 0:
            ts.append(sv.time)
            ms.append(e.mean(0))
    ms = np.array(ms)
    sig = ms[:, np.argmax(np.abs(p))] - ms[:, np.argmax(np.abs(p))].mean()
    sig = sig * np.hanning(len(sig))
    F = np.abs(np.fft.rfft(sig))
    f = np.fft.rfftfreq(len(sig), d=every * dt) * 1e3        # GHz
    k = int(np.argmax(F[1:]) + 1)
    if 1 <= k < len(F) - 1:                                   # parabolic peak interpolation
        y0, y1, y2 = np.log(F[k - 1:k + 2] + 1e-300)
        k = k + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
    fpk = float(np.interp(k, np.arange(len(f)), f))
    if log:
        log(f"  FMR peak at {fpk:.3f} GHz")
    return (f, F), fpk, np.column_stack([ts, ms])


def eigenmodes(system, e, nmodes=10):
    """Lowest Hessian eigenvalues (meV) and LLG eigenfrequencies (GHz) of a (relaxed) state (dense; N < ~3000)."""
    from .htst import hessian, llg_frequencies
    H, basis = hessian(system, e)
    ev, vec = np.linalg.eigh(H)
    f = llg_frequencies(system, e, H)
    return ev[:nmodes], f[:nmodes], vec[:, :nmodes], basis


def run_program_config(cfg, base=".", log=print):
    from .plotstyle import mpl, savefig
    c = dict(cfg.get("program", {}))
    kind = c.get("type", "hysteresis")
    outdir = _out(base, c, f"{kind}_out")
    system = _setup(cfg, base, log, c.get("field", [0, 0, 0]))
    e = _init(system, c)
    plt = mpl()
    if kind == "hysteresis":
        rows, e = hysteresis(system, e, float(c.get("Bmax", 2.0)), int(c.get("npts", 41)), c.get("direction", [0, 0, 1]),
                             float(c.get("temperature", 0.0)), c.get("method", "lbfgs"), int(c.get("mc_sweeps", 500)),
                             int(c.get("seed", 1)), log)
        np.savetxt(os.path.join(outdir, "hysteresis.csv"), rows, delimiter=",", header="B_T,m_u,E_per_spin_meV", comments="")
        fig, ax = plt.subplots(figsize=(4.2, 3.4))
        ax.plot(rows[:, 0], rows[:, 1], "-o", ms=2.5)
        ax.set(xlabel="B (T)", ylabel=r"$m\cdot\hat u$", title="hysteresis loop")
        ax.axhline(0, lw=0.5, color="0.6")
        ax.axvline(0, lw=0.5, color="0.6")
        savefig(fig, os.path.join(outdir, "hysteresis.png"))
        up = rows[: len(rows) // 2 + 1]
        i = np.where(np.diff(np.sign(up[:, 1])) != 0)[0]
        if len(i):
            log(f"  coercive field (descending branch) ~ {abs(up[i[0], 0]):.3f} T")
    elif kind == "field_cooling":
        rows, e = field_cooling(system, e, float(c["T_start"]), float(c.get("T_end", 1.0)), int(c.get("nT", 30)),
                                int(c.get("sweeps", 1000)), c.get("field"), int(c.get("seed", 1)), log)
        np.savetxt(os.path.join(outdir, "field_cooling.csv"), rows, delimiter=",",
                   header="T_K,mx,my,mz,m,order_parameter,E_per_spin_meV", comments="")
        fig, ax = plt.subplots(figsize=(4.2, 3.4))
        ax.plot(rows[:, 0], rows[:, 4], "-o", ms=3, label="|m|")
        ax.plot(rows[:, 0], rows[:, 5], "-s", ms=3, label="order parameter")
        ax.set(xlabel="T (K)", ylabel="m", title="field cooling")
        ax.legend()
        savefig(fig, os.path.join(outdir, "field_cooling.png"))
    elif kind == "anneal":
        e = anneal(system, e, float(c.get("T_start", 100.0)), float(c.get("T_end", 0.1)), int(c.get("nT", 40)),
                   int(c.get("sweeps", 200)), c.get("method", "lbfgs"), int(c.get("seed", 1)), log)
    elif kind == "minimize":
        from .minimize import minimize
        e, info = minimize(system, e, c.get("method", "lbfgs"), float(c.get("tol", 1e-6)), log=log)
        log(f"  {c.get('method', 'lbfgs')}: {info}")
    elif kind == "fmr":
        (f, F), fpk, tab = fmr(system, e, float(c.get("B_pulse", 0.01)), c.get("direction", [1, 0, 0]),
                               float(c.get("alpha", 1e-3)), float(c.get("dt", 2e-3)), int(c.get("steps", 20000)),
                               int(c.get("every", 5)), log, c.get("integrator", "heun"))
        np.savetxt(os.path.join(outdir, "fmr_spectrum.csv"), np.column_stack([f, F]), delimiter=",",
                   header="f_GHz,amplitude", comments="")
        fig, ax = plt.subplots(figsize=(4.2, 3.4))
        ax.plot(f, F / F.max())
        ax.axvline(fpk, ls="--", lw=0.8, color="0.4")
        ax.set(xlabel="f (GHz)", ylabel="|FFT m| (norm.)", xlim=(0, min(f[-1], 4 * fpk)), title=f"FMR: {fpk:.2f} GHz")
        savefig(fig, os.path.join(outdir, "fmr.png"))
    elif kind == "cmc":
        from .cmc import anisotropy_vs_T, constrained_mc
        Ts = c.get("temperatures", [1.0, 10.0, 50.0])
        if c.get("torque_curve", False):
            ths = np.linspace(0, np.pi, int(c.get("nangles", 13)))
            tq = [constrained_mc(system, float(Ts[0]), th, therm=int(c.get("therm", 1000)),
                                 meas=int(c.get("meas", 2000)))["torque"] for th in ths]
            np.savetxt(os.path.join(outdir, "torque_curve.csv"), np.column_stack([ths, tq]), delimiter=",",
                       header="theta_rad,torque_meV_per_spin", comments="")
        R = anisotropy_vs_T(system, Ts, log=log, therm=int(c.get("therm", 1000)), meas=int(c.get("meas", 2000)))
        np.savetxt(os.path.join(outdir, "K_of_T.csv"), R, delimiter=",", header="T_K,m,K_meV_per_spin,K_err", comments="")
        fig, ax = plt.subplots(figsize=(4.2, 3.4))
        ax.errorbar(R[:, 1], R[:, 2] / R[0, 2], R[:, 3] / R[0, 2], fmt="o", label="CMC")
        mm = np.linspace(R[:, 1].min(), 1, 50)
        ax.plot(mm, mm**3, "--", label=r"$m^3$ (Callen-Callen)")
        ax.set(xlabel="m(T)", ylabel="K(T)/K(T$_0$)", title="constrained Monte Carlo")
        ax.legend()
        savefig(fig, os.path.join(outdir, "K_of_T.png"))
    elif kind == "eigenmodes":
        from .minimize import minimize
        e, _ = minimize(system, e, "lbfgs", tol=1e-8)
        ev, f, vec, basis = eigenmodes(system, e, int(c.get("nmodes", 10)))
        np.savetxt(os.path.join(outdir, "eigenmodes.csv"), np.column_stack([ev, f]), delimiter=",",
                   header="hessian_eigenvalue_meV,llg_frequency_GHz", comments="")
        log("  lowest Hessian eigenvalues (meV): " + ", ".join(f"{x:.4g}" for x in ev))
        log("  LLG eigenfrequencies (GHz):       " + ", ".join(f"{x:.4g}" for x in f))
    elif kind == "lsf":
        from .montecarlo import LSFMonteCarlo
        rows = []
        M = system.ref.copy()
        for T in c.get("temperatures", [1.0, 10.0, 50.0]):
            mc = LSFMonteCarlo(system, float(T), int(c.get("seed", 1)), float(c.get("sigma", 0.15)),
                               float(c.get("b", 1.0)), measure=c.get("measure", "d3M"))
            ss, ms = [], []
            for n in range(int(c.get("sweeps", 2000))):
                M = mc.sweep(M)
                if n >= int(c.get("sweeps", 2000)) // 2:
                    ss.append(np.linalg.norm(M, axis=1).mean())
                    ms.append(np.linalg.norm((system.signs[:, None] * M).mean(0)))
            rows.append([T, np.mean(ms), np.mean(ss)])
            log(f"  LSF T = {T:7.2f} K: order parameter = {np.mean(ms):.4f}, <s> = {np.mean(ss):.4f}")
        np.savetxt(os.path.join(outdir, "lsf.csv"), np.array(rows), delimiter=",", header="T_K,m,s_mean", comments="")
        e = normalize(M)
    else:
        raise ValueError(f"unknown program '{kind}'")
    _write_spins(outdir, "final", system, e)
    log(f"program '{kind}' output written to {outdir}/")
    return dict(system=system, spins=e, outdir=outdir)
