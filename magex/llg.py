"""Atomistic stochastic Landau-Lifshitz-Gilbert dynamics.

    de_i/dt = -gamma/(1+alpha^2) [ e_i x B_i + alpha e_i x (e_i x B_i) ]

with B_i = B_eff,i + b_i the effective field (tesla) plus, at T > 0, a Gaussian white-noise
thermal field with <b_ia(t) b_jb(t')> = 2 alpha k_B T / (gamma mu_i mu_B) delta_ij delta_ab
delta(t-t') (Brown).  Integrated with the stochastic Heun scheme (Stratonovich-consistent, the
standard predictor-corrector of atomistic spin dynamics), renormalising |e_i| = 1 after each
stage.  Time in ps, fields in T, energies in meV.
"""
from __future__ import annotations

import os
import time

import numpy as np

from .spinmodel import GAMMA, K_B, MU_B, normalize


def llg_rhs(e, B, alpha, precession=True):
    """de/dt (1/ps) for unit spins e (N,3) in fields B (N,3) tesla."""
    pre = -GAMMA / (1.0 + alpha * alpha)
    exB = np.cross(e, B)
    out = alpha * np.cross(e, exB)
    if precession:
        out = out + exB
    return pre * out


class LLGSolver:
    def __init__(self, system, alpha=0.1, dt=1e-3, temperature=0.0, seed=None, precession=True,
                 thermostat="classical"):
        """thermostat: "classical" (white Brown noise, Boltzmann statistics) or "quantum"
        (coloured Planck-spectrum noise, Bose statistics of the magnons; see magex.quantum)."""
        self.sys = system
        self.alpha = float(alpha)
        self.dt = float(dt)
        self.T = float(temperature)
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.precession = bool(precession)
        self.thermostat = thermostat
        self._qnoise = None
        self.time = 0.0

    def thermal_field(self):
        if self.T <= 0:
            return 0.0
        sigma2 = 2.0 * self.alpha * K_B * self.T / (GAMMA * self.sys.mu * MU_B * self.dt)
        if self.thermostat == "quantum":
            if self._qnoise is None or self._qnoise.T != self.T:
                from .quantum import QuantumNoise
                self._qnoise = QuantumNoise(self.sys.N, sigma2, self.dt, self.T, seed=self.rng.integers(1 << 31))
            return self._qnoise()
        return np.sqrt(sigma2)[:, None] * self.rng.normal(size=(self.sys.N, 3))

    def step(self, e):
        """One stochastic-Heun step; returns new spins."""
        b = self.thermal_field()
        B1 = self.sys.effective_field(e) + b
        k1 = llg_rhs(e, B1, self.alpha, self.precession)
        ep = normalize(e + self.dt * k1)
        B2 = self.sys.effective_field(ep) + b
        k2 = llg_rhs(ep, B2, self.alpha, self.precession)
        self.time += self.dt
        return normalize(e + 0.5 * self.dt * (k1 + k2))

    def max_torque(self, e):
        """max_i |e_i x B_eff,i| (T): zero at an equilibrium."""
        return float(np.max(np.linalg.norm(np.cross(e, self.sys.effective_field(e)), axis=1)))

    def run(self, e, steps, output_every=100, snapshots=0, callback=None, torque_tol=None, log=None):
        """Integrate ``steps`` steps.  Returns (spins, table, snapshot list).

        table columns: t(ps), E/N (meV), <ex>, <ey>, <ez>, order parameter, max torque (T)
        torque_tol: stop early (T = 0 only) when the max torque drops below this value.
        """
        rows, snaps = [], []
        snap_every = max(1, steps // snapshots) if snapshots else 0
        t0 = time.time()
        for n in range(steps + 1):
            if n % output_every == 0 or n == steps:
                m = e.mean(0)
                tq = self.max_torque(e)
                rows.append([self.time, self.sys.energy(e, per_spin=True), *m, self.sys.order_parameter(e), tq])
                if log and (n % (output_every * 10) == 0 or n == steps):
                    log(f"  step {n:8d}  t = {self.time:9.3f} ps  E/N = {rows[-1][1]: .6f} meV  "
                        f"m = ({m[0]: .3f},{m[1]: .3f},{m[2]: .3f})  |torque|max = {tq:.2e} T")
                if torque_tol is not None and self.T == 0 and tq < torque_tol and n > 0:
                    if log:
                        log(f"  converged: max torque {tq:.2e} T < {torque_tol:g} T after {n} steps")
                    break
            if snap_every and n % snap_every == 0:
                snaps.append((self.time, e.copy()))
            if callback:
                callback(n, e)
            if n < steps:
                e = self.step(e)
        if log:
            log(f"  LLG wall time {time.time() - t0:.1f} s")
        return e, np.array(rows), snaps


# ----------------------------------------------------------------------------- driver
def _write_table(path, table):
    np.savetxt(path, table, delimiter=",", fmt="%.8g",
               header="t_ps,E_per_spin_meV,mx,my,mz,order_parameter,max_torque_T", comments="")


def save_spins(path, system, e):
    np.savez_compressed(path, spins=e, positions=system.positions, mu=system.mu,
                        sublattice=system.sublattice, cell=system.uc.cell, supercell=np.array(system.L))
    np.savetxt(path.replace(".npz", ".csv"), np.column_stack([system.positions, e]), delimiter=",",
               fmt="%.6f", header="x_A,y_A,z_A,ex,ey,ez", comments="")


def run_llg_config(cfg, base=".", field_override=None, log=print):
    """Run the [llg] section of a config. Returns dict with outputs."""
    from .spinmodel import SpinSystem, build_from_config
    from . import texture

    from .damping import resolve_alpha
    uc, sm = build_from_config(cfg, base)
    c = dict(dict(alpha=0.1, dt=1e-3, steps=10000, temperature=0.0, field=[0.0, 0.0, 0.0], init="random",
                  init_axis=[0, 0, 1], seed=1, thermostat="classical", output_every=100, snapshots=12, gif=False, precession=True,
                  plane="xy", layer=0, outdir="llg_out", torque_tol=None, compare_field=False,
                  anneal=None, init_file=None), **cfg.get("llg", {}))
    c["alpha"] = resolve_alpha(c["alpha"], base)
    outdir = c["outdir"] if os.path.isabs(c["outdir"]) else os.path.join(base, c["outdir"])
    os.makedirs(outdir, exist_ok=True)
    fields = [np.asarray(c["field"], float) if field_override is None else np.asarray(field_override, float)]
    if c["compare_field"]:
        fields = [np.zeros(3), fields[0]]
    system = SpinSystem(uc, sm.get("supercell", [20, 20, 1]), sm.get("pbc", [True, True, True]), fields[0])
    log(system.summary())
    init_file = c["init_file"]
    if init_file and not os.path.isabs(init_file):
        init_file = os.path.join(base, init_file)
    e_init = system.init_spins(c["init"], c["init_axis"], c["seed"], path=init_file)
    results = []
    for B in fields:
        tag = "B_off" if np.allclose(B, 0) else "B_on"
        system.set_field(B)
        log(f"LLG run '{tag}': B = {B.tolist()} T, alpha = {c['alpha']}, dt = {c['dt']} ps, "
            f"T = {c['temperature']} K, {c['steps']} steps")
        e = e_init.copy()
        tables = []
        solver = LLGSolver(system, c["alpha"], c["dt"], c["temperature"], c["seed"], c["precession"],
                           c["thermostat"])
        if c["anneal"]:
            # anneal = {T_start, steps}: linear cooling to the target temperature first
            a = c["anneal"]
            nseg = 10
            for k in range(nseg):
                solver.T = a["T_start"] + (c["temperature"] - a["T_start"]) * k / (nseg - 1)
                e, tab, _ = solver.run(e, int(a["steps"]) // nseg, c["output_every"])
                tables.append(tab)
            solver.T = c["temperature"]
            log(f"  annealed from {a['T_start']} K to {c['temperature']} K over {a['steps']} steps")
        e, tab, snaps = solver.run(e, int(c["steps"]), int(c["output_every"]), int(c["snapshots"]),
                                   torque_tol=c["torque_tol"], log=log)
        tables.append(tab)
        table = np.vstack(tables)
        _write_table(os.path.join(outdir, f"llg_{tag}.csv"), table)
        save_spins(os.path.join(outdir, f"spins_{tag}.npz"), system, e)
        Q = texture.topological_charge(system, e, c["plane"], c["layer"])
        title = f"{tag.replace('_', ' ')}: B = {np.round(B, 3).tolist()} T,  Q = {Q:+.2f}"
        texture.plot_texture(system, e, os.path.join(outdir, f"texture_{tag}.png"), c["plane"], c["layer"], title)
        texture.plot_timeseries(table, os.path.join(outdir, f"timeseries_{tag}.png"), title=f"LLG {tag}")
        if c["gif"] and snaps:
            texture.make_gif(system, snaps, os.path.join(outdir, f"dynamics_{tag}.gif"), c["plane"], c["layer"])
        log(f"  final E/N = {table[-1, 1]:.6f} meV, m = {np.round(table[-1, 2:5], 4).tolist()}, "
            f"order parameter = {table[-1, 5]:.4f}, topological charge Q = {Q:+.3f}")
        results.append(dict(tag=tag, B=B, spins=e, table=table, Q=Q))
    if len(results) == 2:
        texture.plot_compare(system, [r["spins"] for r in results],
                             [f"field off (Q = {results[0]['Q']:+.2f})",
                              f"B = {np.round(results[1]['B'], 3).tolist()} T (Q = {results[1]['Q']:+.2f})"],
                             os.path.join(outdir, "texture_compare.png"), c["plane"], c["layer"])
    log(f"LLG output written to {outdir}/")
    return dict(system=system, results=results, outdir=outdir)
