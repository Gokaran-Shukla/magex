"""Classical Monte Carlo for the magnetic ordering (Curie / Neel) temperature.

Metropolis importance sampling of exp(-E/k_BT) for the full spin model (exchange tensors,
DMI, single-ion anisotropy, field).  Sites are graph-coloured so that no two sites of one
colour interact; all sites of a colour are then updated simultaneously (exact, vectorised).

Moves per site and sweep:
  * Gaussian trial move  e' = (e + sigma xi)/|...|  (symmetric; sigma adapted during
    thermalisation to ~50% acceptance, Alzate-Cardona et al. 2019),
  * optional over-relaxation  e' = 2 (e.h) h/|h|^2 - e  about the exchange field
    (microcanonical for Heisenberg exchange, accepted with Metropolis when SIA/field/DMI
    make it non-conserving).

Observables (per spin):  E, C = (<E^2>-<E>^2)/(N k_B T^2)  [k_B],
order parameter m = |sum_i s_i e_i|/N  (s_i from the magex reference: FM -> magnetisation,
AFM -> staggered/Neel vector), chi = N(<m^2>-<m>^2)/(k_B T)  [1/meV],
Binder cumulant U4 = 1 - <m^4>/(3<m^2>^2).  Errors from 10-bin blocking.

T_c / T_N: Binder-cumulant crossings of successive system sizes (finite-size scaling), and
the peaks of chi and C of the largest system.
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import scipy.sparse as sp

from .spinmodel import K_B, MU_B, SpinSystem, mean_field_tc, normalize


def greedy_coloring(N, bonds):
    """Colour array (N,) such that bonded sites differ (bonds: (M,2) src,dst)."""
    if len(bonds) == 0:
        return np.zeros(N, int)
    adj = sp.csr_matrix((np.ones(len(bonds)), (bonds[:, 0], bonds[:, 1])), shape=(N, N))
    adj = ((adj + adj.T) > 0).tocsr()
    color = -np.ones(N, int)
    indptr, indices = adj.indptr, adj.indices
    for v in range(N):
        used = set(color[indices[indptr[v]:indptr[v + 1]]].tolist())
        c = 0
        while c in used:
            c += 1
        color[v] = c
    return color


class MetropolisMC:
    def __init__(self, system: SpinSystem, temperature, seed=None, sigma=0.6, overrelax=1):
        self.sys = system
        self.T = float(temperature)
        self.rng = np.random.default_rng(seed)
        self.sigma = float(sigma)
        self.overrelax = int(overrelax)
        col = greedy_coloring(system.N, system.bonds)
        self.groups = []
        J = system.Jsp.tocsr()
        for c in range(col.max() + 1):
            idx = np.where(col == c)[0]
            rows = (3 * idx[:, None] + np.arange(3)[None, :]).ravel()
            self.groups.append((idx, J[rows], system.A[idx], system.mu[idx]))
        self.ncolors = len(self.groups)
        self.accepted = 0
        self.tried = 0

    def _dE(self, idx, A, mu, eo, en, h):
        de = en - eo
        dE = -2.0 * np.einsum("ij,ij->i", de, h) - MU_B * mu * (de @ self.sys.B)
        if self.sys.has_A:
            dE += np.einsum("ia,iab,ib->i", en, A, en) - np.einsum("ia,iab,ib->i", eo, A, eo)
        if self.sys.has_higher:
            e = self._e
            dE += self.sys.local_higher_energy(idx, en, e) - self.sys.local_higher_energy(idx, eo, e)
        return dE

    def sweep(self, e):
        self._e = e
        beta = 1.0 / (K_B * self.T) if self.T > 0 else np.inf
        flat = e.reshape(-1)
        for idx, Jc, A, mu in self.groups:
            h = (Jc @ flat).reshape(-1, 3)
            eo = e[idx]
            en = normalize(eo + self.sigma * self.rng.normal(size=eo.shape))
            dE = self._dE(idx, A, mu, eo, en, h)
            acc = (dE <= 0) | (self.rng.random(len(idx)) < np.exp(-np.clip(dE * beta, 0, 700)))
            e[idx[acc]] = en[acc]
            self.accepted += int(acc.sum())
            self.tried += len(idx)
        for _ in range(self.overrelax):
            for idx, Jc, A, mu in self.groups:
                h = (Jc @ flat).reshape(-1, 3)
                eo = e[idx]
                hh = np.einsum("ij,ij->i", h, h)
                ok = hh > 1e-14
                en = eo.copy()
                en[ok] = 2.0 * (np.einsum("ij,ij->i", eo[ok], h[ok]) / hh[ok])[:, None] * h[ok] - eo[ok]
                en = normalize(en)
                dE = self._dE(idx, A, mu, eo, en, h)
                acc = (dE <= 1e-12) | (self.rng.random(len(idx)) < np.exp(-np.clip(dE * beta, 0, 700)))
                e[idx[acc]] = en[acc]
        return e

    def acceptance(self, reset=True):
        a = self.accepted / max(self.tried, 1)
        if reset:
            self.accepted = self.tried = 0
        return a


def _blocks(x, nb=10):
    n = len(x) // nb
    if n < 1:
        return np.array([x.mean()])
    return x[: n * nb].reshape(nb, n).mean(1)


def _point(args):
    """One (system size, temperature) Monte Carlo run (picklable for process pools)."""
    uc, L, pbc, B, T, c, seed = args
    system = SpinSystem(uc, L, pbc, B)
    mc = MetropolisMC(system, T, seed, c["sigma"], c["overrelax"])
    e = system.init_spins(c["init"], c["init_axis"], seed)
    t0 = time.time()
    for n in range(c["therm_sweeps"]):
        e = mc.sweep(e)
        if (n + 1) % 50 == 0:
            a = mc.acceptance()
            mc.sigma = float(np.clip(mc.sigma * 0.5 / max(1 - a, 0.05), 0.02, 4.0))
    mc.acceptance()
    Es, ms, mvec = [], [], []
    for n in range(c["meas_sweeps"]):
        e = mc.sweep(e)
        if n % c["sample_every"] == 0:
            Es.append(system.energy(e))
            v = (system.signs[:, None] * e).sum(0) / system.N
            mvec.append(v)
            ms.append(np.linalg.norm(v))
    acc = mc.acceptance()
    E = np.array(Es)
    m = np.array(ms)
    N = system.N
    kT = K_B * T

    def obs(E, m):
        C = (np.mean(E**2) - np.mean(E) ** 2) / (N * kT**2)      # specific heat per spin, units of k_B
        chi = N * (np.mean(m**2) - np.mean(m) ** 2) / kT
        U4 = 1 - np.mean(m**4) / (3 * np.mean(m**2) ** 2)
        return np.mean(E) / N, C, np.mean(m), chi, U4

    full = obs(E, m)
    nb = 10
    n = len(E) // nb
    bl = np.array([obs(E[k * n:(k + 1) * n], m[k * n:(k + 1) * n]) for k in range(nb)]) if n > 2 else None
    err = bl.std(0, ddof=1) / np.sqrt(nb) if bl is not None else np.zeros(5)
    return dict(L=list(L), N=N, T=T, E=full[0], C=full[1], m=full[2], chi=full[3], U4=full[4],
                dE=err[0], dC=err[1], dm=err[2], dchi=err[3], dU4=err[4], acceptance=acc,
                sigma=mc.sigma, ncolors=mc.ncolors, mvec=np.mean(mvec, 0).tolist(),
                seconds=time.time() - t0, spins=e if c.get("keep_spins") else None)


def _peak(T, y):
    """Peak position by a parabola through the maximum and its neighbours."""
    k = int(np.argmax(y))
    if 0 < k < len(T) - 1:
        a, b, c = np.polyfit(T[k - 1:k + 2], y[k - 1:k + 2], 2)
        if a < 0:
            t = -b / (2 * a)
            if T[k - 1] <= t <= T[k + 1]:
                return float(t)
    return float(T[k])


def _crossings(T, u1, u2):
    """Temperatures where the larger system's U4 (u2) drops below the smaller one's (u1).

    Below T_C the larger system has the larger cumulant (d = u1 - u2 < 0), above it the smaller
    one, so only - -> + sign changes are physical crossings.
    """
    d = np.asarray(u1) - np.asarray(u2)
    out = []
    for k in range(len(T) - 1):
        if d[k] < 0 <= d[k + 1]:
            out.append(T[k] - d[k] * (T[k + 1] - T[k]) / (d[k + 1] - d[k]))
    return [float(x) for x in out]


def estimate_tc(rows):
    sizes = sorted({tuple(r["L"]) for r in rows}, key=lambda L: np.prod(L))
    est = dict(chi_peak={}, C_peak={}, binder=[])
    for L in sizes:
        rs = sorted([r for r in rows if tuple(r["L"]) == L], key=lambda r: r["T"])
        T = np.array([r["T"] for r in rs])
        est["chi_peak"][str(L)] = _peak(T, np.array([r["chi"] for r in rs]))
        est["C_peak"][str(L)] = _peak(T, np.array([r["C"] for r in rs]))
    for L1, L2 in zip(sizes[:-1], sizes[1:]):
        r1 = sorted([r for r in rows if tuple(r["L"]) == L1], key=lambda r: r["T"])
        r2 = sorted([r for r in rows if tuple(r["L"]) == L2], key=lambda r: r["T"])
        T1, T2 = [r["T"] for r in r1], [r["T"] for r in r2]
        if T1 != T2:
            continue
        cr = _crossings(np.array(T1), [r["U4"] for r in r1], [r["U4"] for r in r2])
        # discard crossings in the paramagnetic noise (U4 -> 4/9 for 3-component spins)
        good = [x for x in cr if 0.47 < np.interp(x, T1, [r["U4"] for r in r1]) < 0.67]
        est["binder"].append(dict(sizes=[list(L1), list(L2)], T=good))
    big = str(sizes[-1])
    Tchi = est["chi_peak"][big]
    if est["binder"] and est["binder"][-1]["T"]:
        cands = est["binder"][-1]["T"]
        est["Tc"] = float(min(cands, key=lambda x: abs(x - Tchi)))
        est["method"] = "Binder-cumulant crossing of the two largest sizes"
    else:
        est["Tc"] = est["chi_peak"][big]
        est["method"] = f"susceptibility peak of the largest size {sizes[-1]}"
    return est


def run_mc_config(cfg, base=".", log=print):
    from .spinmodel import build_from_config
    uc, sm = build_from_config(cfg, base)
    c = dict(dict(temperatures="auto", sizes=None, therm_sweeps=2000, meas_sweeps=4000, sample_every=2,
                  field=[0.0, 0.0, 0.0], init="reference", init_axis=[0, 0, 1], seed=7, nproc=None,
                  sigma=0.6, overrelax=1, outdir="mc_out", refine=False), **cfg.get("mc", {}))
    outdir = c["outdir"] if os.path.isabs(c["outdir"]) else os.path.join(base, c["outdir"])
    os.makedirs(outdir, exist_ok=True)
    pbc = sm.get("pbc", [True, True, True])
    sizes = c["sizes"] or [sm.get("supercell", [8, 8, 8])]
    B = np.asarray(c["field"], float)
    Tmf = mean_field_tc(uc)
    probe = SpinSystem(uc, sizes[-1], pbc, B)
    kind = probe.kind()
    label = {"ferromagnet": "Curie temperature T_C", "antiferromagnet": "Neel temperature T_N",
             "ferrimagnet": "ordering (Neel) temperature of the ferrimagnet"}[kind]
    log(probe.summary())
    log(f"mean-field estimate: T_MF = {Tmf:.1f} K   ({kind}; order parameter = "
        f"{'magnetisation' if kind == 'ferromagnet' else 'staggered (Neel) vector from the magex reference'})")
    Ts = c["temperatures"]
    if Ts == "auto":
        Ts = list(np.round(np.linspace(0.25, 1.15, 19) * Tmf, 3))
    elif isinstance(Ts, dict):
        Ts = list(np.linspace(Ts["start"], Ts["stop"], int(Ts["n"])))
    Ts = [float(t) for t in Ts]
    nproc = c["nproc"] or os.cpu_count()

    method = c.get("method", "metropolis")
    if method == "wang_landau":
        return _run_wl(uc, sizes, pbc, B, Ts, c, outdir, label, Tmf, log)

    def run_grid(Tlist):
        jobs = [(uc, tuple(L), pbc, B, T, c, int(c["seed"]) + 1000 * k + j)
                for k, L in enumerate(sizes) for j, T in enumerate(Tlist)]
        log(f"running {len(jobs)} Monte Carlo points ({len(sizes)} sizes x {len(Tlist)} temperatures) "
            f"on {min(nproc, len(jobs))} processes ...")
        t0 = time.time()
        if method == "parallel_tempering":
            from .advanced_mc import parallel_tempering
            out = []
            for k, L in enumerate(sizes):
                system = SpinSystem(uc, tuple(L), pbc, B)
                rows_L, swap = parallel_tempering(system, Tlist, c["therm_sweeps"], c["meas_sweeps"], c["sample_every"],
                                                  int(c.get("swap_every", 5)), int(c["seed"]) + 1000 * k, c["init"],
                                                  c["init_axis"], c["sigma"], c["overrelax"], log)
                out += rows_L
        elif nproc > 1 and len(jobs) > 1:
            with ProcessPoolExecutor(max_workers=min(nproc, len(jobs))) as ex:
                out = list(ex.map(_point, jobs))
        else:
            out = [_point(j) for j in jobs]
        log(f"  done in {time.time() - t0:.1f} s")
        return out

    rows = run_grid(Ts)
    est = estimate_tc(rows)
    if c["refine"]:
        t = est["Tc"]
        Tr = [float(x) for x in np.linspace(0.85 * t, 1.15 * t, 13)]
        rows_r = run_grid(Tr)
        est_r = estimate_tc(rows_r)
        rows = rows + rows_r
        est = est_r
        est["refined"] = True
    est["T_MF"] = Tmf
    try:
        from .quantum import quantum_rescale, rpa_tc
        T_rpa, _ = rpa_tc(uc)
        est["T_RPA"] = T_rpa
        est["T_quantum_rescaled"], est["S"] = quantum_rescale(est["Tc"], uc.moments)
        est["T_RPA_quantum"] = quantum_rescale(T_rpa, uc.moments)[0]
        log(f"  RPA (Tyablikov) estimate: {T_rpa:.1f} K;  spin S = {est['S']:.2f}: quantum-rescaled "
            f"(S+1)/S x T_MC = {est['T_quantum_rescaled']:.1f} K, x T_RPA = {est['T_RPA_quantum']:.1f} K "
            "(localized moments only)")
    except Exception as exc:  # pragma: no cover
        log(f"  RPA estimate skipped: {exc}")
    est["kind"] = kind
    est["label"] = label
    est["field"] = B.tolist()
    est["model"] = uc.info
    _write(rows, est, outdir)
    plot_mc(rows, est, os.path.join(outdir, "mc_observables.png"))
    log(f"{label} = {est['Tc']:.1f} K  ({est['method']});  T_MF = {Tmf:.1f} K;  "
        f"ratio T/T_MF = {est['Tc'] / Tmf:.3f}")
    for k, v in est["chi_peak"].items():
        log(f"  chi peak L={k}: {v:.1f} K   C peak: {est['C_peak'][k]:.1f} K")
    for b in est["binder"]:
        log(f"  Binder crossing {b['sizes'][0]} / {b['sizes'][1]}: {[round(x, 1) for x in b['T']]} K")
    if np.linalg.norm(B) > 0:
        log("  note: a finite field smears the transition; the estimates above are crossover temperatures")
    log(f"Monte Carlo output written to {outdir}/")
    return dict(rows=rows, estimate=est, outdir=outdir)


def _run_wl(uc, sizes, pbc, B, Ts, c, outdir, label, Tmf, log):
    """Wang-Landau density of states per size; canonical thermodynamics on a fine T grid."""
    from .advanced_mc import wang_landau, wl_thermodynamics
    from .plotstyle import OKABE_ITO, mpl, savefig
    Tf = np.linspace(min(Ts), max(Ts), 400)
    res = []
    for k, L in enumerate(sizes):
        system = SpinSystem(uc, tuple(L), pbc, B)
        wl = wang_landau(system, int(c.get("wl_bins", 120)), c.get("wl_window"), float(c.get("wl_lnf", 1e-7)),
                         float(c.get("wl_flatness", 0.7)), c["sigma"], int(c.get("wl_max_sweeps", 2000000)),
                         int(c["seed"]) + k, log)
        th = wl_thermodynamics(wl, Tf)
        res.append((L, wl, th))
        np.savetxt(os.path.join(outdir, f"dos_L{'x'.join(map(str, L))}.csv"), np.column_stack([wl["E"], wl["lng"]]),
                   delimiter=",", header="E_meV,ln_g", comments="")
        np.savetxt(os.path.join(outdir, f"thermo_L{'x'.join(map(str, L))}.csv"),
                   np.array([[r["T"], r["E"], r["C"], r["S"], r["F"]] for r in th]), delimiter=",",
                   header="T_K,E_per_spin_meV,C_per_spin_kB,S_per_spin_kB,F_per_spin_meV", comments="")
    lnf_req = float(c.get("wl_lnf", 1e-7))
    conv = [r for r in res if r[1]["lnf"] <= 10 * lnf_req]
    for L, wl, th in res:
        if wl["lnf"] > 10 * lnf_req:
            log(f"  WARNING: Wang-Landau not converged for L = {L} (ln f = {wl['lnf']:.1e}); increase "
                "wl_max_sweeps or use a smaller system")
    L, wl, th = (conv or res)[-1]
    C = np.array([r["C"] for r in th])
    Tc = float(Tf[np.argmax(C)])
    est = dict(Tc=Tc, method=f"specific-heat peak of the Wang-Landau density of states (L = {L})", T_MF=Tmf,
               label=label, sizes=[list(r[0]) for r in res])
    with open(os.path.join(outdir, "tc_estimate.json"), "w") as f:
        json.dump(est, f, indent=2, default=float)
    plt = mpl()
    fig, axs = plt.subplots(1, 2, figsize=(8.2, 3.2), constrained_layout=True)
    for k, (L, wl, th) in enumerate(res):
        lab = "×".join(str(x) for x in L if x > 1)
        axs[0].plot(np.asarray(wl["E"]) / wl["N"], np.asarray(wl["lng"]) / wl["N"], color=OKABE_ITO[k], label=f"L = {lab}")
        axs[1].plot(Tf, [r["C"] for r in th], color=OKABE_ITO[k], label=f"L = {lab}")
    axs[0].set_xlabel("$E/N$ (meV)")
    axs[0].set_ylabel(r"$\ln g(E)/N$")
    axs[1].set_xlabel("$T$ (K)")
    axs[1].set_ylabel("$C$ ($k_B$/spin)")
    axs[1].axvline(Tc, color="k", ls="--", lw=0.9)
    for ax in axs:
        ax.legend()
    fig.suptitle(f"Wang-Landau: {label} ~ {Tc:.1f} K (C peak)")
    savefig(fig, os.path.join(outdir, "wang_landau.png"))
    plt.close(fig)
    log(f"{label} ~ {Tc:.1f} K from the specific-heat peak (Wang-Landau, L = {L}); T_MF = {Tmf:.1f} K")
    log(f"Monte Carlo output written to {outdir}/")
    return dict(est=est, wl=res)


def _write(rows, est, outdir):
    keys = ["T", "E", "dE", "C", "dC", "m", "dm", "chi", "dchi", "U4", "dU4", "acceptance", "sigma"]
    with open(os.path.join(outdir, "mc_results.csv"), "w") as f:
        f.write("L1,L2,L3,N," + ",".join(keys) + "\n")
        for r in sorted(rows, key=lambda r: (np.prod(r["L"]), r["T"])):
            f.write(",".join(map(str, r["L"])) + f",{r['N']}," + ",".join(f"{r[k]:.8g}" for k in keys) + "\n")
    with open(os.path.join(outdir, "tc_estimate.json"), "w") as f:
        json.dump(est, f, indent=2, default=float)


def plot_mc(rows, est, path):
    from .plotstyle import OKABE_ITO, mpl, savefig
    plt = mpl()
    sizes = sorted({tuple(r["L"]) for r in rows}, key=lambda L: np.prod(L))
    markers = "osD^v<>"
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.6), constrained_layout=True)
    spec = [("m", "dm", "order parameter $m$"), ("chi", "dchi", r"$\chi$ (meV$^{-1}$)"),
            ("C", "dC", "$C$ ($k_B$/spin)"), ("U4", "dU4", "Binder cumulant $U_4$")]
    for n, (ax, (k, dk, lab)) in enumerate(zip(axs.ravel(), spec)):
        for c, (mk, L) in enumerate(zip(markers, sizes)):
            rs = sorted([r for r in rows if tuple(r["L"]) == L], key=lambda r: r["T"])
            T = [r["T"] for r in rs]
            Ls = "×".join(str(x) for x in L if x > 1) or "1"
            ax.errorbar(T, [r[k] for r in rs], yerr=[r[dk] for r in rs], fmt=mk + "-", ms=3.5, lw=1.1,
                        color=OKABE_ITO[c % len(OKABE_ITO)], mfc="white", mew=1.0, capsize=1.5,
                        label=f"$L$ = {Ls} ($N$ = {rs[0]['N']})")
        ax.axvline(est["Tc"], color="k", ls="--", lw=0.9)
        ax.axvline(est["T_MF"], color="0.6", ls=":", lw=0.9)
        ax.set_xlabel("$T$ (K)")
        ax.set_ylabel(lab)
        ax.set_title(f"({chr(97 + n)})", loc="left")
    axs[0, 0].legend(loc="best")
    sym = "T_C" if est["label"].startswith("Curie") else "T_N"
    axs[0, 0].text(0.97, 0.95, f"${sym[0]}_{sym[2]}$ = {est['Tc']:.0f} K\n$T_{{MF}}$ = {est['T_MF']:.0f} K",
                   transform=axs[0, 0].transAxes, ha="right", va="top")
    fig.suptitle(f"Monte Carlo: {est['label']} = {est['Tc']:.1f} K  ({est['method']})", fontsize=9.5)
    savefig(fig, path)
    plt.close(fig)
