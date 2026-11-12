"""Geodesic nudged elastic band (GNEB) for energy barriers of magnetic transitions.

Bessarab, Uzdin & Jonsson, Comput. Phys. Commun. 196, 335 (2015).  A chain of images
x^0 ... x^P (x^0, x^P local minima) is relaxed on the product of unit spheres:

  tangent      upwind (energy-weighted) tangent, projected onto the tangent space of every spin
  force        F = -grad_perp E + (grad_perp E . t) t + k (L_{v+1,v} - L_{v,v-1}) t
  climbing     the highest image feels F = -grad_perp E + 2 (grad_perp E . t) t  (exact saddle)
  distance     geodesic L = sqrt( sum_i arccos^2(e_i . e'_i) )
  optimiser    velocity-projection (quick-min) with renormalisation of every spin

Works with atomistic SpinSystem (energies in meV) and micromagnetic Micromagnet (J) through the
Adapter classes.  Lifetimes from the Arrhenius law tau = tau_0 exp(Delta E / k_B T) with a
user-given attempt time (harmonic transition-state prefactors need the Hessian and are not
computed).
"""
from __future__ import annotations

import json
import os
import time

import numpy as np


# ----------------------------------------------------------------------------- adapters
class AtomisticAdapter:
    unit = "meV"
    kB = 8.617333262e-2                  # meV/K

    def __init__(self, system):
        self.sys = system

    def energy(self, x):
        return float(self.sys.energy(x))

    def grad(self, x):
        return self.sys.gradient(x)

    def energy_grad(self, x):
        return self.energy(x), self.grad(x)

    def to_state(self, x):
        return x

    def from_state(self, e):
        return np.asarray(e, float)


class MicromagAdapter:
    unit = "J"
    kB = 1.380649e-23

    def __init__(self, system):
        self.sys = system
        self.mask = system.magnetic
        self.w = (system.Ms * system.mesh.V)[self.mask]       # dE/dm = -Ms V B

    def _full(self, x):
        m = np.zeros(self.sys.mesh.n + (3,))
        m[self.mask] = x
        return m

    def energy(self, x):
        return float(self.sys.energies(self._full(x))["total"])

    def grad(self, x):
        B = self.sys.effective_field(self._full(x))[self.mask]
        return -self.w[:, None] * B

    def energy_grad(self, x):
        m = self._full(x)
        F = self.sys.fields(m)
        E = self.sys.energies(m, F)["total"]
        return float(E), -self.w[:, None] * sum(F.values())[self.mask]

    def to_state(self, x):
        return self._full(x)

    def from_state(self, m):
        return np.asarray(m, float)[self.mask]


# ----------------------------------------------------------------------------- geometry
def _normalize(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def _project(x, v):
    return v - np.einsum("ia,ia->i", v, x)[:, None] * x


def geodesic_distance(a, b):
    c = np.clip(np.einsum("ia,ia->i", a, b), -1.0, 1.0)
    s = np.linalg.norm(np.cross(a, b), axis=1)
    return float(np.sqrt(np.sum(np.arctan2(s, c) ** 2)))


def slerp_path(a, b, nimages):
    """Geodesic interpolation spin by spin (rotation about a x b)."""
    out = []
    ang = np.arctan2(np.linalg.norm(np.cross(a, b), axis=1), np.clip(np.einsum("ia,ia->i", a, b), -1, 1))
    axis = np.cross(a, b)
    nrm = np.linalg.norm(axis, axis=1)
    # antiparallel spins: any perpendicular axis
    bad = nrm < 1e-8
    if np.any(bad):
        trial = np.where(np.abs(a[bad, 0:1]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))
        axis[bad] = np.cross(a[bad], trial)
        nrm[bad] = np.linalg.norm(axis[bad], axis=1)
    axis = axis / nrm[:, None]
    for t in np.linspace(0, 1, nimages):
        th = (ang * t)[:, None]
        rot = a * np.cos(th) + np.cross(axis, a) * np.sin(th) + axis * np.einsum("ia,ia->i", axis, a)[:, None] * (1 - np.cos(th))
        out.append(_normalize(rot))
    out[0], out[-1] = a.copy(), b.copy()
    return out


# ----------------------------------------------------------------------------- minimiser
def minimize(adapter, x, tol=1e-6, max_iter=20000):
    """Barzilai-Borwein descent on the spheres; tol on max |grad_perp| (energy units)."""
    x = _normalize(np.asarray(x, float))
    g = _project(x, adapter.grad(x))
    tau = 1e-3 / max(np.abs(g).max(), 1e-30)
    for it in range(max_iter):
        gmax = float(np.max(np.linalg.norm(g, axis=1)))
        if gmax < tol:
            break
        xn = _normalize(x - tau * g)
        gn = _project(xn, adapter.grad(xn))
        s = (xn - x).ravel()
        y = (gn - g).ravel()
        sy = float(s @ y)
        tau = (float(s @ s) / sy if it % 2 else sy / float(y @ y)) if sy > 0 else 1e-3 / max(gmax, 1e-30)
        x, g = xn, gn
    return x, it, gmax


# ----------------------------------------------------------------------------- GNEB
class GNEB:
    def __init__(self, adapter, images, spring=1.0, climbing=True):
        self.ad = adapter
        self.x = [_normalize(np.asarray(im, float)) for im in images]
        # spring constant in units of the typical perpendicular gradient along the initial path, so that
        # the same dimensionless value works for meV (atomistic) and J (micromagnetic) energies
        gref = [float(np.max(np.linalg.norm(_project(x, adapter.grad(x)), axis=1))) for x in self.x[1:-1]]
        self.k = float(spring) * max(float(np.median(gref)) if gref else 1.0, 1e-300)
        self.climbing = climbing
        self.v = [np.zeros_like(im) for im in self.x]

    def _tangents(self, E):
        P = len(self.x)
        t = [None] * P
        for v in range(1, P - 1):
            tp = self.x[v + 1] - self.x[v]
            tm = self.x[v] - self.x[v - 1]
            Ep, E0, Em = E[v + 1], E[v], E[v - 1]
            if Ep > E0 > Em:
                tau = tp
            elif Ep < E0 < Em:
                tau = tm
            else:
                dmax = max(abs(Ep - E0), abs(Em - E0))
                dmin = min(abs(Ep - E0), abs(Em - E0))
                tau = tp * dmax + tm * dmin if Ep > Em else tp * dmin + tm * dmax
            tau = _project(self.x[v], tau)
            t[v] = tau / max(np.linalg.norm(tau), 1e-30)
        return t

    def forces(self):
        EG = [self.ad.energy_grad(x) for x in self.x]
        E = np.array([e for e, _ in EG])
        t = self._tangents(E)
        P = len(self.x)
        F = [np.zeros_like(x) for x in self.x]
        ci = int(np.argmax(E[1:-1])) + 1 if (self.climbing and P > 2) else -1
        for v in range(1, P - 1):
            g = _project(self.x[v], EG[v][1])
            gt = float(np.sum(g * t[v]))
            if v == ci:
                F[v] = -g + 2 * gt * t[v]
            else:
                Lp = geodesic_distance(self.x[v + 1], self.x[v])
                Lm = geodesic_distance(self.x[v], self.x[v - 1])
                F[v] = -g + gt * t[v] + self.k * (Lp - Lm) * t[v]
        return E, F, ci

    def run(self, tol=0.0, max_iter=20000, rel_tol=1e-4, log=None, log_every=500, max_step=0.02):
        """Stop when max |F| < tol (absolute) or < rel_tol x initial max |F|."""
        t0 = time.time()
        E, F, ci = self.forces()
        fmax0 = max(float(np.max(np.linalg.norm(f, axis=1))) for f in F[1:-1]) if len(F) > 2 else 0.0
        scale = 1.0 / max(fmax0, 1e-30)          # forces normalised so that steps are O(0.1 rad)
        it = 0
        for it in range(1, max_iter + 1):
            fmax = max(float(np.max(np.linalg.norm(f, axis=1))) for f in F[1:-1])
            if fmax < tol or fmax < rel_tol * fmax0:
                break
            for v in range(1, len(self.x) - 1):
                f = F[v] * scale
                fn = np.linalg.norm(f)
                vp = float(np.sum(self.v[v] * f))
                self.v[v] = (vp / fn**2) * f if (vp > 0 and fn > 0) else np.zeros_like(f)
                self.v[v] = self.v[v] + 0.05 * f
                disp = 0.05 * self.v[v]
                dmax = float(np.max(np.linalg.norm(disp, axis=1)))
                if dmax > max_step:                    # trust region: at most max_step rad per spin
                    disp *= max_step / dmax
                    self.v[v] *= max_step / dmax
                self.x[v] = _normalize(self.x[v] + disp)
                self.v[v] = _project(self.x[v], self.v[v])
            E, F, ci = self.forces()
            if log and it % log_every == 0:
                log(f"    GNEB it {it:6d}  max force {fmax:.3e}  barrier {E.max() - E[0]:.6g} {self.ad.unit}")
        self.E = E
        self.climb = ci
        self.iterations = it
        self.fmax = max(float(np.max(np.linalg.norm(f, axis=1))) for f in F[1:-1])
        self.seconds = time.time() - t0
        if log:
            log(f"    GNEB: {it} iterations, max force {self.fmax:.2e}, {self.seconds:.1f} s")
        return self

    def reaction_coordinate(self):
        s = [0.0]
        for a, b in zip(self.x[:-1], self.x[1:]):
            s.append(s[-1] + geodesic_distance(a, b))
        return np.array(s)

    def result(self):
        E = np.asarray(self.E)
        s = self.reaction_coordinate()
        ib = int(np.argmax(E))
        return dict(s=s.tolist(), E=E.tolist(), E_rel=(E - E[0]).tolist(), barrier_forward=float(E[ib] - E[0]),
                    barrier_backward=float(E[ib] - E[-1]), saddle_image=ib, climbing_image=int(self.climb),
                    unit=self.ad.unit, iterations=int(self.iterations), max_force=float(self.fmax))


def arrhenius(barrier, unit_kB, T, tau0=1e-9):
    """Lifetime (s) tau0 exp(barrier / k_B T)."""
    if T <= 0:
        return float("inf")
    x = barrier / (unit_kB * T)
    return float(tau0 * np.exp(x)) if x < 700 else float("inf")


def plot_path(res, path, title=None, energy_scale=1.0, energy_label=None):
    from .plotstyle import OKABE_ITO, mpl, savefig
    plt = mpl()
    s = np.array(res["s"])
    E = np.array(res["E_rel"]) * energy_scale
    fig, ax = plt.subplots(figsize=(4.4, 3.2), constrained_layout=True)
    from scipy.interpolate import PchipInterpolator
    ss = np.linspace(0, s[-1], 300)
    ax.plot(ss, PchipInterpolator(s, E)(ss), color=OKABE_ITO[0])
    ax.plot(s, E, "o", color=OKABE_ITO[0], mfc="white")
    ib = res["saddle_image"]
    ax.plot(s[ib], E[ib], "s", color=OKABE_ITO[1], ms=6, label=f"saddle: ΔE = {E[ib]:.3g}" + (f" {res['unit']}" if energy_label is None else ""))
    ax.set_xlabel("reaction coordinate (geodesic, rad)")
    ax.set_ylabel(energy_label or f"$E - E_0$ ({res['unit']})")
    ax.legend(loc="best")
    if title:
        ax.set_title(title)
    savefig(fig, path)
    plt.close(fig)


# ----------------------------------------------------------------------------- drivers
def skyrmion_on_lattice(system, center=None, radius=10.0, core=-1, helicity_deg=0.0):
    """Neel/Bloch skyrmion profile on atomistic positions (radius in A) in a +z background."""
    P = system.positions[:, :2]
    c = P.mean(0) if center is None else np.asarray(center, float)
    r = P - c
    R = np.linalg.norm(r, axis=1)
    w = radius / 2
    th = 2 * np.arctan2(np.sinh(radius / w), np.sinh(R / w))
    ph = np.arctan2(r[:, 1], r[:, 0]) + np.deg2rad(helicity_deg)
    return _normalize(np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), -core * np.cos(th)], 1))


def run_gneb_config(cfg, base=".", log=print):
    """magex gneb input.toml: [gneb] with engine = "atomistic" ([spinmodel]) or "micromag" (mesh/regions)."""
    g = dict(dict(engine="atomistic", images=16, spring=1.0, climbing=True, tol=None, max_iter=20000,
                  relax_endpoints=True, initial={"type": "skyrmion", "radius": 10.0}, final={"type": "uniform"},
                  temperature=300.0, attempt_time=1e-9, outdir="gneb_out", field=None), **cfg.get("gneb", {}))
    outdir = g["outdir"] if os.path.isabs(g["outdir"]) else os.path.join(base, g["outdir"])
    os.makedirs(outdir, exist_ok=True)
    if g["engine"] == "atomistic":
        from .spinmodel import SpinSystem, build_from_config
        uc, sm = build_from_config(cfg, base)
        system = SpinSystem(uc, sm.get("supercell", [30, 30, 1]), sm.get("pbc", [True, True, False]),
                            g["field"] or [0, 0, 0])
        log(system.summary())
        ad = AtomisticAdapter(system)

        def state(spec):
            spec = dict(spec)
            kind = spec.pop("type")
            if kind == "skyrmion":
                hel = spec.get("helicity_deg", "auto")
                cands = [0.0, 180.0, 90.0, -90.0] if hel == "auto" else [float(hel)]
                trial = [skyrmion_on_lattice(system, spec.get("center"), spec.get("radius", 10.0),
                                             spec.get("core", -1), h) for h in cands]
                return min(trial, key=ad.energy)            # helicity favoured by the DMI
            if kind == "uniform":
                return np.tile(np.asarray(spec.get("direction", [0, 0, 1]), float), (system.N, 1))
            if kind == "file":
                p = spec["path"] if os.path.isabs(spec["path"]) else os.path.join(base, spec["path"])
                return np.load(p)["spins"]
            return system.init_spins(kind, **spec)
        tol = g["tol"] or 1e-4                                   # meV per spin
        to_plot = None
    else:
        from .micromag.driver import build
        system = build(cfg, base, log)
        if g["field"] is not None:
            system.set_field(g["field"])
        log(system.summary())
        ad = MicromagAdapter(system)

        def state(spec):
            spec = dict(spec)
            kind = spec.pop("type")
            return ad.from_state(system.init(kind, **spec))
        tol = g["tol"] or 1e-24                                  # J per cell
        to_plot = system
    a, b = state(g["initial"]), state(g["final"])
    if g["relax_endpoints"]:
        a, ita, _ = minimize(ad, a, tol)
        b, itb, _ = minimize(ad, b, tol)
        log(f"  endpoints relaxed ({ita}, {itb} iterations): E_A = {ad.energy(a):.6g}, E_B = {ad.energy(b):.6g} {ad.unit}")
    if geodesic_distance(a, b) < 1e-3 * np.sqrt(len(a)):
        raise RuntimeError("initial and final states relaxed into the same minimum: the initial texture is not "
                           "metastable for these parameters (change field, size or helicity)")
    path = slerp_path(a, b, int(g["images"]))
    if g.get("perturb", 0.02):                    # break the symmetry of a coherent initial path
        rng = np.random.default_rng(int(g.get("seed", 1)))
        for k in range(1, len(path) - 1):
            path[k] = _normalize(path[k] + float(g.get("perturb", 0.02)) * rng.normal(size=path[k].shape))
    neb = GNEB(ad, path, g["spring"], g["climbing"])
    neb.run(0.0, int(g["max_iter"]), float(g.get("rel_tol", 1e-4)), log=log)
    res = neb.result()
    res["lifetime_s"] = arrhenius(res["barrier_forward"], ad.kB, g["temperature"], g["attempt_time"])
    res["log10_lifetime_s"] = float(np.log10(g["attempt_time"]) + res["barrier_forward"] / (ad.kB * g["temperature"])
                                    / np.log(10)) if g["temperature"] > 0 else float("inf")
    res["temperature_K"] = g["temperature"]
    res["attempt_time_s"] = g["attempt_time"]
    kT = ad.kB * g["temperature"]
    log(f"  barrier A -> B: {res['barrier_forward']:.6g} {ad.unit} = {res['barrier_forward'] / kT:.2f} k_B T at "
        f"{g['temperature']} K;  B -> A: {res['barrier_backward']:.6g} {ad.unit}")
    log(f"  Arrhenius lifetime of A (tau0 = {g['attempt_time']:g} s): 10^{res['log10_lifetime_s']:.2f} s")
    with open(os.path.join(outdir, "gneb.json"), "w") as f:
        json.dump(res, f, indent=1)
    if ad.unit == "J":
        plot_path(res, os.path.join(outdir, "gneb_path.png"), "GNEB minimum-energy path", 1 / (ad.kB * 300),
                  r"$(E - E_0)/k_BT_{300\,\mathrm{K}}$")
    else:
        plot_path(res, os.path.join(outdir, "gneb_path.png"), "GNEB minimum-energy path")
    # snapshots of initial, saddle and final states
    ib = res["saddle_image"]
    if to_plot is None:
        from . import texture
        texture.plot_compare(system, [neb.x[0], neb.x[ib], neb.x[-1]], ["initial", f"saddle (image {ib})", "final"],
                             os.path.join(outdir, "gneb_states.png"))
    else:
        from .micromag.plot import plot_state
        for tag, k in (("initial", 0), ("saddle", ib), ("final", len(neb.x) - 1)):
            plot_state(system, ad.to_state(neb.x[k]), os.path.join(outdir, f"gneb_{tag}.png"), title=tag)
    np.savez_compressed(os.path.join(outdir, "gneb_images.npz"), images=np.array(neb.x), E=np.array(res["E"]))
    log(f"GNEB output written to {outdir}/")
    return res, neb
