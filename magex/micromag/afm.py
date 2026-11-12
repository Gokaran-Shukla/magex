"""Two-sublattice micromagnetics: antiferromagnets, ferrimagnets and altermagnets.

Each sublattice s = A, B is a unit-vector field m_s with its own M_s, A, K, D, alpha (a
``Micromagnet`` each, demag switched off) plus

    w_AF   = J_AF  m_A . m_B                     homogeneous inter-sublattice exchange (J/m^3, J_AF > 0: AFM)
    w_AB   = A_AB  sum_i d_i m_A . d_i m_B        inhomogeneous inter-sublattice exchange (J/m, optional)
    w_d    = -(mu0/2) M . H_d[M],  M = Ms_A m_A + Ms_B m_B   (demag of the total magnetisation)

and sublattice-anisotropic stiffness A^s_i = A_s c^s_i (``exchange_axis_scale`` per sublattice),
the continuum model of a d-wave altermagnet: (1+d, 1-d, 1) on A and (1-d, 1+d, 1) on B split the two
magnon chiralities along x and y without spin-orbit coupling.

The state is stacked, m[s] = m_s with shape (2, nx, ny, nz, 3), and the class exposes the interface
of ``Micromagnet`` (effective_field, normalize, alpha, magnetic, energies, average, sot_field, ...)
so the RK45 / Heun / minimise / relax / hysteresis solvers run unchanged.

Fields:  B_A += -(J_AF/Ms_A) m_B + (A_AB/Ms_A) lap m_B,  B_B likewise.
AFMR (easy axis, B = 0):  f = (gamma/2pi) sqrt(B_K (2 B_E + B_K)),  B_E = J_AF/Ms, B_K = 2K/Ms.
"""
from __future__ import annotations

import numpy as np

from .constants import MU0
from .system import Micromagnet


class TwoSublatticeMagnet:
    def __init__(self, mesh, demag=True, workers=-1, log=None):
        self.mesh = mesh
        self.sub = [Micromagnet(mesh, demag=False, workers=workers, log=log),
                    Micromagnet(mesh, demag=False, workers=workers, log=log)]
        self.use_demag = bool(demag)
        self._demag = None
        self.workers = workers
        self.log = log
        self.J_AF = np.zeros(mesh.n)
        self.A_AB = np.zeros(mesh.n)
        self.sot = None
        self._t = 0.0
        self._sync_alpha()

    # ------------------------------------------------------------------ plumbing
    def _sync_alpha(self):
        self._alpha = np.stack([self.sub[0].alpha, self.sub[1].alpha])
        self.sub[0].alpha, self.sub[1].alpha = self._alpha[0], self._alpha[1]

    @property
    def alpha(self):
        return self._alpha

    @property
    def t(self):
        return self._t

    @t.setter
    def t(self, v):
        self._t = v
        for s in self.sub:
            s.t = v

    @property
    def magnetic(self):
        return np.stack([self.sub[0].magnetic, self.sub[1].magnetic])

    @property
    def Ms(self):
        return np.stack([self.sub[0].Ms, self.sub[1].Ms])

    @property
    def B_ext(self):
        return self.sub[0].B_ext

    @property
    def B_func(self):
        return self.sub[0].B_func

    @B_func.setter
    def B_func(self, f):
        for s in self.sub:
            s.B_func = f

    @property
    def B_local(self):
        return self.sub[0].B_local

    def set_field(self, B):
        for s in self.sub:
            s.set_field(B)

    def add_region(self, mask, name="afm", sub_A=None, sub_B=None, J_AF=0.0, A_AB=0.0, **common):
        """Material on both sublattices: ``common`` keys apply to both, ``sub_A``/``sub_B`` dicts override
        (e.g. a ferrimagnet: sub_A = {Ms = 6e5}, sub_B = {Ms = 4e5})."""
        mask = np.asarray(mask, bool)
        for s, own in zip(self.sub, (sub_A or {}, sub_B or {})):
            p = dict(common)
            p.update(own)
            s.add_region(mask, name, **p)
        self.J_AF[mask] = J_AF
        self.A_AB[mask] = A_AB
        self._sync_alpha()

    def set_altermagnet(self, delta, axes=(0, 1)):
        """d-wave altermagnet: A^A = A (1+d, 1-d), A^B = A (1-d, 1+d) along ``axes``."""
        a, b = axes
        for s, sg in zip(self.sub, (1, -1)):
            sc = np.ones(3)
            sc[a] = 1 + sg * delta
            sc[b] = 1 - sg * delta
            s.exchange_axis_scale = sc
            s.invalidate()

    @property
    def demag(self):
        if self._demag is None and self.use_demag:
            from .demag import Demag
            self._demag = Demag(self.mesh, self.workers, log=self.log)
        return self._demag

    def normalize(self, m):
        return np.stack([self.sub[0].normalize(m[0]), self.sub[1].normalize(m[1])])

    # ------------------------------------------------------------------ fields / energies
    def _laplacian(self, s, mo):
        """Plain (Neumann) Laplacian of the other sublattice's field on sublattice s's mesh."""
        lap = np.zeros_like(mo)
        for axis in range(3):
            if self.mesh.n[axis] == 1 and not self.mesh.pbc[axis]:
                continue
            for step in (1, -1):
                exists = self.sub[s]._link(axis, step)[0]
                nb = np.where(exists[..., None], np.roll(mo, -step, axis=axis), mo)
                lap += (nb - mo) / self.mesh.d[axis] ** 2
        return lap

    def fields(self, m, t=None):
        F = [self.sub[0].fields(m[0], t), self.sub[1].fields(m[1], t)]
        for s, o in ((0, 1), (1, 0)):
            Ms = self.sub[s].Ms
            inv = np.divide(1.0, Ms, out=np.zeros_like(Ms), where=Ms > 0)
            F[s]["afm_exchange"] = -(self.J_AF * inv)[..., None] * m[o]
            if np.any(self.A_AB):
                F[s]["afm_exchange"] = F[s]["afm_exchange"] + (self.A_AB * inv)[..., None] * self._laplacian(s, m[o])
            F[s]["afm_exchange"] = np.where(self.sub[s].magnetic[..., None], F[s]["afm_exchange"], 0.0)
        if self.use_demag:
            M = self.sub[0].Ms[..., None] * m[0] + self.sub[1].Ms[..., None] * m[1]
            Bd = MU0 * self.demag.field(M)
            for s in range(2):
                F[s]["demag"] = np.where(self.sub[s].magnetic[..., None], Bd, 0.0)
        return F

    def effective_field(self, m, t=None):
        F = self.fields(m, t)
        return np.stack([sum(F[0].values()), sum(F[1].values())])

    def sot_field(self, m):
        if not self.sot:
            return 0.0
        out = []
        for s in range(2):
            self.sub[s].sot = self.sot
            out.append(self.sub[s].sot_field(m[s]) * np.ones_like(m[s]))
        return np.stack(out)

    def energies(self, m, F=None):
        F = F or self.fields(m)
        V = self.mesh.V
        E = {}
        for s in range(2):
            Fs = {k: v for k, v in F[s].items() if k not in ("demag", "afm_exchange")}
            Es = self.sub[s].energies(m[s], Fs)
            for k, v in Es.items():
                if k != "total":
                    E[k] = E.get(k, 0.0) + v
        # inter-sublattice terms counted once: w = -(1/2) sum_s Ms_s m_s . B_AF,s
        E["afm_exchange"] = float(sum(-0.5 * np.sum(self.sub[s].Ms * np.einsum("...a,...a->...", m[s], F[s]["afm_exchange"]))
                                      for s in range(2)) * V)
        if self.use_demag:
            E["demag"] = float(sum(-0.5 * np.sum(self.sub[s].Ms * np.einsum("...a,...a->...", m[s], F[s]["demag"]))
                                   for s in range(2)) * V)
        E["total"] = sum(E.values())
        return E

    def average(self, m):
        """Net magnetisation direction-weighted average (Ms-weighted, both sublattices)."""
        w = self.Ms
        mag = w > 0
        tot = (m * w[..., None])[mag].sum(0)
        return tot / max(w[mag].sum(), 1e-300)

    def neel(self, m):
        """Average Neel vector n = (m_A - m_B)/2."""
        mag = self.sub[0].magnetic
        return 0.5 * (m[0][mag] - m[1][mag]).mean(0)

    def torque(self, m, B=None):
        B = self.effective_field(m) if B is None else B
        return np.cross(m, B)

    def init(self, kind="neel", direction=(0, 0, 1), **p):
        """Initial state: "neel" (m_A = +d, m_B = -d) or any Micromagnet init for m_A with m_B = -m_A."""
        if kind == "neel":
            mA = self.sub[0].init("uniform", direction=direction)
        else:
            mA = self.sub[0].init(kind, direction=direction, **p)
        return self.normalize(np.stack([mA, -mA]))

    def summary(self):
        mag = self.sub[0].magnetic
        lines = [repr(self.mesh), f"  two-sublattice magnet: {int(mag.sum())} cells per sublattice"]
        for k, s in enumerate(self.sub):
            v = s.Ms[s.magnetic]
            lines.append(f"  sublattice {'AB'[k]}: Ms {v.mean():.4g} A/m, A {s.A[s.magnetic].mean():.3g} J/m, "
                         f"Ku {s.Ku1[s.magnetic].mean():.3g} J/m^3, alpha {s.alpha[s.magnetic].mean():.3g}, "
                         f"stiffness scale {np.round(s.exchange_axis_scale, 3).tolist()}")
        if mag.any():
            J = self.J_AF[mag].mean()
            Ms = self.sub[0].Ms[mag].mean()
            K = self.sub[0].Ku1[mag].mean()
            lines.append(f"  J_AF = {J:.3g} J/m^3 (B_E = {J / Ms:.3g} T), A_AB = {self.A_AB[mag].mean():.3g} J/m")
            if K > 0 and J > 0:
                from .constants import GAMMA_LL
                BK, BE = 2 * K / Ms, J / Ms
                lines.append(f"  AFMR estimate (gamma/2pi) sqrt(B_K (2 B_E + B_K)) = "
                             f"{GAMMA_LL / (2 * np.pi) * np.sqrt(BK * (2 * BE + BK)) / 1e9:.1f} GHz")
        return "\n".join(lines)


# ----------------------------------------------------------------------------- spin waves
def afm_analytic(system, k_nm, axis=0):
    """Two chirality branches f_+(k), f_-(k) (GHz) of a uniaxial (easy z) two-sublattice magnet.

    f_pm = (gamma/2pi) [ sqrt(((H_A + H_B)/2)^2 - B_E^2) +- (H_A - H_B)/2 ],
    H_s = B_K + B_E + (2 A_s c^s_axis / Ms) (2 - 2 cos k d)/d^2  (+ B_z for the field term, sign by chirality)
    """
    from .constants import GAMMA_LL
    s0, s1 = system.sub
    mag = s0.magnetic
    Ms = s0.Ms[mag].mean()
    K = s0.Ku1[mag].mean()
    BE = system.J_AF[mag].mean() / Ms
    d = system.mesh.d[axis]
    k = np.asarray(k_nm) * 1e9
    lam = (2 - 2 * np.cos(k * d)) / d**2
    HA = 2 * K / Ms + BE + 2 * s0.A[mag].mean() * s0.exchange_axis_scale[axis] / Ms * lam
    HB = 2 * K / Ms + BE + 2 * s1.A[mag].mean() * s1.exchange_axis_scale[axis] / Ms * lam
    root = np.sqrt(np.clip(((HA + HB) / 2) ** 2 - BE**2, 0, None))
    g = GAMMA_LL / (2 * np.pi) / 1e9
    Bz = system.B_ext[2]
    return g * (root + (HA - HB) / 2 + Bz), g * (root - (HA - HB) / 2 - Bz)


def afm_dispersion(system, m, time_ns=0.2, fmax_GHz=1000.0, B_amp=0.01, axis=0, dt_fs=None, log=None):
    """Chirality-resolved spin-wave dispersion of a two-sublattice magnet.

    Records c(x, t) = m_Ax + i m_Ay on a line; after a 2D FFT, f > 0 and f < 0 are the two circular
    polarisations (the two magnon chiralities).  Returns k (rad/nm), f (GHz, signed), power (nk, nf).
    """
    from .constants import GAMMA_LL
    from .solver import Heun
    mesh = system.mesh
    n = mesh.n
    if dt_fs is None:
        mag = system.sub[0].magnetic
        Ms = system.sub[0].Ms[mag].min()
        dmin = min(dd for dd, nn in zip(mesh.d, mesh.n) if nn > 1)
        Bmax = (2 * system.J_AF[mag].max() / Ms + 8 * max(s.A[mag].max() * s.exchange_axis_scale.max() for s in system.sub)
                / (Ms * dmin**2) + 2 * abs(system.sub[0].Ku1[mag]).max() / Ms)
        dt_fs = 0.05 / (GAMMA_LL * Bmax) * 1e15
        if log:
            log(f"    AFM dispersion: fixed time step {dt_fs:.3f} fs (B_max ~ {Bmax:.0f} T)")
    X, Y, Z = mesh.coords()
    mask = np.abs((X, Y)[axis]) < mesh.d[axis] / 2 + 1e-15
    fc = fmax_GHz * 1e9
    t0 = 5.0 / fc
    exc = np.array([1.0, 0, 0])
    fn = (lambda t: B_amp * exc * np.sinc(2 * fc * (t - t0)))
    for s in system.sub:
        s.B_local.append((mask, fn))
    rec_dt = 1.0 / (2.5 * fc)
    other = 1 - axis
    sl = [0, slice(None), slice(None), slice(None)]
    sl[1 + other] = n[other] // 2
    sl[3] = 0
    frames = []
    t_start = system.t
    Heun(system, dt=dt_fs * 1e-15).run(m, time_ns * 1e-9, rec_dt, lambda t, mm: frames.append(mm[tuple(sl)].copy()), log=log)
    for s in system.sub:
        s.B_local.pop()
    system.t = t_start
    data = np.array(frames)
    c = data[..., 0] + 1j * data[..., 1]
    c = (c - c.mean(0)) * np.hanning(len(c))[:, None]
    F = np.fft.fftshift(np.fft.fft2(c))
    f = np.fft.fftshift(np.fft.fftfreq(len(c), rec_dt)) / 1e9
    k = np.fft.fftshift(np.fft.fftfreq(n[axis], mesh.d[axis])) * 2 * np.pi * 1e-9
    # c ~ exp(+i 2 pi f t): f > 0 is counter-clockwise precession of m_A about +z (chirality +)
    return k, f, np.abs(F).T ** 2


def plot_afm_dispersion(k, f, P, path, branches=None, title=None, kmax=None):
    from matplotlib.colors import LogNorm
    from ..plotstyle import OKABE_ITO, mpl, savefig
    plt = mpl()
    order = np.argsort(f)
    fig, ax = plt.subplots(figsize=(5.0, 4.0), constrained_layout=True)
    Z = P[:, order].T / P.max()
    mp = ax.pcolormesh(k, f[order], Z, shading="nearest", cmap="magma", norm=LogNorm(vmin=1e-5, vmax=1), rasterized=True)
    fig.colorbar(mp, ax=ax, pad=0.01).set_label("power (norm.)")
    if branches is not None:
        fp, fm = branches
        ax.plot(k, fp, "--", color=OKABE_ITO[5], lw=1.1, label="analytic, chirality +")
        ax.plot(k, -fm, ":", color=OKABE_ITO[4], lw=1.4, label="analytic, chirality −")
        ax.legend(loc="upper center", labelcolor="white", fontsize=7)
    km = kmax or np.abs(k).max()
    ax.set_xlim(-km, km)
    sel = np.abs(k) <= km
    if branches is not None:
        top = 1.12 * max(np.max(branches[0][sel]), np.max(branches[1][sel]))
    else:
        top = np.abs(f).max()
    ax.set_ylim(-top, top)
    ax.axhline(0, color="w", lw=0.4)
    ax.set_xlabel(r"$k$ (rad nm$^{-1}$)")
    ax.set_ylabel(r"$f$ (GHz)   [sign = chirality]")
    ax.minorticks_off()
    if title:
        ax.set_title(title)
    savefig(fig, path)
    plt.close(fig)


# ----------------------------------------------------------------------------- driver
def build_afm(cfg, base=".", log=print):
    import os
    from .driver import _GEOM_KEYS, _MAT_KEYS
    from .materials import material
    from .mesh import Mesh, geometry_mask
    mc = cfg["mesh"]
    mesh = Mesh(mc["n"], np.array(mc["cell_nm"], float) * 1e-9, mc.get("pbc", [False] * 3), mc.get("pbc_images", 10))
    a = cfg["afm"]
    sysm = TwoSublatticeMagnet(mesh, demag=cfg.get("demag", {}).get("enabled", True), log=log)
    for r in cfg.get("region") or [dict()]:
        r = dict(r)
        mat = material(r["material"]) if "material" in r else {}
        mat.update({k: v for k, v in r.items() if k in _MAT_KEYS})
        gp = {k: v for k, v in r.items() if k in _GEOM_KEYS}
        if "path" in gp and not os.path.isabs(gp["path"]):
            gp["path"] = os.path.join(base, gp["path"])
        mask = geometry_mask(mesh, r.get("shape", "rectangle"), **gp)
        sysm.add_region(mask, r.get("name", "afm"), sub_A=r.get("sub_A"), sub_B=r.get("sub_B"),
                        J_AF=float(r.get("J_AF", a.get("J_AF", 0.0))), A_AB=float(r.get("A_AB", a.get("A_AB", 0.0))),
                        **mat)
    if a.get("altermagnet_delta"):
        sysm.set_altermagnet(float(a["altermagnet_delta"]), tuple(a.get("altermagnet_axes", [0, 1])))
    sysm.set_field(cfg.get("field", {}).get("B", [0, 0, 0]))
    if "sot" in cfg:
        s = dict(cfg["sot"])
        if "thickness_nm" in s:
            s["thickness"] = s.pop("thickness_nm") * 1e-9
        sysm.sot = s
    return sysm


def run_afm_config(cfg, base=".", log=print):
    import json
    import os
    from .plot import plot_state
    from .solver import RK45, minimize, relax
    sysm = build_afm(cfg, base, log)
    log(sysm.summary())
    out = cfg.get("output", {}).get("dir", "afm_out")
    out = out if os.path.isabs(out) else os.path.join(base, out)
    os.makedirs(out, exist_ok=True)
    ic = dict(cfg.get("init", {"type": "neel"}))
    m = sysm.init(ic.pop("type", "neel"), **ic)
    summary = []
    for k, st in enumerate(cfg.get("stage") or [dict(type="relax")]):
        st = dict(st)
        typ = st["type"]
        tag = f"{k:02d}_{typ}"
        log(f"stage {k}: {typ}")
        if "B" in st:
            sysm.set_field(st["B"])
        if typ in ("relax", "minimize"):
            m = relax(sysm, m, st.get("torque_tol", 1e-4), log=log) if typ == "relax" else \
                minimize(sysm, m, st.get("torque_tol", 1e-4), log=log)[0]
        elif typ == "run":
            rows = []

            def cb(t, mm):
                rows.append([t, *sysm.neel(mm), *sysm.average(mm)])
            m, _ = RK45(sysm, tol=st.get("tol", 1e-5)).run(m, st["time_ns"] * 1e-9, st.get("record_ps", 1.0) * 1e-12, cb,
                                                          log=log)
            tab = np.array(rows)
            np.savetxt(os.path.join(out, f"neel_{tag}.txt"), tab, header="t_s nx ny nz mx my mz (net)")
            _plot_neel(tab, os.path.join(out, f"neel_{tag}.png"), st.get("title"))
        elif typ == "dispersion":
            axis = {"x": 0, "y": 1}[st.get("axis", "x")]
            kk, f, P = afm_dispersion(sysm, m, st.get("time_ns", 0.1), st.get("fmax_GHz", 2000.0),
                                      st.get("B_amp_T", 0.01), axis, st.get("dt_fs"), log)
            np.savez_compressed(os.path.join(out, f"dispersion_{tag}.npz"), k_rad_nm=kk, f_GHz=f, power=P)
            plot_afm_dispersion(kk, f, P, os.path.join(out, f"dispersion_{tag}.png"), afm_analytic(sysm, kk, axis),
                                st.get("title"), st.get("kmax"))
        else:
            raise ValueError(f"stage type '{typ}' is not available for two-sublattice systems")
        plot_state(sysm.sub[0], m[0], os.path.join(out, f"mA_{tag}.png"), title=f"sublattice A after stage {k}")
        n = sysm.neel(m)
        summary.append(dict(stage=k, type=typ, neel=n.tolist(), m_net=sysm.average(m).tolist()))
        log(f"    Neel vector {np.round(n, 4).tolist()}, net m {np.round(sysm.average(m), 5).tolist()}")
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    log(f"two-sublattice output written to {out}/")
    return sysm, m


def _plot_neel(tab, path, title=None):
    from ..plotstyle import OKABE_ITO, mpl, savefig
    plt = mpl()
    fig, axs = plt.subplots(1, 2, figsize=(8, 2.9), constrained_layout=True)
    t = tab[:, 0] * 1e12
    for c, l in enumerate("xyz"):
        axs[0].plot(t, tab[:, 1 + c], color=OKABE_ITO[c], label=f"$n_{l}$")
        axs[1].plot(t, tab[:, 4 + c], color=OKABE_ITO[c], label=f"$m_{l}$")
    axs[0].set_ylabel("Néel vector")
    axs[1].set_ylabel("net magnetisation")
    for ax in axs:
        ax.set_xlabel("$t$ (ps)")
        ax.legend(ncol=3)
    if title:
        fig.suptitle(title)
    savefig(fig, path)
    plt.close(fig)
