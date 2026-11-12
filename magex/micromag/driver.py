"""TOML-driven micromagnetic simulations:  magex micromag input.toml

[mesh]      n = [nx, ny, nz]   cell_nm = [dx, dy, dz]   pbc = [false, false, false]   pbc_images = 10
[[region]]  material = "Co/Pt" (library) and/or Ms, A, D, dmi, Ku1, Ku2, anisU, Kc1, anisC1, anisC2, alpha
            shape = "disk" | "rectangle" | "ellipse" | "ring" | "polygon" | "triangle" | "png"
            + shape parameters (diameter_nm, size_nm, center_nm, z_layers, subtract = [{shape=...}], ...)
            from_atomistic = "results.json"  (Ms, A, D, Ku taken from the magex bridge)
[demag]     enabled = true
[field]     B = [Bx, By, Bz] (T)      # static; pulse = {B=[..], t0_ns, width_ns} or ac = {B=[..], f_GHz}
[sot]       J = 1e11 (A/m^2)  theta = 0.1  sigma = [0,1,0]  fl_ratio = 0.0  thickness_nm
[init]      type = "uniform" | "random" | "skyrmion" | "vortex" | "domain_wall" | "helix" | "file" + params
[[stage]]   type = "relax" | "minimize" | "run" | "hysteresis" | "thermal"
            run:        time_ns, record_ps, B (optional, T), alpha (optional), movie = true
            thermal:    time_ns, T_K, dt_fs
            hysteresis: Bmax_T, npts, direction, branch = "full" | "down"
[output]    dir = "micromag_out"   ovf = true
"""
from __future__ import annotations

import json
import os

import numpy as np

from .constants import MU0
from .io import write_ovf, write_table
from .materials import material
from .mesh import Mesh, geometry_mask
from .plot import make_movie, plot_hysteresis, plot_state, plot_table, topological_charge
from .solver import TABLE_HEADER, Heun, RK45, hysteresis, minimize, relax
from .system import Micromagnet

_GEOM_KEYS = {"center_nm", "z_layers", "rotate_deg", "size_nm", "diameter_nm", "diameters_nm", "outer_nm",
              "inner_nm", "vertices_nm", "side_nm", "path", "threshold", "invert", "subtract"}
_MAT_KEYS = {"Ms", "A", "D", "dmi", "Ku1", "Ku2", "anisU", "Kc1", "anisC1", "anisC2", "alpha"}


def build(cfg, base=".", log=print):
    mc = cfg["mesh"]
    mesh = Mesh(mc["n"], np.array(mc["cell_nm"], float) * 1e-9, mc.get("pbc", [False] * 3), mc.get("pbc_images", 10))
    sysm = Micromagnet(mesh, demag=cfg.get("demag", {}).get("enabled", True), log=log)
    regions = cfg.get("region") or [dict(material="Py")]
    for k, r in enumerate(regions):
        r = dict(r)
        mat = {}
        if "material" in r:
            mat = material(r["material"])
        if "from_atomistic" in r:
            from .bridge import micromagnetic_parameters
            p = r["from_atomistic"]
            p = p if os.path.isabs(p) else os.path.join(base, p)
            mp = micromagnetic_parameters(p, J=r.get("J", "mft"), rcut=r.get("rcut"),
                                          thickness_A=r.get("thickness_A"), log=log)
            D = mp["D_int"] if abs(mp["D_int"]) >= abs(mp["D_bulk"]) else mp["D_bulk"]
            mat.update(Ms=mp["Ms"], A=mp["A"], D=D, dmi="interfacial" if abs(mp["D_int"]) >= abs(mp["D_bulk"]) else "bulk",
                       Ku1=mp["Ku"], anisU=[0, 0, 1])
        mat.update({k2: v for k2, v in r.items() if k2 in _MAT_KEYS})
        if isinstance(mat.get("alpha"), dict):          # alpha = {from = "damping_n001.json", eta_meV = 50}
            from ..damping import resolve_alpha
            mat["alpha"] = resolve_alpha(mat["alpha"], base)
        shape = r.get("shape", "rectangle")
        gp = {k2: v for k2, v in r.items() if k2 in _GEOM_KEYS}
        if "path" in gp and not os.path.isabs(gp["path"]):
            gp["path"] = os.path.join(base, gp["path"])
        mask = geometry_mask(mesh, shape, **gp)
        sysm.add_region(mask, r.get("name", r.get("material", f"region{k}")), **mat)
    for pair in cfg.get("interlayer", []):        # [[interlayer]] regions=[0,1] scale=0.5
        sysm.exchange_scale[tuple(pair["regions"])] = float(pair.get("scale", 1.0))
    fc = cfg.get("field", {})
    sysm.set_field(fc.get("B", [0, 0, 0]))
    if "pulse" in fc:
        p = fc["pulse"]
        B, t0, w = np.array(p["B"], float), p.get("t0_ns", 0.0) * 1e-9, p["width_ns"] * 1e-9
        sysm.B_func = lambda t: B * (t0 <= t < t0 + w)
    elif "ac" in fc:
        p = fc["ac"]
        B, f = np.array(p["B"], float), p["f_GHz"] * 1e9
        sysm.B_func = lambda t: B * np.sin(2 * np.pi * f * t)
    elif "sinc" in fc:                         # broadband excitation for spin-wave spectra
        p = fc["sinc"]
        B, fc_, t0 = np.array(p["B"], float), p["fmax_GHz"] * 1e9, p.get("t0_ns", 0.05) * 1e-9
        sysm.B_func = lambda t: B * np.sinc(2 * fc_ * (t - t0))
    if "sot" in cfg:
        s = dict(cfg["sot"])
        if "thickness_nm" in s:
            s["thickness"] = s.pop("thickness_nm") * 1e-9
        sysm.sot = s
    return sysm


def run_micromag_config(cfg, base=".", log=print):
    if "afm" in cfg:                       # two-sublattice antiferro-/ferri-/altermagnet
        from .afm import run_afm_config
        return run_afm_config(cfg, base, log)
    sysm = build(cfg, base, log)
    log(sysm.summary())
    oc = cfg.get("output", {})
    outdir = oc.get("dir", "micromag_out")
    outdir = outdir if os.path.isabs(outdir) else os.path.join(base, outdir)
    os.makedirs(outdir, exist_ok=True)
    ic = dict(cfg.get("init", {"type": "uniform"}))
    kind = ic.pop("type", "uniform")
    if kind == "file" and not os.path.isabs(ic.get("path", "/")):
        ic["path"] = os.path.join(base, ic["path"])
    m = sysm.init(kind, **ic)
    save_ovf = oc.get("ovf", True)
    plot_state(sysm, m, os.path.join(outdir, "m_initial.png"), title="initial state")
    summary = dict(stages=[])
    stages = cfg.get("stage") or [dict(type="relax")]
    for k, st in enumerate(stages):
        st = dict(st)
        typ = st["type"]
        tag = f"{k:02d}_{typ}"
        log(f"stage {k}: {typ} {({a: b for a, b in st.items() if a != 'type'})}")
        if "B" in st:
            sysm.set_field(st["B"])
        a0 = sysm.alpha.copy()
        if "alpha" in st:
            sysm.alpha[sysm.magnetic] = st["alpha"]
        info = dict(stage=k, type=typ)
        if typ in ("relax", "minimize"):
            tol = st.get("torque_tol", 1e-5)
            m = relax(sysm, m, tol, log=log) if typ == "relax" else minimize(sysm, m, tol, log=log)[0]
        elif typ == "run":
            frames = []
            rec = st.get("record_ps", 5.0) * 1e-12
            mov_every = st.get("movie_ps", None)
            cb = None
            if st.get("movie"):
                nxt = [0.0]
                step = (mov_every or st["time_ns"] * 1e3 / 40) * 1e-12

                def cb(t, mm, nxt=nxt, step=step):
                    if t >= nxt[0]:
                        frames.append((t, mm.copy()))
                        nxt[0] += step
            solver = RK45(sysm, tol=st.get("tol", 1e-5))
            m, table = solver.run(m, st["time_ns"] * 1e-9, rec, cb, log=log)
            write_table(os.path.join(outdir, f"table_{tag}.txt"), table, TABLE_HEADER)
            plot_table(table, os.path.join(outdir, f"table_{tag}.png"), title=st.get("title", f"stage {k}: run"))
            if frames:
                make_movie(sysm, frames, os.path.join(outdir, f"movie_{tag}.gif"))
            if st.get("spectrum"):
                _spectrum(table, os.path.join(outdir, f"spectrum_{tag}.png"))
        elif typ == "thermal":
            solver = Heun(sysm, dt=st.get("dt_fs", 10.0) * 1e-15, temperature=st["T_K"], seed=st.get("seed", 1))
            m, table = solver.run(m, st["time_ns"] * 1e-9, st.get("record_ps", 5.0) * 1e-12, log=log)
            write_table(os.path.join(outdir, f"table_{tag}.txt"), table, TABLE_HEADER)
            plot_table(table, os.path.join(outdir, f"table_{tag}.png"), title=f"stage {k}: T = {st['T_K']} K")
        elif typ == "dispersion":
            from ..spectra import analytic_film_dispersion, micromag_dispersion, plot_dispersion
            axis = {"x": 0, "y": 1}[st.get("axis", "x")]
            k, f, P, comp = micromag_dispersion(sysm, m, st.get("time_ns", 1.0), st.get("fmax_GHz", 100.0),
                                                st.get("B_amp_T", 1e-3), axis, st.get("source_nm"),
                                                st.get("width_cells", 1), st.get("dt_fs"), st.get("layer", 0),
                                                st.get("line"), log)
            np.savez_compressed(os.path.join(outdir, f"dispersion_{tag}.npz"), k_rad_nm=k, f_GHz=f, power=P)
            an = analytic_film_dispersion(sysm, k, axis) if st.get("analytic", True) else None
            plot_dispersion(k, f, P, os.path.join(outdir, f"dispersion_{tag}.png"), an,
                            st.get("title", "spin-wave dispersion"), st.get("kmax"), st.get("fmax_GHz"))
        elif typ == "hysteresis":
            snaps = []
            every = max(1, st.get("npts", 41) // 4)

            def hcb(i, b, mm):
                if i % every == 0:
                    snaps.append((b, mm.copy()))
            m, loop = hysteresis(sysm, m, st["Bmax_T"], st.get("npts", 41), st.get("direction", [0, 0, 1]),
                                 st.get("branch", "full"), st.get("torque_tol", 1e-4), hcb, log)
            np.savetxt(os.path.join(outdir, f"loop_{tag}.txt"), loop, header="B_T  m.u  mx  my  mz")
            plot_hysteresis(loop, os.path.join(outdir, f"loop_{tag}.png"), title=st.get("title"))
            for b, mm in snaps:
                plot_state(sysm, mm, os.path.join(outdir, f"m_{tag}_B{b * 1e3:+.0f}mT.png"),
                           title=f"B = {b * 1e3:.0f} mT")
            Bc = _coercive(loop)
            info.update(coercive_field_T=Bc)
            log(f"    coercive field ~ {Bc * 1e3 if Bc is not None else float('nan'):.1f} mT")
        else:
            raise ValueError(f"unknown stage type '{typ}'")
        sysm.alpha[:] = a0
        Q = plot_state(sysm, m, os.path.join(outdir, f"m_{tag}.png"), title=st.get("title", f"after stage {k} ({typ})"))
        if save_ovf:
            write_ovf(os.path.join(outdir, f"m_{tag}.ovf"), m, sysm.mesh, title=f"m after stage {k}")
        E = sysm.energies(m)
        info.update(m=sysm.average(m).tolist(), Q=Q, energies_J=E)
        summary["stages"].append(info)
        log(f"    <m> = {np.round(sysm.average(m), 4).tolist()}   Q = {Q:+.3f}   E = {E['total']:.4e} J")
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    log(f"micromagnetic output written to {outdir}/")
    return sysm, m, summary


def _coercive(loop):
    half = (len(loop) + 1) // 2
    b, mu = loop[:half, 0], loop[:half, 1]
    s = np.where(np.diff(np.sign(mu)) != 0)[0]
    if len(s) == 0:
        return None
    i = s[0]
    return float(abs(b[i] - mu[i] * (b[i + 1] - b[i]) / (mu[i + 1] - mu[i])))


def _spectrum(table, path):
    from ..plotstyle import OKABE_ITO, mpl, savefig
    plt = mpl()
    t = table[:, 0]
    dt = np.median(np.diff(t))
    fig, ax = plt.subplots(figsize=(4.5, 3.2), constrained_layout=True)
    for k, l in enumerate("xyz"):
        y = table[:, 1 + k] - table[:, 1 + k].mean()
        P = np.abs(np.fft.rfft(y * np.hanning(len(y)))) ** 2
        f = np.fft.rfftfreq(len(y), dt) / 1e9
        ax.semilogy(f, P + 1e-30, color=OKABE_ITO[k], label=f"$m_{l}$")
    ax.set_xlabel("$f$ (GHz)")
    ax.set_ylabel("power (arb.)")
    ax.legend()
    savefig(fig, path)
    plt.close(fig)
