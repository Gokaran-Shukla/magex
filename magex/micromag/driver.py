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
              "inner_nm", "vertices_nm", "side_nm", "path", "threshold", "invert", "subtract", "height_nm",
              "rotate_x_deg", "rotate_y_deg", "shapes"}
_MAT_KEYS = {"Ms", "A", "D", "dmi", "Ku1", "Ku2", "anisU", "Kc1", "Kc2", "Kc3", "anisC1", "anisC2", "alpha",
             "B1", "B2", "frozen"}


def build(cfg, base=".", log=print):
    mc = cfg["mesh"]
    mesh = Mesh(mc["n"], np.array(mc["cell_nm"], float) * 1e-9, mc.get("pbc", [False] * 3), mc.get("pbc_images", 10))
    sysm = Micromagnet(mesh, demag=cfg.get("demag", {}).get("enabled", True), log=log)
    sysm.demag_single = cfg.get("demag", {}).get("precision", "double") == "single"
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
        if "temperature_K" in r:                  # Callen-Callen scaling to a finite temperature
            from .llb import MCScaling, m_eq, scaled_parameters
            T = float(r["temperature_K"])
            if "mc_results" in r:
                pth = r["mc_results"] if os.path.isabs(r["mc_results"]) else os.path.join(base, r["mc_results"])
                me = MCScaling(pth)(T)
            else:
                me = m_eq(T, float(r["Tc"]))
            sc = scaled_parameters(mat, T, r.get("Tc", 1e9), me, r.get("A_exponent", 1.8), r.get("K_exponent", 3.0))
            sc.pop("m_e")
            mat = sc
            log(f"  region {k}: T = {T:g} K, m_e = {me:.4f} -> Ms = {mat['Ms']:.4g} A/m, A = {mat.get('A', 0):.4g} J/m, "
                f"Ku1 = {mat.get('Ku1', 0):.4g} J/m^3")
        if isinstance(mat.get("alpha"), dict):          # alpha = {from = "damping_n001.json", eta_meV = 50}
            from ..damping import resolve_alpha
            mat["alpha"] = resolve_alpha(mat["alpha"], base)
        shape = r.get("shape", "rectangle")
        gp = {k2: v for k2, v in r.items() if k2 in _GEOM_KEYS}
        if "path" in gp and not os.path.isabs(gp["path"]):
            gp["path"] = os.path.join(base, gp["path"])
        mask = geometry_mask(mesh, shape, **gp)
        sysm.add_region(mask, r.get("name", r.get("material", f"region{k}")), **mat)
        if r.get("edge_smooth"):                  # mumax3 EdgeSmooth: sub-cell volume fractions in the demag field
            from .mesh import volume_fraction
            f = volume_fraction(mesh, shape, int(r["edge_smooth"]), **gp)
            if sysm.geom is None:
                sysm.geom = np.ones(mesh.n)
            sysm.geom = np.where(f > 0, np.where(mask, f, sysm.geom), sysm.geom)
            if np.any((f > 0) & ~mask):           # partially filled cells outside the staircase mask
                extra = (f > 0) & ~mask & (sysm.region < 0)
                sysm.add_region(extra, sysm.region_names[-1] + "_edge", **mat)
                sysm.geom[extra] = f[extra]
    from .custom import custom_energy, custom_field
    for d in cfg.get("inter_dmi", []):            # [[inter_dmi]] regions = [0, 1], D = 1e-3 (J/m^2)
        sysm.inter_dmi[tuple(d["regions"])] = float(d["D"])
    for cf in cfg.get("custom_field", []):
        sysm.custom_fields.append(custom_field(cf, mesh))
    for ce in cfg.get("custom_energy", []):
        sysm.custom_energies.append(custom_energy(ce, mesh))
    if "magnetoelastic" in cfg:                   # [magnetoelastic] strain = [exx, eyy, ezz, exy, exz, eyz] or file
        me = cfg["magnetoelastic"]
        st_ = me["strain"]
        if isinstance(st_, str):
            st_ = np.load(st_ if os.path.isabs(st_) else os.path.join(base, st_))
        sysm.strain = np.asarray(st_, float)
        for key in ("B1", "B2"):
            if key in me:
                sysm.__dict__[key][sysm.magnetic] = float(me[key])
    for pair in cfg.get("interlayer", []):        # [[interlayer]] regions=[0,1] scale=0.5
        sysm.exchange_scale[tuple(pair["regions"])] = float(pair.get("scale", 1.0))
    from . import extras
    for k, r in enumerate(regions):
        mask = sysm.region == k
        for par_name, path in (r.get("maps") or {}).items():      # maps = {Ms = "ms.npy", Ku1 = "k.npy"}
            pth = path if os.path.isabs(path) else os.path.join(base, path)
            arr = extras.load_map(pth, mesh.n)
            getattr(sysm, par_name)[mask] = arr[mask] if r.get("maps_mode", "absolute") == "absolute" \
                else getattr(sysm, par_name)[mask] * arr[mask]
            sysm.invalidate()
        if "exchange_bias" in r:                                  # exchange_bias = {J = 1e-4, direction = [1,0,0]}
            eb = r["exchange_bias"]
            Beb = extras.exchange_bias_field(sysm, mask, float(eb["J"]), eb.get("direction", [1, 0, 0]),
                                             eb.get("thickness_nm", None) and eb["thickness_nm"] * 1e-9)
            sysm.B_ext_cell = Beb if sysm.B_ext_cell is None else sysm.B_ext_cell + Beb
            log(f"  region {k}: exchange bias {np.abs(Beb[mask]).max() * 1e3:.1f} mT along {eb.get('direction', [1, 0, 0])}")
    if "grains" in cfg:                                   # [grains] size_nm, Ku_spread, axis_spread_deg, gb_scale
        g = cfg["grains"]
        ng = extras.apply_grains(sysm, g["size_nm"], g.get("Ku_spread", 0.0), g.get("Ms_spread", 0.0),
                                 g.get("axis_spread_deg", 0.0), g.get("gb_scale", 1.0), g.get("seed", 1))
        log(f"  {ng} Voronoi grains of ~{g['size_nm']} nm")
    for rk in cfg.get("rkky", []):                         # [[rkky]] layers = [k1, k2], J1 = -1e-3, J2 = 0
        sysm.rkky.append((int(rk["layers"][0]), int(rk["layers"][1]), float(rk.get("J1", 0.0)), float(rk.get("J2", 0.0))))
    if "oersted" in cfg:                                   # [oersted] J = [Jx,Jy,Jz], shape/z_layers of the conductor
        oc = dict(cfg["oersted"])
        Jv = np.asarray(oc.pop("J"), float)
        shape = oc.pop("shape", "rectangle")
        cmask = geometry_mask(mesh, shape, **{k: v for k, v in oc.items() if k in _GEOM_KEYS})
        Jfield = np.zeros(mesh.n + (3,))
        Jfield[cmask] = Jv
        sysm.oersted = extras.oersted_field(mesh, Jfield)
        log(f"  Oersted field: max {np.abs(sysm.oersted[sysm.magnetic]).max() * 1e3:.2f} mT in the magnet")
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
        if isinstance(s.get("theta"), dict):              # theta from spin/orbital Hall conductivities
            from .fp_torques import resolve_theta
            s["theta"] = resolve_theta(s["theta"], base)
            log(f"  SOT efficiency from first-principles conductivities: theta_eff = {s['theta']:.4f}")
        sysm.sot = s
    if "spin_pumping" in cfg:                  # adds the spin-pumping damping and reports j_s / V_ISHE
        from .fp_torques import spin_pumping_damping
        sp = cfg["spin_pumping"]
        mag = sysm.magnetic
        t_fm = sp.get("thickness_nm", mesh.n[2] * mesh.d[2] * 1e9) * 1e-9
        da = spin_pumping_damping(sp["g_mix"], sysm.Ms[mag].mean(), t_fm)
        sysm.alpha[mag] += da
        sysm.spin_pumping = dict(sp, delta_alpha=da)
        log(f"  spin pumping: g_mix = {sp['g_mix']:.3g} m^-2 -> Delta alpha = {da:.4g}")
    if "stt" in cfg:                       # [stt] type = "zhang-li" (J = [Jx,Jy,Jz], P, beta) | "slonczewski"
        s = dict(cfg["stt"])
        if "thickness_nm" in s:
            s["thickness"] = s.pop("thickness_nm") * 1e-9
        sysm.stt = s
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
    ckpt = os.path.join(outdir, "checkpoint.npz")
    done_stage, resume = -1, None
    if oc.get("restart") and os.path.exists(ckpt):       # [output] restart = true
        z = np.load(ckpt, allow_pickle=True)
        m = z["m"]
        sysm.t = float(z["t"])
        done_stage = int(z["stage_done"])
        resume = dict(stage=int(z["stage_running"]), t0=float(z["t_stage_start"])) if int(z["stage_running"]) >= 0 else None
        if "summary" in z.files:
            summary = json.loads(str(z["summary"]))
        log(f"restart from {ckpt}: stages 0..{done_stage} done, t = {sysm.t * 1e9:.4f} ns"
            + (f", resuming stage {resume['stage']}" if resume else ""))

    def save_ckpt(stage_done, mm, running=-1, t_start=0.0):
        np.savez_compressed(ckpt, m=np.asarray(mm.get() if hasattr(mm, "get") else mm), t=sysm.t,
                            stage_done=stage_done, stage_running=running, t_stage_start=t_start,
                            summary=json.dumps(summary, default=float))
    for k, st in enumerate(stages):
        st = dict(st)
        typ = st["type"]
        tag = f"{k:02d}_{typ}"
        if k <= done_stage:
            if "B" in st:
                sysm.set_field(st["B"])
            continue
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
            track = []
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
            if st.get("track"):
                from .stt import texture_center
                cb0 = cb

                def cb(t, mm, cb0=cb0):
                    track.append([t, *texture_center(sysm, mm)])
                    if cb0:
                        cb0(t, mm)
            t_stage = sysm.t
            duration = st["time_ns"] * 1e-9
            if resume and resume["stage"] == k:
                t_stage = resume["t0"]
                duration = max(duration - (sysm.t - t_stage), 0.0)
                log(f"    resuming: {duration * 1e9:.4f} ns left")
            if st.get("checkpoint_ps"):
                every = st["checkpoint_ps"] * 1e-12
                nxt = [sysm.t + every]
                cb1 = cb

                def cb(t, mm, cb1=cb1, nxt=nxt, k=k, t_stage=t_stage):
                    if t >= nxt[0]:
                        save_ckpt(k - 1, mm, k, t_stage)
                        nxt[0] += every
                    if cb1:
                        cb1(t, mm)
            if st.get("moving_frame"):
                from .custom import MovingFrame
                mf_ = st["moving_frame"]
                mf_ = {} if mf_ is True else mf_
                frame = MovingFrame(sysm, mf_.get("component", "z"), mf_.get("axis", "x"))
                cb2 = cb

                def cb(t, mm, cb2=cb2, frame=frame):
                    frame(t, mm)
                    if cb2:
                        cb2(t, mm)
            else:
                frame = None
            if st.get("autosave_ps"):                     # mumax3 autosave(m, dt): OVF snapshots
                every_a = st["autosave_ps"] * 1e-12
                nxa = [sysm.t]
                cnt = [0]
                cb3 = cb

                def cb(t, mm, cb3=cb3, nxa=nxa, cnt=cnt, every_a=every_a, tag=tag):
                    if t >= nxa[0] - 1e-18:
                        write_ovf(os.path.join(outdir, f"m_{tag}_{cnt[0]:06d}.ovf"), mm, sysm.mesh, title=f"t = {t:.4e} s")
                        cnt[0] += 1
                        nxa[0] += every_a
                    if cb3:
                        cb3(t, mm)
            from .solver import make_solver
            solver = make_solver(sysm, st.get("solver", "rk45"), st.get("tol", 1e-5),
                                 st.get("dt_fs", 10.0) * 1e-15 if st.get("solver", "rk45") not in ("rk45", "rk23") else None)
            m, table = solver.run(m, duration, rec, cb, log=log)
            if frame is not None and len(frame.positions) > 2:
                fp = np.array(frame.positions)
                np.savetxt(os.path.join(outdir, f"wall_position_{tag}.txt"), fp, header="t_s x_wall_m (lab frame)")
                sel = fp[:, 0] > fp[0, 0] + 0.3 * (fp[-1, 0] - fp[0, 0])
                v = float(np.polyfit(fp[sel, 0], fp[sel, 1], 1)[0])
                info.update(wall_velocity_m_s=v, frame_shift_cells=frame.shift)
                log(f"    moving frame: shifted {frame.shift} cells, wall velocity {v:.2f} m/s")
            for fname in st.get("save_fields", []):       # field maps (mumax3 save(B_demag) ...)
                Fd = sysm.fields(m)
                if fname in Fd:
                    write_ovf(os.path.join(outdir, f"B_{fname}_{tag}.ovf"), np.asarray(Fd[fname]), sysm.mesh,
                              title=f"B_{fname} (T)")
            if getattr(sysm, "spin_pumping", None):
                from .fp_torques import ishe_voltage, pumped_spin_current
                sp = sysm.spin_pumping
                jdc, _ = pumped_spin_current(table, sp["g_mix"])
                info.update(spin_current_dc_J_m2=jdc.tolist())
                log("    pumped DC spin current j_s = (" + ", ".join(f"{x:.3e}" for x in jdc) + ") J/m^2")
                if all(k in sp for k in ("theta_SH", "rho", "lambda_s_nm", "nm_thickness_nm", "length_um")):
                    comp = int(np.argmax(np.abs(jdc)))
                    V = ishe_voltage(jdc[comp], sp["theta_SH"], sp["rho"], sp["lambda_s_nm"] * 1e-9,
                                     sp["nm_thickness_nm"] * 1e-9, sp["length_um"] * 1e-6)
                    info.update(V_ISHE=V)
                    log(f"    inverse spin Hall voltage {V * 1e6:.3f} uV")
            if track:
                tr = np.array(track)
                np.savetxt(os.path.join(outdir, f"track_{tag}.txt"), tr, header="t_s x_m y_m (texture centre)")
                sel = tr[:, 0] > tr[0, 0] + 0.2 * (tr[-1, 0] - tr[0, 0])
                v = [np.polyfit(tr[sel, 0], tr[sel, k], 1)[0] for k in (1, 2)]
                info.update(velocity_m_s=v, hall_angle_deg=float(np.degrees(np.arctan2(v[1], v[0]))))
                log(f"    texture velocity ({v[0]:.2f}, {v[1]:.2f}) m/s, angle {info['hall_angle_deg']:.2f} deg")
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
        elif typ == "thiele":
            from .stt import thiele, zhang_li_u
            stt = sysm.stt or {}
            beta = float(stt.get("beta", 0.0))
            u = zhang_li_u(sysm, stt.get("J", [0, 0, 0]), stt.get("P", 0.5), beta)[sysm.magnetic].mean(0)[:2]
            th = thiele(sysm, m, u, beta=beta)
            info.update(thiele=th)
            log(f"    Thiele: Q = {th['Q']:+.3f}, u = {np.round(u, 3).tolist()} m/s -> v = "
                f"{np.round(th['v'], 3).tolist()} m/s, skyrmion Hall angle {th['hall_angle_deg']:.2f} deg")
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
        save_ckpt(k, m)
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
