"""magex command line

  magex run       input.toml   Green's-function exchange from Wannier90 (MPI: mpirun -n N magex run ...)
  magex magnon    input.toml   LSWT magnon bands from results.json
  magex damping   input.toml   Gilbert damping tensor from the Wannier Hamiltonian (torque correlation, MPI)
  magex llg       input.toml   atomistic stochastic LLG spin dynamics + real-space textures (B on/off)
  magex sqw       input.toml   dynamical structure factor S(q,w) from LLG, compared with LSWT
  magex gneb      input.toml   energy barriers (geodesic nudged elastic band), atomistic or micromagnetic
  magex mc        input.toml   Metropolis Monte Carlo: M(T), chi, C, Binder -> Curie / Neel temperature
  magex micromag  input.toml   finite-difference micromagnetics (thin films, nanostructures, SOT, loops)
  magex params    results.json atomistic -> micromagnetic parameters (Ms, A, D, K)
  magex tc        results.json fast ordering-temperature estimates: mean field, RPA, quantum-rescaled
  magex model     out.json     write a model spin Hamiltonian (lattice, J, D, K) in results.json format
  magex import-tb2j TB2J_results/ [--out results.json]   use TB2J exchange with every magex engine
  magex export    results.json|input.toml --to vampire|uppasd|spirit|mumax3 [--outdir DIR]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

try:
    import tomllib
except ImportError:  # pragma: no cover
    import tomli as tomllib


def load_cfg(path):
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    base = os.path.dirname(os.path.abspath(path))
    for k in ("win", "hr", "hr_up", "hr_dn", "centres", "poscar"):
        if k in cfg.get("input", {}):
            cfg["input"][k] = os.path.join(base, cfg["input"][k])
    cfg.setdefault("calc", {})
    cfg["calc"]["outdir"] = os.path.join(base, cfg["calc"].get("outdir", "magex_results"))
    return cfg


def magnon_from_results(cfg):
    from .magnon import LSWT, kpath_from_labels
    out = cfg["calc"]["outdir"]
    d = json.load(open(os.path.join(out, "results.json")))
    mc = cfg.get("magnon", {})
    source = mc.get("J", "mft")             # mft | bruno | downfold | bruno+downfold
    mp = d["mag_pos"]
    M = np.array(d["moments"])
    axis = np.array(mc.get("axis", d["refs"][-1]), float)
    axis /= np.linalg.norm(axis)
    s = np.sign(M @ axis)
    s[s == 0] = 1
    S = np.array([np.linalg.norm(M[i]) / 2 for i in mp])
    dirs = [s[i] * axis for i in mp]
    corr = None if source == "mft" else np.array(d["corrections"][source])
    from .heisenberg import all_images
    cell = np.array(d["cell"])
    frac = np.array(d["positions"]) @ np.linalg.inv(cell)
    images = {(p["i"], p["j"], tuple(p["nu"])): all_images(p["nu"], d["mesh"], frac[p["i"]], frac[p["j"]], cell)
              for p in d["pairs"]}
    cpl = []
    for p in d["pairs"]:
        if p["i"] not in mp or p["j"] not in mp:
            continue
        J = np.array(p["J"])
        if corr is not None:
            iso = np.trace(J) / 3
            J = J - iso * np.eye(3) + corr[mp.index(p["i"]), mp.index(p["j"])][tuple(p["nu"])] * np.eye(3)
        imgs = images[(p["i"], p["j"], tuple(p["nu"]))]
        for R in imgs:          # couplings at supercell-boundary ties are shared equally
            cpl.append((mp.index(p["i"]), mp.index(p["j"]), tuple(R), J / len(imgs)))
    A = np.array(d["A"])[mp] if mc.get("anisotropy", True) else None
    lsw = LSWT(S, dirs, cpl, A)
    special = {k: v for k, v in mc.get("special_points", {}).items()}
    labels = list(mc.get("path", "GXMG"))
    labels = [l if l != "G" else "G" for l in labels]
    path, ticks = kpath_from_labels(labels, special, int(mc.get("npts", 60)))
    W = lsw.band(path)
    np.savetxt(os.path.join(out, f"magnon_band_{source}.dat"), np.column_stack([np.arange(len(path)), path, W]),
               header="idx kx ky kz  energies(meV)")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 3.6))
        ax.plot(W, color="k", lw=1.2)
        ax.set_xticks(ticks)
        ax.set_xticklabels([r"$\Gamma$" if l == "G" else l for l in labels])
        for t in ticks:
            ax.axvline(t, color="0.8", lw=0.6)
        ax.set_xlim(0, len(path) - 1)
        ax.set_ylim(bottom=0)
        ax.set_ylabel("Energy (meV)")
        ax.set_title(f"LSWT magnons ({source})")
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"magnon_band_{source}.png"), dpi=180)
    except Exception as e:  # pragma: no cover
        print("plot skipped:", e)
    if getattr(lsw, "unstable", False):
        print("WARNING: complex magnon energies -> the chosen reference is not a local minimum")
    print(f"magnon bands written to {out}/magnon_band_{source}.dat")
    return W


def _raw_cfg(path):
    with open(path, "rb") as f:
        return tomllib.load(f), os.path.dirname(os.path.abspath(path))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="magex", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", choices=["run", "magnon", "damping", "llg", "sqw", "gneb", "mc", "micromag", "params", "tc", "model", "import-tb2j", "export", "program"])
    ap.add_argument("config", help="input TOML (results.json for 'params', output path for 'model')")
    ap.add_argument("--field", type=float, nargs=3, default=None, help="llg: override B (T)")
    ap.add_argument("--J", default="mft", help="params: mft | bruno | downfold")
    ap.add_argument("--rcut", type=float, default=None, help="params: coupling cutoff (A)")
    ap.add_argument("--thickness", type=float, default=None, help="params: film thickness (A)")
    ap.add_argument("--lattice", default="sc")
    ap.add_argument("--a", type=float, default=3.0)
    ap.add_argument("--Jnn", type=float, nargs="+", default=[1.0], help="model: J per shell (meV, per ordered pair)")
    ap.add_argument("--D", type=float, default=0.0)
    ap.add_argument("--dmi", default="interfacial")
    ap.add_argument("--K", type=float, default=0.0)
    ap.add_argument("--moment", type=float, default=2.0)
    ap.add_argument("--order", default="ferro")
    ap.add_argument("--out", default="results_tb2j.json", help="import-tb2j: output file")
    ap.add_argument("--to", default="vampire", help="export: vampire | uppasd | spirit | mumax3")
    ap.add_argument("--outdir", default=None, help="export: output directory (or .mx3 file)")
    a = ap.parse_args(argv)
    if a.task in ("run", "magnon"):
        cfg = load_cfg(a.config)
        if a.task == "run":
            from .driver import run
            run(cfg)
            if "magnon" in cfg:
                from . import parallel as par
                if par.is_root():
                    magnon_from_results(cfg)
        else:
            magnon_from_results(cfg)
    elif a.task == "damping":
        from .damping import run_damping
        run_damping(load_cfg(a.config))
    elif a.task == "llg":
        from .llg import run_llg_config
        cfg, base = _raw_cfg(a.config)
        run_llg_config(cfg, base, field_override=a.field)
    elif a.task == "sqw":
        from .spectra import run_sqw_config
        cfg, base = _raw_cfg(a.config)
        run_sqw_config(cfg, base)
    elif a.task == "program":
        from .programs import run_program_config
        cfg, base = _raw_cfg(a.config)
        run_program_config(cfg, base)
    elif a.task == "gneb":
        from .gneb import run_gneb_config
        cfg, base = _raw_cfg(a.config)
        run_gneb_config(cfg, base)
    elif a.task == "mc":
        from .montecarlo import run_mc_config
        cfg, base = _raw_cfg(a.config)
        run_mc_config(cfg, base)
    elif a.task == "micromag":
        from .micromag.driver import run_micromag_config
        cfg, base = _raw_cfg(a.config)
        run_micromag_config(cfg, base)
    elif a.task == "params":
        from .micromag.bridge import micromagnetic_parameters
        out = micromagnetic_parameters(a.config, J=a.J, rcut=a.rcut, thickness_A=a.thickness, log=print)
        print(json.dumps(out, indent=1))
    elif a.task == "tc":
        from .quantum import quantum_rescale, rpa_tc
        from .spinmodel import load_results, jq_stability
        uc = load_results(a.config, J=a.J, rcut=a.rcut)
        T_rpa, T_mf = rpa_tc(uc, log=print)
        lam0, lmax, q = jq_stability(uc)
        Tq, S = quantum_rescale(T_rpa, uc.moments)
        print(f"{a.config}: rcut = {uc.info['rcut']:.2f} A, {len(uc.info['shells'])} shells, S = {S:.2f}")
        print(f"  mean field  T_MF  = {T_mf:8.1f} K")
        print(f"  RPA         T_RPA = {T_rpa:8.1f} K   (Tyablikov, classical spins)")
        print(f"  RPA x (S+1)/S     = {Tq:8.1f} K   (quantum spins, localized moments)")
        if lmax > lam0 + 1e-9:
            print(f"  WARNING: J(q) maximum at q = {np.round(q, 3).tolist()} -> the reference is not the ground state")
        print("  classical Monte Carlo: magex mc <input.toml>")
    elif a.task == "import-tb2j":
        from .interop import read_tb2j
        d = read_tb2j(a.config)
        with open(a.out, "w") as f:
            json.dump(d, f)
        print(f"{len(d['pairs'])} TB2J couplings of {len(d['mag_pos'])} magnetic sites -> {a.out}")
    elif a.task == "export":
        from . import interop
        if a.to == "mumax3":
            cfg, _ = _raw_cfg(a.config)
            out = interop.export_mumax3(cfg, a.outdir or os.path.splitext(a.config)[0] + ".mx3")
        else:
            fn = {"vampire": interop.export_vampire, "uppasd": interop.export_uppasd,
                  "spirit": interop.export_spirit}[a.to]
            if a.config.endswith(".toml"):          # full magex set-up: model + structure/program/llg translated
                cfg, base = _raw_cfg(a.config)
                res = cfg.get("spinmodel", {}).get("results", "magex_results/results.json")
                res = res if os.path.isabs(res) else os.path.join(base, res)
                out = fn(res, a.outdir or f"{a.to}_input", J=a.J, rcut=a.rcut, cfg=cfg)
            else:
                out = fn(a.config, a.outdir or f"{a.to}_input", J=a.J, rcut=a.rcut)
        print(f"{a.to} input written to {out}")
    elif a.task == "model":
        from .models import write_model
        write_model(a.config, lattice=a.lattice, a=a.a, J=a.Jnn, D=a.D, dmi=a.dmi, K=a.K, moment=a.moment,
                    order=a.order)
        print(f"model written to {a.config}")


if __name__ == "__main__":
    sys.exit(main())
