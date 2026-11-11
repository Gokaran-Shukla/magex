"""Command line:  magex run input.toml   |   magex magnon input.toml"""
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


def main(argv=None):
    ap = argparse.ArgumentParser(prog="magex", description=__doc__)
    ap.add_argument("task", choices=["run", "magnon"])
    ap.add_argument("config")
    a = ap.parse_args(argv)
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


if __name__ == "__main__":
    sys.exit(main())
