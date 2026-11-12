"""Interoperability: TB2J import, and export to VAMPIRE, UppASD, Spirit and mumax3.

Conventions (all checked against the exporters of TB2J where available):

  magex / TB2J / UppASD (LKAG):  E = -sum_{i != j} [ J_ij e_i.e_j + D_ij.(e_i x e_j) + e_i.J^ani_ij.e_j ]
                                 (every ordered pair once; J in meV for magex, eV for TB2J, mRy for UppASD)
  VAMPIRE (.UCF):                tensorial interactions in joules, J_vampire = 2 J_magex (the UCF lists every
                                 ordered pair but VAMPIRE's Hamiltonian counts each bond once)
  Spirit:                        unique pairs, H = -sum_<ij> [J n.n + D.(n x n)] in meV: J_spirit = 2 J_magex,
                                 D_spirit = 2 D_magex (the symmetric anisotropic exchange has no Spirit term
                                 and is dropped with a warning)
"""
from __future__ import annotations

import json
import os
import re

import numpy as np

MEV = 1.602176634e-22          # J
MRY_MEV = 13.605693122994      # meV per mRy


def _skew(D):
    return np.array([[0, D[2], -D[1]], [-D[2], 0, D[0]], [D[1], -D[0], 0]])


# ----------------------------------------------------------------------------- TB2J import
def _tb2j_from_pickle(path):
    import pickle
    with open(path, "rb") as f:
        d = pickle.load(f)
    atoms = d["atoms"]
    cell = np.array(atoms.cell, float)
    pos = np.array(atoms.get_positions(), float)
    sym = list(atoms.get_chemical_symbols())
    ind = list(d["index_spin"])
    spinat = np.array(d.get("spinat"), float) if d.get("spinat") is not None else None
    pairs = []
    for key, J in d["exchange_Jdict"].items():
        R, i, j = key
        Jt = float(J) * 1e3 * np.eye(3)
        if d.get("dmi_ddict"):
            Jt = Jt + _skew(np.array(d["dmi_ddict"][key], float) * 1e3)
        if d.get("Jani_dict"):
            Jt = Jt + np.array(d["Jani_dict"][key], float) * 1e3
        pairs.append((int(i), int(j), tuple(int(x) for x in R), Jt))
    spin2atom = {int(ind[a]): a for a in range(len(sym)) if ind[a] >= 0}
    mags = [spin2atom[k] for k in sorted(spin2atom)]
    # TB2J indices i, j are spin indices (index_spin)
    pairs = [(spin2atom[i], spin2atom[j], R, Jt) for (i, j, R, Jt) in pairs]
    if spinat is None:
        spinat = np.zeros((len(sym), 3))
        spinat[:, 2] = np.array(d["magmoms"], float)
    return cell, pos, sym, mags, spinat, pairs, d.get("description", "")


def _tb2j_from_text(path):
    if str(path).endswith(".gz"):
        import gzip
        txt = gzip.open(path, "rt").read()
    else:
        txt = open(path).read()
    blk = txt.split("Cell (Angstrom):")[1].split("=====")[0].split("\n")
    cell = np.array([[float(x) for x in ln.split()] for ln in blk if len(ln.split()) == 3][:3])
    atoms = txt.split("Atom number")[1].split("=====")[0].split("\n")[1:]
    sym, pos, mom = [], [], []
    for ln in atoms:
        p = ln.split()
        if len(p) >= 6 and p[0] != "Total":
            sym.append(re.sub(r"\d+$", "", p[0]))
            pos.append([float(x) for x in p[1:4]])
            mom.append(float(p[-1]) if len(p) == 6 else [float(x) for x in p[-3:]])
    pos = np.array(pos)
    names = [ln.split()[0] for ln in atoms if len(ln.split()) >= 6 and ln.split()[0] != "Total"]
    spinat = np.zeros((len(sym), 3))
    for a, mm in enumerate(mom):
        spinat[a] = mm if isinstance(mm, list) else [0, 0, mm]
    pairs = []
    for m in re.finditer(r"^\s+(\S+)\s+(\S+)\s+\(\s*(-?\d+),\s*(-?\d+),\s*(-?\d+)\)\s+(-?[\d.]+).*?\n"
                         r"(?:.*?\n)*?Combined J tensor.*?\n\[\[(.*?)\]\n\s*\[(.*?)\]\n\s*\[(.*?)\]\]", txt, re.M):
        i, j = names.index(m.group(1)), names.index(m.group(2))
        R = tuple(int(m.group(k)) for k in (3, 4, 5))
        Jt = np.array([[float(x) for x in m.group(k).split()] for k in (7, 8, 9)])
        pairs.append((i, j, R, Jt))
    mags = sorted({p[0] for p in pairs} | {p[1] for p in pairs})
    return cell, pos, sym, mags, spinat, pairs, "parsed from exchange.out"


def read_tb2j(path):
    """TB2J_results directory (or TB2J.pickle / exchange.out) -> magex results.json dict."""
    if os.path.isdir(path):
        pk, ex = os.path.join(path, "TB2J.pickle"), os.path.join(path, "exchange.out")
    else:
        pk, ex = (path, None) if str(path).endswith(".pickle") else (None, path)
    src = None
    if pk and os.path.exists(pk):
        try:
            data = _tb2j_from_pickle(pk)
            src = pk
        except Exception:
            data = None
    if src is None:
        data = _tb2j_from_text(ex)
        src = ex
    cell, pos, sym, mags, spinat, pairs, desc = data
    maxR = np.max(np.abs([p[2] for p in pairs]), axis=0) if pairs else np.zeros(3, int)
    mesh = [int(2 * r + 3) for r in maxR]
    out_pairs = []
    for (i, j, R, Jt) in pairs:
        vec = pos[j] + np.array(R) @ cell - pos[i]
        out_pairs.append(dict(i=i, j=j, R=list(R), nu=[int(R[k]) % mesh[k] for k in range(3)],
                              dist=float(np.linalg.norm(vec)), vec=vec.tolist(), J=np.asarray(Jt).tolist(),
                              resid=0.0, degeneracy=1))
    ref = spinat[mags[0]] if len(mags) else np.array([0, 0, 1.0])
    ref = ref / max(np.linalg.norm(ref), 1e-12)
    return dict(version="tb2j-import", source=os.path.abspath(src), description=desc, sites=list(range(len(sym))),
                symbols=sym, cell=cell.tolist(), positions=pos.tolist(), mesh=mesh, moments=spinat.tolist(),
                charges=[0.0] * len(sym), refs=[ref.tolist()], full_tensor=True, A=[np.zeros((3, 3)).tolist()] * len(sym),
                mu=0.0, mag_pos=mags, lig_pos=[], pairs=out_pairs, corrections={}, lloyd=None, mae=None, symmetry=None)


# ----------------------------------------------------------------------------- exporters
def _uc(results, J="mft", rcut=None):
    from .spinmodel import load_results
    return load_results(results, J=J, rcut=rcut)


def export_vampire(results, outdir, J="mft", rcut=None, system_size_nm=15.0, temperature_range=(0, 1500, 25), cfg=None):
    uc = _uc(results, J, rcut)
    os.makedirs(outdir, exist_ok=True)
    lengths = np.linalg.norm(uc.cell, axis=1)
    frac = uc.positions @ np.linalg.inv(uc.cell)
    with open(os.path.join(outdir, "vampire.UCF"), "w") as f:
        f.write("# Unit cell size (Angstrom):\n" + " ".join(f"{x:.6f}" for x in lengths) + "\n")
        f.write("# Unit cell lattice vectors:\n")
        for v, L in zip(uc.cell, lengths):
            f.write(" ".join(f"{x:.6f}" for x in v / L) + "\n")
        f.write(f"# Atoms\n{uc.nsites} {uc.nsites}\n")
        for k, p in enumerate(frac):
            f.write(f"{k} {p[0]:.8f} {p[1]:.8f} {p[2]:.8f} {k}\n")
        f.write(f"# Interactions\n{len(uc.couplings)} tensorial\n")
        for n, (i, j, R, Jt) in enumerate(uc.couplings):
            vals = (2 * np.asarray(Jt) * MEV).ravel()
            f.write(f"{n:5d} {i:3d} {j:3d} {R[0]:3d} {R[1]:3d} {R[2]:3d} " + " ".join(f"{x: .5e}" for x in vals) + "\n")
    with open(os.path.join(outdir, "vampire.mat"), "w") as f:
        f.write(f"material:num-materials = {uc.nsites}\n")
        for k in range(uc.nsites):
            f.write(f"material[{k + 1}]:material-name = {uc.symbols[k]}{k + 1}\n")
            f.write(f"material[{k + 1}]:damping-constant = 1.0\n")
            f.write(f"material[{k + 1}]:atomic-spin-moment = {uc.moments[k]:.6f} !muB\n")
            d = uc.ref_dirs[k]
            f.write(f"material[{k + 1}]:initial-spin-direction = {d[0]:.6f},{d[1]:.6f},{d[2]:.6f}\n")
            A = np.asarray(uc.A[k])
            ev, V = np.linalg.eigh(A)
            Ku = (0.5 * (ev[1] + ev[2]) - ev[0]) * MEV           # easy axis = lowest eigenvalue
            if abs(Ku) > 0:
                ax = V[:, 0]
                f.write(f"material[{k + 1}]:uniaxial-anisotropy-constant = {Ku:.6e}\n")
                f.write(f"material[{k + 1}]:uniaxial-anisotropy-direction = {ax[0]:.6f},{ax[1]:.6f},{ax[2]:.6f}\n")
    T0, T1, dT = temperature_range
    with open(os.path.join(outdir, "input"), "w") as f:
        f.write(f"""# generated by magex from {os.path.basename(str(results))}
create:full
create:periodic-boundaries-x
create:periodic-boundaries-y
create:periodic-boundaries-z
material:file = vampire.mat
material:unit-cell-file = "vampire.UCF"
dimensions:unit-cell-size-x = {lengths[0]:.6f}
dimensions:unit-cell-size-y = {lengths[1]:.6f}
dimensions:unit-cell-size-z = {lengths[2]:.6f}
dimensions:system-size-x = {system_size_nm} !nm
dimensions:system-size-y = {system_size_nm} !nm
dimensions:system-size-z = {system_size_nm} !nm
sim:minimum-temperature = {T0}
sim:maximum-temperature = {T1}
sim:temperature-increment = {dT}
sim:equilibration-time-steps = 2500
sim:loop-time-steps = 3000
sim:program = curie-temperature
sim:integrator = monte-carlo
output:temperature
output:mean-magnetisation-length
""" + ("".join(x + "\n" for x in vampire_extras(cfg)) if cfg else ""))
    return outdir


def export_uppasd(results, outdir, J="mft", rcut=None, cfg=None):
    """posfile, momfile, jfile (isotropic, mRy) and dmfile (DMI, mRy) for UppASD."""
    uc = _uc(results, J, rcut)
    os.makedirs(outdir, exist_ok=True)
    frac = uc.positions @ np.linalg.inv(uc.cell)
    with open(os.path.join(outdir, "posfile"), "w") as f:
        for k, p in enumerate(frac):
            f.write(f"{k + 1} {k + 1} {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n")
    with open(os.path.join(outdir, "momfile"), "w") as f:
        for k in range(uc.nsites):
            d = uc.ref_dirs[k]
            f.write(f"{k + 1} 1 {uc.moments[k]:.6f} {d[0]:.6f} {d[1]:.6f} {d[2]:.6f}\n")
    has_dm = False
    with open(os.path.join(outdir, "jfile"), "w") as fj, open(os.path.join(outdir, "dmfile"), "w") as fd:
        for (i, j, R, Jt) in uc.couplings:
            Jt = np.asarray(Jt)
            vec = (frac[j] + np.asarray(R) - frac[i])
            fj.write(f"{i + 1} {j + 1} {vec[0]: .8f} {vec[1]: .8f} {vec[2]: .8f} {np.trace(Jt) / 3 / MRY_MEV: .8e}\n")
            JA = 0.5 * (Jt - Jt.T)
            D = np.array([JA[1, 2], JA[2, 0], JA[0, 1]]) / MRY_MEV
            if np.max(np.abs(D)) > 0:
                has_dm = True
                fd.write(f"{i + 1} {j + 1} {vec[0]: .8f} {vec[1]: .8f} {vec[2]: .8f} {D[0]: .8e} {D[1]: .8e} {D[2]: .8e}\n")
    # single-ion anisotropy (UppASD: E = -K1 sin^2(theta) - K2 sin^4(theta) about e_K, mRy; K1 < 0 = easy axis)
    has_k = False
    with open(os.path.join(outdir, "kfile"), "w") as fk:
        for k in range(uc.nsites):
            A = np.asarray(uc.A[k])
            ev, V = np.linalg.eigh(A)
            Ku = 0.5 * (ev[1] + ev[2]) - ev[0]                  # meV: E(hard plane) - E(easy axis)
            ax = V[:, 0]
            has_k = has_k or abs(Ku) > 0
            fk.write(f"{k + 1} 1 {-Ku / MRY_MEV: .8e} 0.0 {ax[0]: .8f} {ax[1]: .8f} {ax[2]: .8f} 0.0\n")
    with open(os.path.join(outdir, "inpsd.dat.template"), "w") as f:
        f.write(f"""simid magex
ncell 20 20 20
BC P P P
cell {' '.join(f'{x:.6f}' for x in uc.cell[0])}
     {' '.join(f'{x:.6f}' for x in uc.cell[1])}
     {' '.join(f'{x:.6f}' for x in uc.cell[2])}
Sym 0
posfiletype D
posfile ./posfile
exchange ./jfile
""" + ("dm ./dmfile\n" if has_dm else "") + """momfile ./momfile
""" + ("anisotropy ./kfile\n" if has_k else "") + """maptype 1
initmag 3
ip_mode M
ip_mcanneal 1
10000 300 1.0e-16 0.01
mode M
temp 300
mcNstep 20000
""" + ("".join(x + "\n" for x in uppasd_extras(cfg)) if cfg else ""))
    return outdir


def export_spirit(results, outdir, J="mft", rcut=None, cfg=None):
    uc = _uc(results, J, rcut)
    os.makedirs(outdir, exist_ok=True)
    seen = set()
    rows = []
    dropped = 0.0
    for (i, j, R, Jt) in uc.couplings:
        key = (i, j, tuple(R))
        rkey = (j, i, tuple(-np.asarray(R)))
        if rkey in seen:
            continue
        seen.add(key)
        Jt = np.asarray(Jt)
        JS = 0.5 * (Jt + Jt.T)
        JA = 0.5 * (Jt - Jt.T)
        iso = np.trace(Jt) / 3
        dropped = max(dropped, float(np.max(np.abs(JS - iso * np.eye(3)))))
        D = np.array([JA[1, 2], JA[2, 0], JA[0, 1]])
        nD = np.linalg.norm(D)
        Dhat = D / nD if nD > 0 else np.zeros(3)
        rows.append((i, j, *R, 2 * iso, 2 * nD, *Dhat))
    with open(os.path.join(outdir, "couplings.txt"), "w") as f:
        f.write("i j da db dc Jij Dij Dijx Dijy Dijz\n")
        for r in rows:
            f.write(f"{r[0]} {r[1]} {r[2]} {r[3]} {r[4]} {r[5]:.8f} {r[6]:.8f} {r[7]:.8f} {r[8]:.8f} {r[9]:.8f}\n")
    frac = uc.positions @ np.linalg.inv(uc.cell)
    A = np.asarray(uc.A[0])
    ev, V = np.linalg.eigh(A)
    K = 0.5 * (ev[1] + ev[2]) - ev[0]
    with open(os.path.join(outdir, "input.cfg"), "w") as f:
        f.write("### generated by magex\n### Geometry\nbravais_vectors\n")
        for v in uc.cell:
            f.write(" ".join(f"{x:.8f}" for x in v) + "\n")
        f.write("lattice_constant 1.0\nbasis\n" + f"{uc.nsites}\n")
        for p in frac:
            f.write(" ".join(f"{x:.8f}" for x in p) + "\n")
        f.write("n_basis_cells 30 30 1\nmu_s " + " ".join(f"{m:.4f}" for m in uc.moments) + "\n")
        f.write("### Hamiltonian\nhamiltonian heisenberg_pairs\nboundary_conditions 1 1 0\n")
        f.write(f"anisotropy_magnitude {K:.6f}\nanisotropy_normal {V[0, 0]:.6f} {V[1, 0]:.6f} {V[2, 0]:.6f}\n")
        f.write("interaction_pairs_file couplings.txt\n")
        for x in (spirit_extras(cfg) if cfg else []):
            f.write(x + "\n")
    if dropped > 1e-6:
        print(f"  Spirit export: symmetric anisotropic exchange up to {dropped:.4f} meV has no Spirit term and was dropped")
    return outdir


_MX3_SHAPE = {
    "disk": lambda r: f"Circle({r['diameter_nm']}e-9)",
    "ellipse": lambda r: f"Ellipse({r['diameters_nm'][0]}e-9, {r['diameters_nm'][1]}e-9)",
    "rectangle": lambda r: (f"Rect({r['size_nm'][0]}e-9, {r['size_nm'][1]}e-9)" if "size_nm" in r else "Universe()"),
    "sphere": lambda r: f"Ball({r['diameter_nm']}e-9)",
    "cylinder": lambda r: f"Cylinder({r['diameter_nm']}e-9, {r.get('height_nm', 1e6)}e-9)",
    "cuboid": lambda r: f"Cuboid({r['size_nm'][0]}e-9, {r['size_nm'][1]}e-9, {r['size_nm'][2]}e-9)",
    "ellipsoid": lambda r: f"Ellipsoid({r['diameters_nm'][0]}e-9, {r['diameters_nm'][1]}e-9, {r['diameters_nm'][2]}e-9)",
    "cone": lambda r: f"Cone({r['diameter_nm']}e-9, {r['height_nm']}e-9)",
    "ring": lambda r: f"Circle({r['outer_nm']}e-9).Sub(Circle({r['inner_nm']}e-9))",
}
_MX3_SOLVER = {"euler": 1, "heun": 2, "rk23": 3, "rk4": 4, "rk45": 5}


def _mx3_shape(r):
    shape = r.get("shape", "rectangle")
    if shape in ("union", "intersect", "difference", "xor"):
        op = {"union": "Add", "intersect": "Intersect", "difference": "Sub", "xor": "Xor"}[shape]
        parts = [_mx3_shape(dict(q)) for q in r["shapes"]]
        out = parts[0]
        for q in parts[1:]:
            out = f"{out}.{op}({q})"
        return out
    if shape not in _MX3_SHAPE:
        return None
    g = _MX3_SHAPE[shape](r)
    if r.get("rotate_deg"):
        g += f".RotZ({np.deg2rad(r['rotate_deg']):.8f})"
    c = list(r.get("center_nm", [0, 0])) + [0, 0]
    if any(c[:3]):
        g += f".Transl({c[0]}e-9, {c[1]}e-9, {c[2]}e-9)"
    for sub in r.get("subtract", []):
        q = _mx3_shape(dict(sub))
        if q:
            g += f".Sub({q})"
    if "z_layers" in r:
        g += f"  /* restrict to z layers {r['z_layers']} with Layers({r['z_layers'][0]}, {r['z_layers'][1] + 1}) */"
    return g


def export_mumax3(cfg, path):
    """Translate a magex micromagnetic TOML to a mumax3 .mx3 script (regions, CSG shapes, materials incl. Kc2/Kc3,
    magnetoelastic B1/B2, frozen spins, inter-region exchange/DMI, SOT/STT, solvers, moving frame, autosave)."""
    from .micromag.materials import material
    mc = cfg["mesh"]
    n = mc["n"]
    d = mc["cell_nm"]
    pbc = mc.get("pbc", [False] * 3)
    lines = ["// generated by magex", f"SetGridsize({n[0]}, {n[1]}, {n[2]})",
             f"SetCellsize({d[0]}e-9, {d[1]}e-9, {d[2]}e-9)"]
    if any(pbc):
        im = mc.get("pbc_images", 10)
        lines.append(f"SetPBC({im if pbc[0] else 0}, {im if pbc[1] else 0}, {im if pbc[2] else 0})")
    regs = cfg.get("region") or [dict(material="Py")]
    geoms = []
    smooth = max([int(r.get("edge_smooth", 0)) for r in regs] + [0])
    if smooth:
        lines.append(f"EdgeSmooth = {smooth}")
    for k, r in enumerate(regs):
        mat = material(r["material"]) if "material" in r else {}
        mat.update({q: v for q, v in r.items() if q in ("Ms", "A", "D", "dmi", "Ku1", "Ku2", "anisU", "Kc1", "Kc2", "Kc3",
                                                         "anisC1", "anisC2", "alpha", "B1", "B2", "frozen")})
        g = _mx3_shape(r)
        if g is None:
            lines.append(f"// region {k}: shape '{r.get('shape')}' has no mumax3 primitive (use a mask file)")
            g = "Universe()"
        lines.append(f"geom{k} := {g}")
        geoms.append(f"geom{k}")
        lines.append(f"DefRegion({k}, geom{k})")
        lines += [f"Msat.SetRegion({k}, {mat.get('Ms', 8e5):g})", f"Aex.SetRegion({k}, {mat.get('A', 1.3e-11):g})",
                  f"alpha.SetRegion({k}, {mat.get('alpha', 0.02):g})"]
        if mat.get("Ku1") or mat.get("Ku2"):
            u = mat.get("anisU", [0, 0, 1])
            lines += [f"Ku1.SetRegion({k}, {mat.get('Ku1', 0):g})", f"Ku2.SetRegion({k}, {mat.get('Ku2', 0):g})",
                      f"anisU.SetRegion({k}, vector({u[0]}, {u[1]}, {u[2]}))"]
        if mat.get("Kc1") or mat.get("Kc2") or mat.get("Kc3"):
            c1 = mat.get("anisC1", [1, 0, 0])
            c2 = mat.get("anisC2", [0, 1, 0])
            lines += [f"Kc1.SetRegion({k}, {mat.get('Kc1', 0):g})", f"Kc2.SetRegion({k}, {mat.get('Kc2', 0):g})",
                      f"Kc3.SetRegion({k}, {mat.get('Kc3', 0):g})",
                      f"anisC1.SetRegion({k}, vector({c1[0]}, {c1[1]}, {c1[2]}))",
                      f"anisC2.SetRegion({k}, vector({c2[0]}, {c2[1]}, {c2[2]}))"]
        if mat.get("D"):
            lines.append(("Dind" if mat.get("dmi", "interfacial") == "interfacial" else "Dbulk") + f".SetRegion({k}, {mat['D']:g})")
        if mat.get("B1") or mat.get("B2"):
            lines += [f"B1.SetRegion({k}, {mat.get('B1', 0):g})", f"B2.SetRegion({k}, {mat.get('B2', 0):g})"]
        if mat.get("frozen"):
            lines.append(f"FrozenSpins.SetRegion({k}, 1)")
    lines.append("SetGeom(" + ".Add(".join(geoms) + ")" * (len(geoms) - 1) + ")")
    for pair in cfg.get("interlayer", []):
        a, b = pair["regions"]
        lines.append(f"ext_ScaleExchange({a}, {b}, {pair.get('scale', 1.0)})")
    for dd in cfg.get("inter_dmi", []):
        a, b = dd["regions"]
        lines.append(f"ext_InterDind({a}, {b}, {dd['D']})")
    if "magnetoelastic" in cfg and not isinstance(cfg["magnetoelastic"]["strain"], str):
        e = cfg["magnetoelastic"]["strain"]
        lines += [f"exx = {e[0]}", f"eyy = {e[1]}", f"ezz = {e[2]}", f"exy = {e[3]}", f"exz = {e[4]}", f"eyz = {e[5]}"]
        for key in ("B1", "B2"):
            if key in cfg["magnetoelastic"]:
                lines.append(f"{key} = {cfg['magnetoelastic'][key]:g}")
    if not cfg.get("demag", {}).get("enabled", True):
        lines.append("EnableDemag = false")
    fc = cfg.get("field", {})
    B = fc.get("B", [0, 0, 0])
    if "ac" in fc:
        a_ = fc["ac"]
        lines.append(f"B_ext = vector({B[0]}, {B[1]}, {B[2]}).Add(vector({a_['B'][0]}, {a_['B'][1]}, {a_['B'][2]}).Mul(sin(2*pi*{a_['f_GHz']}e9*t)))")
    else:
        lines.append(f"B_ext = vector({B[0]}, {B[1]}, {B[2]})")
    if "sot" in cfg:
        s_ = cfg["sot"]
        sg = s_.get("sigma", [0, 1, 0])
        lines += [f"// SOT as Slonczewski torque with fixed polarisation (mumax3 convention)",
                  f"J = vector(0, 0, {s_.get('J', 0.0)})", f"FixedLayer = vector({sg[0]}, {sg[1]}, {sg[2]})",
                  f"Pol = {s_.get('theta', 0.1)}", "Lambda = 1", "EpsilonPrime = " + f"{s_.get('fl_ratio', 0.0)}"]
    if "stt" in cfg and cfg["stt"].get("type", "zhang-li") == "zhang-li":
        s_ = cfg["stt"]
        Jv = s_.get("J", [0, 0, 0])
        lines += [f"J = vector({Jv[0]}, {Jv[1]}, {Jv[2]})", f"Pol = {s_.get('P', 0.5)}", f"xi = {s_.get('beta', 0.0)}"]
    if cfg.get("custom_field") or cfg.get("custom_energy"):
        lines.append("// custom field / energy expressions: translate by hand with AddFieldTerm / AddEdensTerm")
    ic = cfg.get("init", {"type": "uniform"})
    if ic.get("type") == "uniform":
        u = ic.get("direction", [0, 0, 1])
        lines.append(f"m = Uniform({u[0]}, {u[1]}, {u[2]})")
    elif ic.get("type") == "skyrmion":
        lines.append(f"m = NeelSkyrmion(1, -1).scale(3, 3, 1)  // adjust size: radius {ic.get('radius_nm', 10)} nm")
    elif ic.get("type") == "vortex":
        lines.append("m = Vortex(1, 1)")
    elif ic.get("type") == "random":
        lines.append("m = RandomMag()")
    elif ic.get("type") == "domain_wall":
        lines.append("m = TwoDomain(0, 0, 1, 1, 0, 0, 0, 0, -1)")
    lines.append("TableAdd(E_total)\nTableAutosave(5e-12)")
    for st in cfg.get("stage") or [dict(type="relax")]:
        if "B" in st:
            b = st["B"]
            lines.append(f"B_ext = vector({b[0]}, {b[1]}, {b[2]})")
        if "alpha" in st:
            lines.append(f"alpha = {st['alpha']}")
        if st["type"] == "relax":
            lines.append("Relax()")
        elif st["type"] == "minimize":
            lines.append("Minimize()")
        elif st["type"] == "run":
            sv = st.get("solver", "rk45")
            lines.append(f"SetSolver({_MX3_SOLVER.get(sv, 5)})")
            if sv not in ("rk45", "rk23"):
                lines.append(f"FixDt = {st.get('dt_fs', 10.0)}e-15")
            if st.get("moving_frame"):
                mf = st["moving_frame"] if isinstance(st["moving_frame"], dict) else {}
                lines.append(f"ext_centerWall({ {'x': 0, 'y': 1, 'z': 2}[mf.get('component', 'z')] })")
            if st.get("autosave_ps"):
                lines.append(f"AutoSave(m, {st['autosave_ps']}e-12)")
            lines.append(f"Run({st['time_ns']}e-9)")
        elif st["type"] == "thermal":
            lines += ["SetSolver(2)", f"FixDt = {st.get('dt_fs', 10.0)}e-15", f"Temp = {st['T_K']}",
                      f"Run({st['time_ns']}e-9)", "Temp = 0"]
        elif st["type"] == "hysteresis":
            u = np.asarray(st.get("direction", [0, 0, 1]), float)
            Bm, npts = st["Bmax_T"], st.get("npts", 41)
            lines += [f"for B := {Bm}; B >= {-Bm}; B -= {2 * Bm / (npts - 1)} {{",
                      f"    B_ext = vector({u[0]}*B, {u[1]}*B, {u[2]}*B)", "    Minimize()", "    TableSave()", "}"]
            if st.get("branch", "full") == "full":
                lines += [f"for B := {-Bm}; B <= {Bm}; B += {2 * Bm / (npts - 1)} {{",
                          f"    B_ext = vector({u[0]}*B, {u[1]}*B, {u[2]}*B)", "    Minimize()", "    TableSave()", "}"]
        else:
            lines.append(f"// stage '{st['type']}' has no direct mumax3 equivalent")
        for fname in st.get("save_fields", []):
            lines.append({"demag": "Save(B_demag)", "exchange": "Save(B_exch)", "anisotropy": "Save(B_anis)",
                          "zeeman": "Save(B_ext)"}.get(fname, f"// save field '{fname}' by hand"))
        lines.append("Save(m)")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


# ----------------------------------------------------------------------------- structure / program translation
_VAMP_SHAPE = {"sphere": "sphere", "cylinder": "cylinder", "cube": "cube", "ellipsoid": "ellipsoid",
               "truncated_octahedron": "truncated-octahedron"}
_VAMP_PROG = {"hysteresis": "hysteresis-loop", "field_cooling": "field-cool", "cmc": "cmc-anisotropy",
              "anneal": "curie-temperature", "minimize": "time-series", "fmr": "time-series", "lsf": "curie-temperature"}


def vampire_extras(cfg):
    """Extra VAMPIRE input lines for a magex [structure] / [program] / [llg] set-up (VAMPIRE 6 keyword names)."""
    out = []
    st = cfg.get("structure", {}) or {}
    sh = st.get("shape")
    if sh in _VAMP_SHAPE:
        out.append(f"create:{_VAMP_SHAPE[sh]}")
        size = st.get("size", 10.0)
        size = 2 * float(size if np.isscalar(size) else max(size)) if sh != "cube" else float(size)
        out.append(f"dimensions:particle-size = {size / 10:.4f} !nm")
    if st.get("dipolar"):
        out += ["dipole:solver = atomistic", "dipole:field-update-rate = 1"]     # exact pair sum, as magex
    if "vacancies" in st:
        v = st["vacancies"]
        out.append(f"# vacancies: use material[n]:density = {1 - (v if np.isscalar(v) else v['fraction']):.4f}")
    if "roughness" in st:
        ro = st["roughness"] if isinstance(st["roughness"], dict) else {}
        out += ["create:interfacial-roughness", f"create:interfacial-roughness-random-seed = {ro.get('seed', 1)}",
                f"create:interfacial-roughness-mean-height = {float(ro.get('amplitude', 1.0)) * 3:.3f} !A",
                f"create:interfacial-roughness-seed-radius = {float(ro.get('correlation', 10.0)):.3f} !A"]
    pr = cfg.get("program", {}) or {}
    if pr:
        typ = pr.get("type", "hysteresis")
        out.append(f"sim:program = {_VAMP_PROG.get(typ, 'time-series')}")
        if typ == "hysteresis":
            u = np.asarray(pr.get("direction", [0, 0, 1]), float)
            out += [f"sim:maximum-applied-field-strength = {pr.get('Bmax', 2.0)} !T",
                    f"sim:applied-field-strength-increment = {2 * pr.get('Bmax', 2.0) / (pr.get('npts', 41) - 1):.4f} !T",
                    f"sim:applied-field-unit-vector = {u[0]},{u[1]},{u[2]}"]
        if typ == "cmc":
            out += ["sim:integrator = constrained-monte-carlo",
                    "# torque curve: sim:constraint-angle-theta-minimum/maximum/increment"]
        if typ == "field_cooling":
            out += [f"sim:cooling-time = 1 !ns", f"sim:maximum-temperature = {pr.get('T_start', 1000)}",
                    f"sim:minimum-temperature = {pr.get('T_end', 0)}"]
    llg = cfg.get("llg", {}) or {}
    if llg:
        out += ["sim:integrator = llg-heun", f"sim:time-step = {llg.get('dt', 1e-3) * 1e-12:.3e}",
                f"sim:total-time-steps = {llg.get('steps', 10000)}", f"sim:temperature = {llg.get('temperature', 0.0)}"]
        if llg.get("sot"):
            out.append("# spin-orbit torque: material[n]:spin-orbit-torque-* keywords (VAMPIRE 7)")
    return out


def uppasd_extras(cfg):
    """Extra inpsd.dat lines for a magex [structure] / [llg] / [program] set-up."""
    out = []
    st = cfg.get("structure", {}) or {}
    if st.get("dipolar"):
        out.append("do_dip 1")
    llg = cfg.get("llg", {}) or {}
    if llg:
        B = llg.get("field", [0, 0, 0])
        out += ["mode S", f"damping {llg.get('alpha', 0.1)}", f"timestep {llg.get('dt', 1e-3) * 1e-12:.3e}",
                f"Nstep {llg.get('steps', 10000)}", f"temp {llg.get('temperature', 0.0)}",
                f"hfield {B[0]} {B[1]} {B[2]}", "SDEalgh 1"]
        if llg.get("stt"):
            u = llg["stt"].get("u", [0, 0, 0])
            out += ["stt A", f"jvec {u[0]} {u[1]} {u[2]}", f"adibeta {llg['stt'].get('beta', 0.0)}"]
    pr = cfg.get("program", {}) or {}
    if pr.get("type") == "minimize":
        out.append("mode G  # GNEB/minimisation module (UppASD 6)")
    return out


def spirit_extras(cfg):
    out = []
    st = cfg.get("structure", {}) or {}
    if st.get("dipolar"):
        out += ["ddi_method fft", "ddi_n_periodic_images 4 4 0"]
    llg = cfg.get("llg", {}) or {}
    B = (llg or cfg.get("program", {}) or {}).get("field", [0, 0, 0])
    nB = float(np.linalg.norm(B))
    if nB > 0:
        out += [f"external_field_magnitude {nB}", "external_field_normal " + " ".join(f"{x / nB:.6f}" for x in B)]
    if llg:
        out += [f"llg_damping {llg.get('alpha', 0.1)}", f"llg_dt {llg.get('dt', 1e-3)}",
                f"llg_n_iterations {llg.get('steps', 10000)}", f"llg_temperature {llg.get('temperature', 0.0)}"]
        if llg.get("stt"):
            out.append("llg_stt_use_gradient 1  # set llg_stt_magnitude / llg_stt_polarisation_normal")
    return out
