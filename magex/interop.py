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


def export_vampire(results, outdir, J="mft", rcut=None, system_size_nm=15.0, temperature_range=(0, 1500, 25)):
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
""")
    return outdir


def export_uppasd(results, outdir, J="mft", rcut=None):
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
    with open(os.path.join(outdir, "jfile"), "w") as fj, open(os.path.join(outdir, "dmfile"), "w") as fd:
        for (i, j, R, Jt) in uc.couplings:
            Jt = np.asarray(Jt)
            vec = (frac[j] + np.asarray(R) - frac[i])
            fj.write(f"{i + 1} {j + 1} {vec[0]: .8f} {vec[1]: .8f} {vec[2]: .8f} {np.trace(Jt) / 3 / MRY_MEV: .8e}\n")
            JA = 0.5 * (Jt - Jt.T)
            D = np.array([JA[1, 2], JA[2, 0], JA[0, 1]]) / MRY_MEV
            if np.max(np.abs(D)) > 0:
                fd.write(f"{i + 1} {j + 1} {vec[0]: .8f} {vec[1]: .8f} {vec[2]: .8f} {D[0]: .8e} {D[1]: .8e} {D[2]: .8e}\n")
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
dm ./dmfile
momfile ./momfile
maptype 1
initmag 3
ip_mode M
ip_mcanneal 1
10000 300 1.0e-16 0.01
mode M
temp 300
mcNstep 20000
""")
    return outdir


def export_spirit(results, outdir, J="mft", rcut=None):
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
    if dropped > 1e-6:
        print(f"  Spirit export: symmetric anisotropic exchange up to {dropped:.4f} meV has no Spirit term and was dropped")
    return outdir


def export_mumax3(cfg, path):
    """Translate a magex micromagnetic TOML (single region, rectangle/disk/ellipse, relax/minimize/run) to .mx3."""
    from .micromag.materials import material
    mc = cfg["mesh"]
    n = mc["n"]
    d = mc["cell_nm"]
    pbc = mc.get("pbc", [False] * 3)
    lines = ["// generated by magex", f"SetGridsize({n[0]}, {n[1]}, {n[2]})",
             f"SetCellsize({d[0]}e-9, {d[1]}e-9, {d[2]}e-9)"]
    if any(pbc):
        lines.append(f"SetPBC({10 if pbc[0] else 0}, {10 if pbc[1] else 0}, 0)")
    regs = cfg.get("region") or [dict(material="Py")]
    if len(regs) > 1:
        lines.append("// NOTE: only the first region is exported")
    r = regs[0]
    mat = material(r["material"]) if "material" in r else {}
    mat.update({k: v for k, v in r.items() if k in ("Ms", "A", "D", "dmi", "Ku1", "anisU", "Kc1", "alpha")})
    shape = r.get("shape", "rectangle")
    if shape == "disk":
        lines.append(f"SetGeom(Circle({r['diameter_nm']}e-9))")
    elif shape == "ellipse":
        a, b = r["diameters_nm"]
        lines.append(f"SetGeom(Ellipse({a}e-9, {b}e-9))")
    elif shape == "rectangle" and "size_nm" in r:
        a, b = r["size_nm"]
        lines.append(f"SetGeom(Rect({a}e-9, {b}e-9))")
    lines += [f"Msat = {mat.get('Ms', 8e5):g}", f"Aex = {mat.get('A', 1.3e-11):g}", f"alpha = {mat.get('alpha', 0.02):g}"]
    if mat.get("Ku1"):
        u = mat.get("anisU", [0, 0, 1])
        lines += [f"Ku1 = {mat['Ku1']:g}", f"anisU = vector({u[0]}, {u[1]}, {u[2]})"]
    if mat.get("Kc1"):
        lines += [f"Kc1 = {mat['Kc1']:g}", "anisC1 = vector(1, 0, 0)", "anisC2 = vector(0, 1, 0)"]
    if mat.get("D"):
        lines.append(("Dind" if mat.get("dmi", "interfacial") == "interfacial" else "Dbulk") + f" = {mat['D']:g}")
    if not cfg.get("demag", {}).get("enabled", True):
        lines.append("EnableDemag = false")
    B = cfg.get("field", {}).get("B", [0, 0, 0])
    lines.append(f"B_ext = vector({B[0]}, {B[1]}, {B[2]})")
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
    lines.append('TableAdd(E_total)\nTableAutosave(5e-12)')
    for st in cfg.get("stage") or [dict(type="relax")]:
        if "B" in st:
            b = st["B"]
            lines.append(f"B_ext = vector({b[0]}, {b[1]}, {b[2]})")
        if st["type"] == "relax":
            lines.append("Relax()")
        elif st["type"] == "minimize":
            lines.append("Minimize()")
        elif st["type"] == "run":
            lines.append(f"Run({st['time_ns']}e-9)")
        else:
            lines.append(f"// stage '{st['type']}' has no direct mumax3 equivalent")
        lines.append("Save(m)")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path
