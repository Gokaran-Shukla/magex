"""Writers: human-readable exchange.out, machine-readable results.json and pair CSV."""
from __future__ import annotations

import datetime
import json
import os

import numpy as np

from . import __version__


def _f(x, w=10, p=4):
    return f"{x:{w}.{p}f}"


def _corr_value(res, name, p):
    J = res.extra.get("corrections", {}).get(name)
    if J is None:
        return None
    mp = res.extra["mag_pos"]
    if p.i not in mp or p.j not in mp:
        return None
    return float(J[mp.index(p.i), mp.index(p.j)][p.nu])


def write_all(res, model, cfg, outdir):
    write_out(res, model, cfg, os.path.join(outdir, "exchange.out"))
    write_json(res, os.path.join(outdir, "results.json"))
    write_csv(res, os.path.join(outdir, "pairs.csv"))


def pairs_in_range(res, rcut, mag_only=True):
    mp = set(res.extra.get("mag_pos", range(len(res.sites))))
    out = []
    for p in res.pairs:
        if mag_only and (p.i not in mp or p.j not in mp):
            continue
        if rcut is not None and p.dist > rcut:
            continue
        out.append(p)
    return out


def write_out(res, model, cfg, fname):
    rcut = cfg.get("calc", {}).get("rcut")
    L = []
    bar = "=" * 96
    L += [bar, f"magex {__version__}   {datetime.datetime.now():%Y-%m-%d %H:%M}", bar]
    L.append("Convention: E = sum_i e_i.A_i.e_i - sum_{i!=j} e_i.J_ij.e_j  (unit spins, each ordered pair once,")
    L.append("            J>0 ferromagnetic, D_ij = (J^A_yz, J^A_zx, J^A_xy), energies in meV)  [= TB2J convention]")
    L.append(f"Source: {model.info.get('source')}   SOC: {res.extra.get('soc')}   "
             f"references: {res.refs}   full tensor: {res.full_tensor}")
    L.append(f"Exchange-field partition: {res.extra.get('partition')} "
             "('weighted' = rotationally invariant; 'onsite' = LKAG/TB2J definition)")
    L.append(f"mu = {res.extra['mu']:.5f} eV   T = {res.extra['temperature']} K   k-mesh = {res.mesh}")
    if "symmetry" in res.extra:
        s = res.extra["symmetry"]
        L.append(f"Symmetrisation: {s['spacegroup']} ({s['nops']} ops); max change |dJ| = {s['max_change_J']:.2e} meV, "
                 f"|dA| = {s['max_change_A']:.2e} meV  (~ Wannier noise)")
    L += [bar, "Cell (Angstrom):"]
    for v in model.cell:
        L.append("  " + " ".join(_f(x) for x in v))
    L += [bar, "Spin sites (charge and moment from the Wannier projection; moment in mu_B):",
          f"{'site':>5} {'atom':>6} {'x':>9} {'y':>9} {'z':>9} {'charge':>9} {'Mx':>9} {'My':>9} {'Mz':>9}  sum-rule(meV)"]
    for k, a in enumerate(res.sites):
        x = res.positions[k]
        m = res.moments[k]
        sr = res.extra.get("sumrule", np.zeros(len(res.sites)))[k]
        L.append(f"{k:5d} {res.symbols[k] + str(a + 1):>6} " + " ".join(_f(v, 9) for v in x) +
                 f" {res.charges[k]:9.4f} " + " ".join(_f(v, 9) for v in m) + f"  {sr:10.3e}")
    L.append("(sum rule: sum_jR Kb_ij(R) + T_i ; zero without SOC when all spin-polarised sites are included)")
    L += [bar, "Exchange (magnetic sites).  J_MFT: force theorem (full tensor fit, iso part);",
          "J_RMFT: Bruno-renormalised (moment-direction) MFT; J_DF: ligand-downfolded MFT. meV; R in lattice units.", ""]
    hdr = f"{'i':>4} {'j':>4} {'R':>12} {'dist':>7} {'J_MFT':>9} {'J_RMFT':>9} {'J_DF':>9}   {'Dx':>8} {'Dy':>8} {'Dz':>8}"
    L.append(hdr)
    for p in pairs_in_range(res, rcut):
        vals = [_corr_value(res, n, p) for n in ("bruno", "downfold")]
        vs = " ".join(f"{v:9.4f}" if v is not None else f"{'-':>9}" for v in vals)
        D = p.D
        L.append(f"{res.symbols[p.i] + str(res.sites[p.i] + 1):>4} {res.symbols[p.j] + str(res.sites[p.j] + 1):>4} "
                 f"{str(p.R):>12} {p.dist:7.3f} {p.iso:9.4f} {vs}   {D[0]:8.4f} {D[1]:8.4f} {D[2]:8.4f}"
                 + ("  *alias" if p.degeneracy > 1 else ""))
    if res.full_tensor:
        L += ["", "Symmetric anisotropic exchange J_ani (traceless, meV) and reference-dependence residual:"]
        for p in pairs_in_range(res, rcut)[:200]:
            a = p.ani
            L.append(f"{res.sites[p.i] + 1:>3}-{res.sites[p.j] + 1:<3} {str(p.R):>12} resid={p.resid:8.2e}  "
                     f"[[{a[0,0]:8.4f} {a[0,1]:8.4f} {a[0,2]:8.4f}] [{a[1,1]:8.4f} {a[1,2]:8.4f}] [{a[2,2]:8.4f}]] (xx xy xz | yy yz | zz)")
        L += ["", "Single-ion anisotropy A_i (meV; E_i = e.A.e, traceless):"]
        for k in res.extra.get("mag_pos", range(len(res.sites))):
            A = res.A[k]
            w, v = np.linalg.eigh(A)
            L.append(f"  {res.symbols[k]}{res.sites[k] + 1}: diag-xx,yy,zz = {A[0,0]:.4f} {A[1,1]:.4f} {A[2,2]:.4f}; "
                     f"off xy,xz,yz = {A[0,1]:.4f} {A[0,2]:.4f} {A[1,2]:.4f}; easy axis {np.round(v[:, 0], 3)} "
                     f"(fit resid {res.extra['sia_resid'][k]:.1e})")
    if "lloyd" in res.extra:
        L += [bar, "Finite-angle (Lloyd) pair interaction energies, i rotated by -t/2 and j by +t/2 (frozen potential):",
              "  E_int(t) = 4 s J' c(1-c) - 2 B cos t (cos t - 1),  c = cos(t/2), s = s_i s_j;  J_small = J' + 2 s B",
              "%18s %9s %9s %9s %9s %9s %8s" % ("pair", "J_small", "J_bilin", "B_biq", "J'+2sB", "Heis.err", "fit res")]
        lk = {(p.i, p.j, p.nu): p for p in res.pairs}
        for d in res.extra["lloyd"]:
            i, j, nu = d["pair"]
            Rl = lk[(i, j, tuple(nu))].R if (i, j, tuple(nu)) in lk else nu
            L.append(f"{res.symbols[i] + str(res.sites[i] + 1) + '-' + res.symbols[j] + str(res.sites[j] + 1) + str(Rl):>18} "
                     f"{d['J_small_angle']:9.4f} {d['J_bilinear']:9.4f} {d['B']:9.4f} {d['J_eff_from_fit']:9.4f} "
                     f"{100 * d['heisenberg_error']:8.1f}% {100 * d['fit_resid']:7.1f}%")
        L.append("Heis.err = max_t |E_Lloyd - E_bilinear(J_small)| / max|E| ; large => the bilinear model fails at large angles")
        L.append("E_int(t) in meV at t = " + " ".join(f"{np.degrees(t):.0f}" for t in res.extra["lloyd"][0]["angles"]) + " deg:")
        for d in res.extra["lloyd"]:
            L.append("  Lloyd    " + " ".join(f"{x:8.3f}" for x in d["E"]))
            L.append("  bilinear " + " ".join(f"{x:8.3f}" for x in d["E_bilinear"]))
    if "mae" in res.extra:
        m = res.extra["mae"]
        L += [bar, f"Magnetocrystalline anisotropy (band grand potential, k-mesh {m['mesh']}), meV/cell relative to minimum:"]
        for d, e in zip(m["directions"], m["E"]):
            L.append(f"  n = {np.round(d, 3)}  E = {e:10.5f}")
        w, v = np.linalg.eigh(m["A_band"])
        L.append(f"  quadratic fit: easy axis {np.round(v[:, 0], 3)}, K_eff = {w[1] - w[0]:.5f} meV (fit rms {m['fit_rms']:.1e})")
        wh, vh = np.linalg.eigh(m["A_heisenberg"])
        L.append(f"  time-reversal check: max |E(n) - E(-n)|/2 = {m.get('tr_odd_max', 0):.2e} meV "
                 "(non-zero => spin/SOC split of the spinor Wannier gauge is imperfect)")
        L.append(f"  Heisenberg model (sum A_i + two-site): easy axis {np.round(vh[:, 0], 3)}, K_eff = {wh[1] - wh[0]:.5f} meV")
    if res.extra.get("orbital"):
        L += [bar, "Orbital-resolved isotropic exchange (meV), rows: orbitals of i, cols: orbitals of j"]
        for (i, j, nu), M in res.extra["orbital"].items():
            L.append(f"  {res.symbols[i]}{res.sites[i] + 1} - {res.symbols[j]}{res.sites[j] + 1}  nu={nu}  sum = {M.sum():.4f}")
            for row in M:
                L.append("    " + " ".join(f"{x:8.3f}" for x in row))
    t = res.extra.get("timing", {})
    L += [bar, f"timing: eigen {t.get('eig', 0):.1f} s, energy integration {t.get('z', 0):.1f} s, total {t.get('total', 0):.1f} s", bar]
    with open(fname, "w") as f:
        f.write("\n".join(L) + "\n")


def _jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def write_json(res, fname):
    d = dict(version=__version__, sites=res.sites, symbols=res.symbols, cell=res.cell, positions=res.positions,
             mesh=res.mesh, moments=res.moments, charges=res.charges, refs=res.refs, full_tensor=res.full_tensor,
             A=res.A, mu=res.extra.get("mu"), mag_pos=res.extra.get("mag_pos"), lig_pos=res.extra.get("lig_pos"),
             pairs=[dict(i=p.i, j=p.j, R=p.R, nu=p.nu, dist=p.dist, vec=p.vec, J=p.J, resid=p.resid,
                         degeneracy=p.degeneracy) for p in res.pairs],
             corrections=res.extra.get("corrections"), lloyd=res.extra.get("lloyd"),
             mae=res.extra.get("mae"), symmetry=res.extra.get("symmetry"))
    with open(fname, "w") as f:
        json.dump(_jsonable(d), f)


def write_csv(res, fname):
    mp = res.extra.get("mag_pos", [])
    with open(fname, "w") as f:
        f.write("i,j,atom_i,atom_j,R1,R2,R3,dist,J_iso,J_RMFT,J_DF,Dx,Dy,Dz,Jxx,Jxy,Jxz,Jyx,Jyy,Jyz,Jzx,Jzy,Jzz\n")
        for p in res.pairs:
            if p.i not in mp or p.j not in mp:
                continue
            vals = [_corr_value(res, n, p) for n in ("bruno", "downfold")]
            vals = ["" if v is None else f"{v:.6f}" for v in vals]
            f.write(",".join(map(str, [p.i, p.j, res.sites[p.i] + 1, res.sites[p.j] + 1, *p.R, f"{p.dist:.5f}",
                                       f"{p.iso:.6f}", *vals, *[f"{x:.6f}" for x in p.D],
                                       *[f"{x:.6f}" for x in p.J.ravel()]])) + "\n")
