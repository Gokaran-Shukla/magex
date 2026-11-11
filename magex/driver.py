"""High-level workflow: load -> references -> tensors -> corrections -> magnons -> output."""
from __future__ import annotations

import itertools
import json
import os
import time

import numpy as np

from . import parallel as par
from .analysis import (default_directions, fit_quadratic_anisotropy, grand_potential_directions,
                       heisenberg_anisotropy, lloyd_fit)
from .contour import Contour, KB_EV
from .engine import compute_reference, kmesh_frac
from .engine_partition import compute_reference_partition
from .heisenberg import build_results, min_image
from .model import load_wannier90, rotation_to
from .renorm import renormalize

DEFAULTS = dict(kmesh=[9, 9, 9], contour="ozaki", temperature=300.0, npole=None, emin=None, nz=120,
                rcut=None, soc="auto", partition="weighted", refs="auto", symmetrize=True, bruno=True, downfold=True,
                orbital_shells=2, lloyd_shells=2, lloyd_angles=13, mae=False, mae_kmesh=None,
                nel=None, outdir="magex_results")


def native_direction(model, atoms):
    """Direction of the time-odd on-site field of the most strongly polarised magnetic atom."""
    best, v = -1.0, np.array([0, 0, 1.0])
    for a in atoms:
        p = model.onsite_exchange(a)
        t = np.array([np.trace(p[c]).real for c in range(3)])
        if np.linalg.norm(t) > best:
            best, v = np.linalg.norm(t), t
    return v / np.linalg.norm(v) if best > 0 else np.array([0, 0, 1.0])


def find_mu(model, mesh, nel, kT, guess):
    kpts = kmesh_frac(mesh)
    myk = par.split(len(kpts))
    HR = model.spinor_HR()
    E = np.concatenate([np.linalg.eigvalsh(model.Hk(kpts[myk[c:c + 64]], HR)).ravel()
                        for c in range(0, len(myk), 64)]) if len(myk) else np.zeros(0)
    lo, hi = guess - 20, guess + 20
    for _ in range(100):
        mu = 0.5 * (lo + hi)
        x = np.clip((E - mu) / max(kT, 1e-6), -200, 200)
        n = par.allreduce_sum(np.array([np.sum(1 / (np.exp(x) + 1))]))[0] / len(kpts)
        lo, hi = (mu, hi) if n < nel else (lo, mu)
    return mu


def shell_pairs(model, sites, mag_pos, mesh, nshell):
    """(si, sj, nu) for mag-mag pairs in the first nshell distance shells (unique per orbit-ish)."""
    if nshell <= 0:
        return []
    frac = model.frac_positions
    cand = []
    for i in mag_pos:
        for j in mag_pos:
            for nu in itertools.product(*[range(n) for n in mesh]):
                if i == j and nu == (0, 0, 0):
                    continue
                R, d, deg = min_image(nu, mesh, frac[sites[i]], frac[sites[j]], model.cell)
                cand.append((round(d, 3), i, j, nu, deg))
    shells = sorted(set(c[0] for c in cand))[:nshell]
    out, seen = [], set()
    for d, i, j, nu, deg in sorted(cand):
        if d in shells and d not in seen and deg == 1:
            out.append((i, j, nu))
            seen.add(d)
    return out


def run(cfg):
    t0 = time.time()
    c = dict(DEFAULTS)
    c.update(cfg.get("calc", {}))
    outdir = c["outdir"]
    if par.is_root():
        os.makedirs(outdir, exist_ok=True)
    timing = {}
    model = load_wannier90(cfg["input"])
    mesh = tuple(c["kmesh"])
    mag_el = cfg["magnetic"]["elements"]
    lig_el = cfg["magnetic"].get("ligands", [])
    mag_atoms = [a for a, s in enumerate(model.symbols) if s in mag_el]
    lig_atoms = [a for a, s in enumerate(model.symbols) if s in lig_el]
    lig_min = float(cfg["magnetic"].get("ligand_min_field", 0.005))   # eV
    dropped = [a for a in lig_atoms if np.linalg.norm(model.onsite_exchange(a)) < lig_min]
    if dropped:
        par.log(f"ligands without on-site exchange field (< {lig_min} eV) ignored: "
                + ", ".join(f"{model.symbols[a]}{a + 1}" for a in dropped))
    lig_atoms = [a for a in lig_atoms if a not in dropped]
    sites = mag_atoms + lig_atoms
    mag_pos = list(range(len(mag_atoms)))
    lig_pos = list(range(len(mag_atoms), len(sites)))
    par.log(f"magex: {len(model.symbols)} atoms, {model.norb} orbitals, {len(model.R)} R-vectors, "
            f"mesh {mesh}, {par.SIZE} MPI ranks")

    # ---------------- spin-orbit & reference directions
    soc = c["soc"]
    collinear_input = model.info["source"] == "wannier90-collinear"
    if soc == "auto":
        soc = "atomic" if (collinear_input and cfg.get("soc_xi") is not None) else ("input" if model.has_soc() else "none")
    if soc == "replace":
        # discard the time-even spin part of a spinor Hamiltonian (and the imaginary part of H0),
        # then add atomic SOC: removes spin/SOC mixing from an imperfect spinor Wannier gauge
        model.S[:] = 0
        model.H0 = model.H0.real.astype(complex)
        model.X = model.X.real.astype(complex)
        soc = "atomic"
    if soc == "atomic":
        xi = None
        if cfg.get("soc_xi"):
            from .orbitals import L_OF, XI_DEFAULT
            xi = dict(XI_DEFAULT)
            for k, v in cfg["soc_xi"].items():
                el, sh = k.split(":")
                xi[(el, L_OF[sh])] = float(v)
        model = model.add_atomic_soc(xi)
        par.log("atomic SOC added:", {f"{k[0]}:{'spdf'[k[1]]}": v for k, v in model.info["soc_xi"].items()})
    has_soc = model.has_soc()
    n0 = np.array([0, 0, 1.0]) if collinear_input else native_direction(model, mag_atoms)
    if c["refs"] == "auto":
        refs = [(1, 0, 0), (0, 1, 0), (0, 0, 1)] if has_soc else [(0, 0, 1)]
    elif c["refs"] == "native":
        refs = [tuple(n0)]
    else:
        refs = [tuple(r) for r in c["refs"]]
    base = model.rotated(rotation_to(n0).T)          # magnetisation along z

    # ---------------- contour / Fermi level
    e0 = np.linalg.eigvalsh(base.Hk(np.zeros((1, 3))))[0]
    bw = float(e0[-1] - e0[0]) + 10.0
    kT = KB_EV * float(c["temperature"])
    mu = float(cfg["input"].get("efermi", 0.0))
    if c["nel"] is not None:
        mu = find_mu(base, mesh, float(c["nel"]), kT, mu)
        par.log(f"chemical potential fixed by nel={c['nel']}: mu = {mu:.5f} eV")
    contour = Contour(c["contour"], mu, float(c["temperature"]), c["npole"],
                      emin=c["emin"] if c["emin"] is not None else float(e0[0]) - 2.0,
                      nz=int(c["nz"]), bandwidth=bw)
    par.log(f"contour: {c['contour']}, {len(contour)} points, T = {c['temperature']} K, mu = {mu:.4f} eV")

    lloyd = None
    ll_pairs = shell_pairs(base, sites, mag_pos, mesh, int(c["lloyd_shells"])) if c["contour"] == "ozaki" else []
    if ll_pairs:
        lloyd = dict(pairs=ll_pairs, angles=np.linspace(0, np.pi, int(c["lloyd_angles"])))
    orb_pairs = shell_pairs(base, sites, mag_pos, mesh, int(c["orbital_shells"]))

    refdata = []
    for d in refs:
        dd = np.asarray(d, float)
        m = base.with_direction(dd)
        ll = dict(lloyd, axis=np.cross(dd, [1, 0, 0] if abs(dd[0]) < 0.9 else [0, 1, 0])) if lloyd else None
        if c["partition"] == "onsite":
            rd = compute_reference(m, sites, mesh, contour, lloyd=ll, orbital_pairs=orb_pairs, timing=timing)
        else:
            rd = compute_reference_partition(m, sites, mesh, contour, mode=c["partition"], lloyd=ll,
                                             timing=timing)
        refdata.append(rd)
        par.log(f"  reference {tuple(float(x) for x in np.round(dd, 3))}: done ({time.time() - t0:.1f} s); "
                f"N_el(model) = {rd['nel']:.4f}")

    res = build_results(base, sites, refdata, refs, None, has_soc)
    res.extra["mu"] = mu
    res.extra["temperature"] = float(c["temperature"])
    res.extra["mag_pos"] = mag_pos
    res.extra["lig_pos"] = lig_pos
    res.extra["soc"] = soc
    res.extra["partition"] = c["partition"]
    if c["symmetrize"]:
        try:
            from .symmetry import symmetrize
            symmetrize(res, base)
        except ImportError:
            par.log("spglib not installed: skipping symmetrisation")

    # ---------------- Bruno / downfolding (isotropic channel, z or native reference)
    rd = refdata[-1]
    dref = np.asarray(refs[-1], float)
    corr = {}
    rigid = "sumrule" if c["partition"] == "onsite" else "T1"
    kw = dict(rigid=rigid)
    corr["mft"] = renormalize(rd, dref, mag_pos, lig_pos, bruno=False, downfold=False, **kw)
    if c["bruno"]:
        corr["bruno"] = renormalize(rd, dref, mag_pos, lig_pos, bruno=True, downfold=False, **kw)
    if c["downfold"] and lig_pos:
        corr["downfold"] = renormalize(rd, dref, mag_pos, lig_pos, bruno=False, downfold=True, **kw)
    res.extra["corrections"] = {k: v["J"] for k, v in corr.items()}
    res.extra["sumrule"] = corr["mft"]["sumrule"]

    # ---------------- Lloyd finite-angle analysis
    if lloyd:
        lk = res.pair_lookup()
        sgn = np.sign(rd["moments"] @ dref)
        sgn[sgn == 0] = 1
        Js = []
        for (i, j, nu) in ll_pairs:
            p = next(q for q in res.pairs if q.i == i and q.j == j and q.nu == nu)
            Js.append(p.J)
        ax = np.cross(dref, [1, 0, 0] if abs(dref[0]) < 0.9 else [0, 1, 0])
        res.extra["lloyd"] = lloyd_fit(rd["lloyd"], sgn, Js, dref, ax)

    res.extra["orbital"] = {}
    for key, v in refdata[-1]["orbital"].items():
        i, j, nu = key
        s = np.sign(rd["moments"] @ dref)
        u1, u2 = np.linalg.svd(np.eye(3) - np.outer(dref, dref))[0][:, :2].T
        Kt = 0.5 * (np.einsum("a,abmn,b->mn", u1, v, u1) + np.einsum("a,abmn,b->mn", u2, v, u2))
        res.extra["orbital"][key] = -Kt / (2 * s[i] * s[j]) * 1000.0

    # ---------------- MAE (global rotation, band grand potential)
    if c["mae"] and has_soc:
        dirs = np.vstack([default_directions(), -default_directions()])
        mesh_mae = tuple(c["mae_kmesh"] or mesh)
        Om = grand_potential_directions(base, dirs, mesh_mae, mu, kT) * 1000.0
        nd = len(dirs) // 2
        odd = 0.5 * (Om[:nd] - Om[nd:])                 # must vanish by time reversal
        even = 0.5 * (Om[:nd] + Om[nd:])
        A_dft, _, rms = fit_quadratic_anisotropy(dirs[:nd], even)
        A_heis, _, _ = fit_quadratic_anisotropy(dirs[:nd], heisenberg_anisotropy(res, dirs[:nd]))
        res.extra["mae"] = dict(directions=dirs[:nd], E=even - even.min(), A_band=A_dft, A_heisenberg=A_heis,
                                fit_rms=rms, mesh=mesh_mae, tr_odd_max=float(np.max(np.abs(odd))))
    res.extra["timing"] = dict(timing, total=time.time() - t0)
    if par.is_root():
        from .output import write_all
        write_all(res, base, cfg, outdir)
        par.log(f"results written to {outdir}/  ({time.time() - t0:.1f} s)")
    return res
